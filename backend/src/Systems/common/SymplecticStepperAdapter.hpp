#pragma once

#include <type_traits>

//
#include "Systems/Block/Block/TypeTraits.hpp"
#include "Systems/States/States.hpp"
#include "Systems/common/Tags.hpp"
#include "Systems/common/Types.hpp"
//
// #include "Time/SimulationTime.hpp"
//
#include "Utilities/TMPL.hpp"

namespace elastica {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  namespace detail {

    template <typename... X>
    struct Inherit : public X... {
      //**Block type definitions************************************************
      //! List of computed variables
      using ComputedVariables = tmpl::append<typename X::ComputedVariables...>;
      //! List of initialized variables
      using InitializedVariables =
          tmpl::append<typename X::InitializedVariables...>;
      //! List of all variables
      using Variables = tmpl::append<typename X::Variables...>;
      //************************************************************************

      template <class DownstreamBlock, class SystemInitializer>
      static void initialize(DownstreamBlock& downstream_block,
                             SystemInitializer&& initializers) {
        EXPAND_PACK_LEFT_TO_RIGHT(X::initialize(
            downstream_block, std::forward<SystemInitializer>(initializers)));
      }
    };

    struct DataCache {
      template <typename Var>
      struct AsCachedVariable {
        using type = std::add_lvalue_reference_t<typename Var::type>;
        using data_type = typename Var::type;
      };
    };
    struct ViewCache {
      template <typename Var>
      struct AsCachedVariable {
        using type = typename Var::SliceType::type;
        using data_type = type;
      };
    };
    struct ConstViewCache {
      template <typename Var>
      struct AsCachedVariable {
        using type = typename Var::ConstSliceType::type;
        using data_type = type;
      };
    };

    template <typename CacheType>
    struct Wrapped : public CacheType {
      template <typename Var>
      struct ToCachedVariable {
        using type = typename CacheType::template AsCachedVariable<Var>;
      };
    };

    template <typename B>
    using ChooseCacheView =
        std::conditional_t<::blocks::IsBlockSlice<B>::value ||
                               ::blocks::IsBlockView<B>::value,
                           ViewCache, ConstViewCache>;

    // The state mechanism relies on pointer semantics, so a pointer to a valid
    // address is necessary. This conflicts with the lazy slicing mechanism
    // used in block slices and views. To WAR this, we store the data here as
    // 1. Lvalue References in case of a block (which returns lvalue references
    // via get())
    // 2. Values in case of block slice/views (which returns rvalue references)
    // And use this cache store address to refer into the states mechanism.
    // It's a somewhat long-winded solution, but it has the benefit of not
    // modifying any state/SE3/SO3-logic.

    // TODO : can be optimized and refactored
    template <typename B>
    using ChooseCache =
        Wrapped<std::conditional_t<::blocks::IsBlock<B>::value, DataCache,
                                   ChooseCacheView<B>>>;

  }  // namespace detail

  //****************************************************************************
  /*!\brief Adapter for enabling symplectic temporal integration
   * \ingroup systems
   *
   * \details
   * Adapter for system block hierarchies (i.e. conforming to
   * protocols::PhysicalSystem) to interface with symplectic time integrators
   * steppers
   *
   * \tparam BlockCache The block (along with data)
   *
   * \example
   * \snippet Test_SymplecticStepperAdapter.cpp
   */
  template <typename Traits, typename BlockLike,
            template <typename, typename> class... OtherComponents>
  class SymplecticPolicy
      : public detail::Inherit<OtherComponents<Traits, BlockLike>...> {
   private:
    //**Type definitions********************************************************
    //! Parent type
    using P = detail::Inherit<OtherComponents<Traits, BlockLike>...>;
    //! This type
    using This = SymplecticPolicy<Traits, BlockLike, OtherComponents...>;
    //**************************************************************************

   private:
    //**************************************************************************
    // CRTP section
    // \note : cant use CRTP helper because it does not expect template
    // template parameters
    //! Type of the bottom level derived class
    using Self = BlockLike;
    //! Reference type of the bottom level derived class
    using Reference = Self&;
    //! const reference type of the bottom level derived class
    using ConstReference = Self const&;
    //**************************************************************************

   protected:
    //**Type definitions********************************************************
    //! List of computed variables
    using typename P::ComputedVariables;
    //! List of initialized variables
    using typename P::InitializedVariables;
    //! List of all variables
    using typename P::Variables;
    //**************************************************************************

    //**************************************************************************
    /*!\name Variable data */
    //@{
    using typename P::Acceleration;
    using typename P::AngularAcceleration;
    using typename P::AngularVelocity;
    using typename P::Director;
    using typename P::Position;
    using typename P::Velocity;
    //@}
    //**************************************************************************

   private:
    //**************************************************************************
    /*!\name State types */
    //@{
    using CacheChoice = detail::ChooseCache<BlockLike>;
    using StateVariables =
        tmpl::list<Position, Velocity, Acceleration, Director, AngularVelocity,
                   AngularAcceleration>;

    using CachedData = tmpl::as_tagged_tuple<tmpl::transform<
        StateVariables,
        typename CacheChoice::template ToCachedVariable<tmpl::_1>>>;

    using PositionData =
        typename CacheChoice::template AsCachedVariable<Position>::data_type;
    // std::remove_reference_t<decltype(std::declval<P>().get_position())>;
    using PositionState =
        ::elastica::states::SE3<PositionData,
                                ::elastica::states::tags::PrimitiveTag>;
    using VelocityData =
        typename CacheChoice::template AsCachedVariable<Velocity>::data_type;
    using VelocityState =
        ::elastica::states::SE3<VelocityData,
                                ::elastica::states::tags::DerivativeTag>;
    using AccelerationData = typename CacheChoice::template AsCachedVariable<
        Acceleration>::data_type;
    using AccelerationState =
        ::elastica::states::SE3<AccelerationData,
                                ::elastica::states::tags::DoubleDerivativeTag>;

    using DirectorData =
        typename CacheChoice::template AsCachedVariable<Director>::data_type;
    using DirectorState =
        ::elastica::states::SO3<DirectorData,
                                ::elastica::states::tags::PrimitiveTag>;
    using AngularVelocityData = typename CacheChoice::template AsCachedVariable<
        AngularVelocity>::data_type;
    using AngularVelocityState =
        ::elastica::states::SO3<AngularVelocityData,
                                ::elastica::states::tags::DerivativeTag>;

    using AngularAccelerationData =
        typename CacheChoice::template AsCachedVariable<
            AngularAcceleration>::data_type;
    using AngularAccelerationState =
        ::elastica::states::SO3<AngularAccelerationData,
                                ::elastica::states::tags::DoubleDerivativeTag>;

    //! Collection of all kinematic states
    using KinematicState =
        ::elastica::states::States<PositionState, DirectorState>;
    //! Collection of all dynamic states
    using DynamicState =
        ::elastica::states::States<VelocityState, AngularVelocityState>;
    //! Collection of all dynamic rates
    using DynamicRate =
        ::elastica::states::States<AccelerationState, AngularAccelerationState>;

    //@}
    //**************************************************************************

   protected:
    template <typename... Vars>
    static auto fill(Reference ref,
                     tmpl::list<Vars...> = StateVariables{}) noexcept {
      return CachedData(::blocks::get<::blocks::parameter_t<Vars>>(ref)...);
    }

    template <typename Var>
    inline constexpr decltype(auto) get_cached_variable() & noexcept {
      return tuples::get<typename CacheChoice::template AsCachedVariable<Var>>(
          cached_data_);
    }

    template <typename Var>
    inline constexpr decltype(auto) get_cached_variable() const& noexcept {
      return tuples::get<typename CacheChoice::template AsCachedVariable<Var>>(
          cached_data_);
    }

    static auto initialize_kinematic_states(This& dis) {
      return KinematicState(
          PositionState(&dis.template get_cached_variable<Position>()),
          DirectorState(&dis.template get_cached_variable<Director>()));
    }

    static auto initialize_dynamic_states(This& dis) {
      return DynamicState(
          VelocityState(&dis.template get_cached_variable<Velocity>()),
          AngularVelocityState(
              &dis.template get_cached_variable<AngularVelocity>()));
    }

    static auto initialize_dynamic_rates(This& dis) {
      return DynamicRate(
          AccelerationState(&dis.template get_cached_variable<Acceleration>()),
          AngularAccelerationState(
              &dis.template get_cached_variable<AngularAcceleration>()));
    }

    SymplecticPolicy()
        : P(),
          cached_data_(fill(self(), StateVariables{})),
          kinematic_states_(initialize_kinematic_states(*this)),
          dynamic_states_(initialize_dynamic_states(*this)),
          dynamic_rates_(initialize_dynamic_rates(*this)) {}

    SymplecticPolicy(SymplecticPolicy const& other)
        : P(other),
          cached_data_(fill(self(), StateVariables{})),
          kinematic_states_(initialize_kinematic_states(*this)),
          dynamic_states_(initialize_dynamic_states(*this)),
          dynamic_rates_(initialize_dynamic_rates(*this)) {}

    SymplecticPolicy(SymplecticPolicy&& other) noexcept
        : P(std::move(other)),
          cached_data_(fill(self(), StateVariables{})),
          kinematic_states_(initialize_kinematic_states(*this)),
          dynamic_states_(initialize_dynamic_states(*this)),
          dynamic_rates_(initialize_dynamic_rates(*this)) {}

   public:
    //**************************************************************************
    /*!\name CRTP methods */
    //@{

    //**Self method*************************************************************
    /*!\brief Access to the underlying derived
     *
     * \return Mutable reference to the underlying derived
     *
     * Safely down-casts this module to the underlying derived type, using
     * the Curiously Recurring Template Pattern (CRTP).
     */
    inline constexpr auto self() & noexcept -> Reference {
      return static_cast<Reference>(*this);
    }
    //**************************************************************************

    //**Self method*************************************************************
    /*!\brief Access to the underlying derived
     *
     * \return Const reference to the underlying derived
     *
     * Safely down-casts this module to the underlying derived type, using
     * the Curiously Recurring Template Pattern (CRTP).
     */
    inline constexpr auto self() const& noexcept -> ConstReference {
      return static_cast<ConstReference>(*this);
    }
    //**************************************************************************

    //@}
    //**************************************************************************

   public:
    // Do not need const overloads : const here should be a hard compile error
    //**************************************************************************
    /*!\name State functions */
    //@{

    //**************************************************************************
    /*!\brief Access to kinematic states
     */
    inline auto kinematic_states() & noexcept -> KinematicState& {
      return kinematic_states_;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to kinematic rates
     */
    // inline auto kinematic_rates(::elastica::Time /*time_v*/) const noexcept
    inline auto kinematic_rates(double /*time_v*/) const noexcept
        -> DynamicState const& {
      return dynamic_states();
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to dynamic states
     */
    inline auto dynamic_states() & noexcept -> DynamicState& {
      return dynamic_states_;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to immutable dynamic states
     */
    inline auto dynamic_states() const& noexcept -> DynamicState const& {
      return dynamic_states_;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to dynamic rates
     */
    // inline auto dynamic_rates(::elastica::Time /*time_v*/) /*const*/ noexcept(
    inline auto dynamic_rates(double /*time_v*/) /*const*/ noexcept(
        noexcept(update_acceleration(self()))) -> DynamicRate const& {
      /* To be truly lazy, one should only compute acceleration
       * data and internal dynamics here.
       * but the cosserat rods need precomputing to account
       * for environment effects etc. so we can skip that here
       */
      update_acceleration(self());
      return dynamic_rates_;
    }
    //**************************************************************************

    //@}
    //**************************************************************************

   private:
    //! Cache information of the system to populate the states below.
    CachedData cached_data_;
    //! Kinematic state information of the system.
    KinematicState kinematic_states_;
    //! Dynamic state information of the system.
    DynamicState dynamic_states_;
    //! Dynamic rate information of the system.
    DynamicRate dynamic_rates_;
  };
  //****************************************************************************

}  // namespace elastica

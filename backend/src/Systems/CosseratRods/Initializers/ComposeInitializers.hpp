#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <tuple>

#include "Systems/CosseratRods/Initializers/detail/Affordances.hpp"
#include "Utilities/TMPL.hpp"

namespace elastica {

  namespace cosserat_rod {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Compose initializers from different components of a Cosserat rod
     * \ingroup cosserat_rod
     *
     * \details
     * ComposeInitializers provides a facade for quickly creating client-facing
     * initializers (i.e. those which accept POD types rather than lambda types)
     * from their simplest possible definitions.
     *
     * \tparam ModularizedInitializers The modularized initializers found in
     * ::elastica::cosserat_rod::Initializers
     */
    template <typename... ModularizedInitializers>
    class ComposeInitializers
        : public ModularizedInitializers...,
          public detail::CosseratRodInitializerInterface<
              ComposeInitializers<ModularizedInitializers...>> {
     protected:
      //**Type definitions******************************************************
      //! This type
      using This = ComposeInitializers<ModularizedInitializers...>;
      //! The interface type
      using Interface = detail::CosseratRodInitializerInterface<This>;
      //! The types needed to be friended
      using typename Interface::Friends;
      //************************************************************************

      static constexpr std::size_t n_friends() {
        return tmpl::size<Friends>::value;
      }

      //**Friendships***********************************************************
      // Friend the Interface
      friend Interface;
      // friend the interface and its friends for access

#define REPEAT_2(M, N) M(N) M(((N) + 1))
#define REPEAT_4(M, N) REPEAT_2(M, N) REPEAT_2(M, ((N) + 2))
#define REPEAT_8(M, N) REPEAT_4(M, N) REPEAT_4(M, ((N) + 4))
#define FRIEND(N)                                    \
  friend tmpl::at_c<tmpl::push_front<Friends, void>, \
                    std::min((N + 1), n_friends())>;

      // Declare friendship with each interface so that they have access
      // to protected functions
      REPEAT_8(FRIEND, 0UL)
      // friend elastica::io::Serialize<This>;
      //************************************************************************

#undef FRIEND
#undef REPEAT_8
#undef REPEAT_4
#undef REPEAT_2

     public:
      //**Type definitions******************************************************
      //! Typelist of required parameters for initialization
      using RequiredParameters =
          tmpl::append<typename ModularizedInitializers::RequiredParameters...>;
      //! Typelist of default parameters automatically filled in while
      //! initialization
      using DefaultParameters =
          tmpl::append<typename ModularizedInitializers::DefaultParameters...>;
      //! The default initializer types
      using DefaultInitializers =
          tmpl::transform<DefaultParameters,
                          tmpl::bind<tmpl::type_from, tmpl::_1>>;
      //! The type of cache
      using OptionsCache = tmpl::as_tuple<RequiredParameters>;
      //! The type of cache for defatul initializers
      using DefaultInitializersCache = tmpl::as_tuple<DefaultInitializers>;
      //************************************************************************

      //**Constructors**********************************************************
      /*!\name Constructors */
      //@{
     private:
      //************************************************************************
      /*!\brief Delegated constructor
       *
       * \param tup The tuple to construct the required parameter cache from
       */
      template <typename... Args, typename... RequiredTags,
                typename... DefaultTags>
      ComposeInitializers(std::tuple<Args...> tup,
                          tmpl::list<RequiredTags...> /*meta*/,
                          tmpl::list<DefaultTags...> /* meta*/)
          : ModularizedInitializers()...,
            Interface(),
            cache_(std::get<RequiredTags>(std::move(tup))...),
            default_initializers_cache_((std::move(DefaultTags{}.data))...) {}
      //************************************************************************

     public:
      //************************************************************************
      /*!\brief Tuple constructor
       *
       * \param tup The tuple to construct the required parameter cache from
       */
      template <typename... Args>
      explicit ComposeInitializers(std::tuple<Args...> tup)
          : ComposeInitializers(std::move(tup), RequiredParameters{},
                                DefaultParameters{}) {}
      //************************************************************************

      //************************************************************************
      /*!\brief Forwarding constructor
       *
       * \param farg First argument to be forwarded
       * \param sarg Second argument to be forwarded
       * \param Args Other arguments to be forwarded
       */
      template <typename First, typename Second, typename... Args>
      explicit ComposeInitializers(First&& farg, Second&& sarg, Args&&... args)
          : ComposeInitializers(std::make_tuple(std::forward<First>(farg),
                                                std::forward<Second>(sarg),
                                                std::forward<Args>(args)...)) {}
      //************************************************************************

      //@}
      //************************************************************************

     protected:
      //**Implementation
      // details*************************************************
      /*!\name Implementation details */
      //@{

      //************************************************************************
      /*!\brief Gets the required parameters as a tuple of initializers
       */
      decltype(auto) get_required_parameters() const /*noexcept*/ {
        return std::tuple_cat(
            ModularizedInitializers::get_required_parameters(data())...);
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Gets the optional parameters as a tuple of initializers
       */
      using Interface::get_optional_parameters;
      //************************************************************************

      //@}
      //************************************************************************

     public:
      //**Data access***********************************************************
      /*!\name Data access */
      //@{

      //************************************************************************
      /*!\brief Access to the underlying required parameter data
       *
       * \return Mutable lvalue reference to the underlying required parameter
       * data
       */
      auto data() & noexcept -> OptionsCache& { return cache_; }
      //************************************************************************

      //************************************************************************
      /*!\brief Access to the underlying required parameter data
       *
       * \return Constant lvalue reference to the underlying required parameter
       * data
       */
      auto data() const& noexcept -> OptionsCache const& { return cache_; }
      //************************************************************************

      //************************************************************************
      /*!\brief Access to the underlying default initializers data
       *
       * \return Constant lvalue reference to the underlying default parameter
       * data
       */
      auto default_initializers() const& noexcept
          -> DefaultInitializersCache const& {
        return default_initializers_cache_;
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Access to the underlying default parameter data
       *
       * \return Mutable lvalue reference to the underlying default parameter
       * data
       */
      auto default_initializers() & noexcept -> DefaultInitializersCache& {
        return default_initializers_cache_;
      }
      //************************************************************************

      //@}
      //************************************************************************

     private:
      //**Member variables******************************************************
      /*!\name Member variables */
      //@{
      //! All required variables
      OptionsCache cache_;
      //! All default initializers
      DefaultInitializersCache default_initializers_cache_;
      //@}
      //************************************************************************
    };
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

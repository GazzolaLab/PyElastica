#pragma once

//******************************************************************************
// Includes
//******************************************************************************
///// Types always first
#include "Systems/CosseratRods/Components/Elasticity/detail/Types.hpp"
/////
#include "Systems/Block.hpp"
//
#include "Systems/CosseratRods/Components/Elasticity/detail/Tags/ExplicitDampingAdapterFacadeTags.hpp"
#include "Systems/CosseratRods/Components/Initialization.hpp"
#include "Systems/CosseratRods/Components/Noexcept.hpp"
#include "Systems/CosseratRods/Components/Tags.hpp"
//
#include "Utilities/CRTP.hpp"
#include "Utilities/Size.hpp"
#include "Utilities/TMPL.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      namespace detail {

        //**********************************************************************
        /*!\brief Component for creating external Damping
         * \ingroup cosserat_rod_component
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         * \tparam ComputationalBlock The final block which is derived from the
         * current component
         * \tparam ElasticityModelWithBlock Elasticity model
         * \tparam VariableInfo Variables for stamping classes
         */
        template <typename CRT,                 // Cosserat Rod Traits
                  typename ComputationalBlock,  // Block
                  class ElasticityModelWithBlock,
                  template <typename /* CRT */> class VariableInfo>
        class ExplicitDampingAdapterFacade : public ElasticityModelWithBlock,
                                             public VariableInfo<CRT> {
         private:
          //**Type definitions**************************************************
          //! Traits type
          using Traits = CRT;
          //! Variable definitions
          using VariableDefinitions = VariableInfo<Traits>;
          //! This type
          using This = ExplicitDampingAdapterFacade<Traits, ComputationalBlock,
                                                    ElasticityModelWithBlock,
                                                    VariableInfo>;
          //! Parent type
          using Parent = ElasticityModelWithBlock;

         protected:
          //! List of initialized variables
          using InitializedVariables =
              tmpl::append<typename Parent::InitializedVariables,
                           typename VariableDefinitions::InitializedVariables>;
          //! List of computed variables
          using ComputedVariables =
              tmpl::append<typename Parent::ComputedVariables,
                           typename VariableDefinitions::ComputedVariables>;
          //! List of all variables
          using Variables =
              tmpl::append<typename Parent::Variables,
                           typename VariableDefinitions::Variables>;
          //! Variables in this class
          using typename VariableDefinitions::ForceDampingRate;
          using typename VariableDefinitions::TorqueDampingRate;
          //********************************************************************

         protected:
          //        // Here instead of doing
          //         using Interface = typename Parent::Interface;
          //        // we do the following
          //        // this is because in the earlier case the CRTP class to the
          //        Interface
          //        // is the Parent, and only its
          //        compute_internal_modeled_loads_impl()
          //        // will be called. Rather now the current class's
          //        // compute_internal_modeled_loads_impl() will be called
          //        // using Interface = ElasticityInterface<Traits, This>;
          //        // correspondingly for unambigious recognition of the
          //        compute_internal_forces
          //        // member we bring it explicitly into this namespace
          ////       public:
          ////        using Interface::compute_internal_forces;
          ////        using Interface::compute_internal_torques;

         private:
          //********************************************************************
          // CRTP section
          // \note : cant use CRTP helper because it does not expect template
          // template parameters
          //! Type of the bottom level derived class
          using Self = ComputationalBlock;
          //! Reference type of the bottom level derived class
          using Reference = Self&;
          //! const reference type of the bottom level derived class
          using ConstReference = Self const&;
          //********************************************************************

         public:
          //**Self method*******************************************************
          /*!\brief Access to the underlying derived
          //
          // \return Mutable reference to the underlying derived
          //
          // Safely down-casts this module to the underlying derived type, using
          // the Curiously Recurring Template Pattern (CRTP).
          */
          inline constexpr auto self() & noexcept -> Reference {
            return static_cast<Reference>(*this);
          }
          //********************************************************************

          //**Self method*******************************************************
          /*!\brief Access to the underlying derived
          //
          // \return Const reference to the underlying derived
          //
          // Safely down-casts this module to the underlying derived type, using
          // the Curiously Recurring Template Pattern (CRTP).
          */
          inline constexpr auto self() const& noexcept -> ConstReference {
            return static_cast<ConstReference>(*this);
          }
          //********************************************************************

         protected:
          //********************************************************************
          /*!\copydoc ComponentInitializationDocStub
           */
          template <typename BlockLike, typename ElasticityModel,
                    typename CosseratInitializer>
          static void initialize(
              ExplicitDampingAdapterFacade<Traits, BlockLike, ElasticityModel,
                                           VariableInfo>& this_component,
              CosseratInitializer&& initializer) {
            // 1. Initializer parent's first
            Parent::initialize(this_component,
                               std::forward<CosseratInitializer>(initializer));

            // 2. Initialize required variables
            initialize_component<
                typename VariableDefinitions::InitializedVariables>(
                this_component.self(),
                std::forward<CosseratInitializer>(initializer));
          }
          //********************************************************************
        };
        //**********************************************************************

      }  // namespace detail

    }  // namespace component

    //**************************************************************************
    /*!\brief Computes the internal loads modeled by the elasticity component
     * \ingroup cosserat_rod_custom_entries
     */
    template <typename Traits,              // Cosserat Rod Traits
              typename ComputationalBlock,  // Block
              class ElasticityModelWithBlock,
              template <typename /* CRT */> class VariableInfo>
    void compute_internal_modeled_loads_impl(
        component::detail::ExplicitDampingAdapterFacade<
            Traits, ComputationalBlock, ElasticityModelWithBlock, VariableInfo>&
            block_like) COSSERATROD_LIB_NOEXCEPT {
      auto&& internal_force(
          blocks::get<tags::InternalLoads>(block_like.self()));
      auto&& dissipation_constant(
          blocks::get<tags::ForceDampingRate>(block_like.self()));
      auto&& mass(blocks::get<tags::Mass>(block_like.self()));
      auto&& velocity(blocks::get<tags::Velocity>(block_like.self()));

      // static dispatch to parent type
      compute_internal_modeled_loads_impl(
          static_cast<ElasticityModelWithBlock&>(block_like));

      internal_force -= Traits::Operations::transpose(
                            Traits::Operations::expand_for_broadcast(
                                mass * dissipation_constant)) %
                        velocity;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Computes the internal torques modeled by the elasticity component
     * \ingroup cosserat_rod_custom_entries
     */
    template <typename Traits, typename ComputationalBlock,
              class ElasticityModelWithBlock,
              template <typename /* Traits */> class VariableInfo>
    void compute_internal_modeled_torques_impl(
        component::detail::ExplicitDampingAdapterFacade<
            Traits, ComputationalBlock, ElasticityModelWithBlock, VariableInfo>&
            block_like) COSSERATROD_LIB_NOEXCEPT {
      auto&& internal_torque(
          blocks::get<tags::InternalTorques>(block_like.self()));
      auto&& dissipation_constant(
          blocks::get<tags::TorqueDampingRate>(block_like.self()));
      auto&& angular_velocity(
          blocks::get<tags::AngularVelocity>(block_like.self()));
      auto&& mass(blocks::get<tags::Mass>(block_like.self()));

      auto&& elemental_mass = Traits::Operations::average_kernel(mass);

      // static dispatch to parent type
      compute_internal_modeled_torques_impl(
          static_cast<ElasticityModelWithBlock&>(block_like));

      // do not need ghosts here since dissipation constant in zero in
      // element ghosts
      // may need evaluation, but we don't have a dummy allocated
      internal_torque -= Traits::Operations::transpose(
                             Traits::Operations::expand_for_broadcast(
                                 elemental_mass * dissipation_constant)) %
                         angular_velocity;
    }
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

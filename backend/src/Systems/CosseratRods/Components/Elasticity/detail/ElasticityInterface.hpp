#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

///// Types always first
#include "Systems/CosseratRods/Components/Elasticity/detail/Types.hpp"
/////
#include "Systems/Block.hpp"
#include "Systems/CosseratRods/Components/Elasticity/detail/Tags/ElasticityInterfaceTags.hpp"
#include "Systems/CosseratRods/Components/Noexcept.hpp"
//
// #include "Time/SimulationTime.hpp"
//
#include "Utilities/CRTP.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsDetected.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      namespace detail {

        //======================================================================
        //
        //  CLASS DEFINITION
        //
        //======================================================================

        //**********************************************************************
        /*!\brief Variables corresponding to the elasticity interface component
         * \ingroup cosserat_rod_component
         *
         * \details
         * ElasticityInterfaceVariables contains the definitions of
         * variables used within the Blocks framework for a 1D elastic model
         * with an interface expected as by other components.
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         *
         * \see ElasticityInterface
         */
        template <typename CRT>
        class ElasticityInterfaceVariables {
         private:
          //**Type definitions**************************************************
          //! Traits type
          using Traits = CRT;
          //********************************************************************

         protected:
          //**Variable definitions**********************************************
          /*!\name Variable definitions*/
          //@{

          //********************************************************************
          /*!\brief Variable marking InternalLoads within the Cosserat rod
           * hierarchy
           */
          struct InternalLoads : public Traits::template CosseratRodVariable<
                                     ::elastica::tags::InternalLoads,    //
                                     typename Traits::DataType::Vector,  //
                                     typename Traits::Place::OnNode> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking InternalStress within the Cosserat rod
           * hierarchy
           */
          struct InternalStress : public Traits::template CosseratRodVariable<
                                      ::elastica::tags::InternalStress,   //
                                      typename Traits::DataType::Vector,  //
                                      typename Traits::Place::OnElement> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking InternalTorques within the Cosserat rod
           * hierarchy
           */
          struct InternalTorques : public Traits::template CosseratRodVariable<
                                       ::elastica::tags::InternalTorques,  //
                                       typename Traits::DataType::Vector,  //
                                       typename Traits::Place::OnElement> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking InternalCouple within the Cosserat rod
           * hierarchy
           */
          struct InternalCouple : public Traits::template CosseratRodVariable<
                                      ::elastica::tags::InternalCouple,   //
                                      typename Traits::DataType::Vector,  //
                                      typename Traits::Place::OnVoronoi> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Temporary vector and matrix for dummy purpose
           * Avoid temp memory allocations. Use only if must
           *  - Some exception: branch of operation
           */
          struct _DummyElementVector
              : public Traits::template CosseratRodVariable<
                    ::elastica::tags::_DummyElementVector,
                    typename Traits::DataType::Vector,
                    typename Traits::Place::OnElement> {};
          struct _DummyElementVector2
              : public Traits::template CosseratRodVariable<
                    ::elastica::tags::_DummyElementVector2,
                    typename Traits::DataType::Vector,
                    typename Traits::Place::OnElement> {};
          struct _DummyVoronoiVector
              : public Traits::template CosseratRodVariable<
                    ::elastica::tags::_DummyVoronoiVector,
                    typename Traits::DataType::Vector,
                    typename Traits::Place::OnVoronoi> {};
          //********************************************************************

          //@}
          //********************************************************************

          //**Type definitions**************************************************
          //! List of initialized variables
          using InitializedVariables = tmpl::list<>;
          //! List of computed variables
          using ComputedVariables =
              tmpl::list<InternalLoads, InternalTorques, InternalStress,
                         InternalCouple, _DummyElementVector,
                         _DummyElementVector2, _DummyVoronoiVector>;
          //! List of all variables
          using Variables =
              tmpl::append<InitializedVariables, ComputedVariables>;
          //********************************************************************
        };

        //======================================================================
        //
        //  CLASS DEFINITION
        //
        //======================================================================

        //**********************************************************************
        /*!\brief Interface component corresponding to an elasticity model
         * \ingroup cosserat_rod_component
         *
         * \details
         * ElasticityInterface implements the final Elasticity component use
         * within Cosserat rods implemented in the Blocks framework. It denotes
         * an Elasticity model that operates on a 1-D parametrized entity
         * spanning a single spatial dimension (i.e. along center-line
         * coordinates, such as a Cosserat rod data-structure) and gives the
         * loads and torques. By design it only binds to classes with signature
         * of a Component
         *
         * \usage
         * Since ElasticityInterface is useful for wrapping an ElasticityModel
         * with an interface, one should use it using the Adapt class as follows
         * \code
         * template <typename Traits, typename Block>
         * class MyCustomElasticityModel : public
         * Component<MyCustomElasticityModel<Traits, Block>> {
         *   // ...implementation
         * };
         *
         * template <typename Traits, typename Block>
         * class MyCustomElasticityModelWithTheCorrectInterface : public
         * Adapt<MyCustomElasticityModel<CRT, ComputationalBlock>>::
         *      template with<ElasticityInterface> {};
         * \endcode
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         * \tparam ComputationalBlock The final block which is derived from the
         * current component
         * \tparam ElasticityModelParam The user defined elasticity model which
         * is to be wrapped in the interface type
         *
         * \see CosseratRodCrossSectionInterfaceVariables
         */
        template <typename CRT, typename ComputationalBlock,
                  template <typename /*CRT*/, typename /* ComputationalBlock*/,
                            typename... /*ElasticityModelMetaArgs*/>
                  class ElasticityModelParam,
                  typename... ElasticityModelMetaArgs>
        class ElasticityInterface<
            CRT, ComputationalBlock,
            ElasticityModelParam<CRT, ComputationalBlock,
                                 ElasticityModelMetaArgs...>>
            : public ElasticityInterfaceVariables<CRT>,
              public ElasticityModelParam<CRT, ComputationalBlock,
                                          ElasticityModelMetaArgs...>,
              public CRTPHelper<ComputationalBlock, ElasticityInterface> {
         private:
          //**Type definitions**************************************************
          //! Traits type
          using Traits = CRT;
          //! Parent type
          using Parent = ElasticityModelParam<Traits, ComputationalBlock,
                                              ElasticityModelMetaArgs...>;
          //! This type
          using This = ElasticityInterface<Traits, ComputationalBlock, Parent>;
          //! Type defining variables
          using VariableDefinitions = ElasticityInterfaceVariables<CRT>;
          //! CRTP type
          using CRTP = CRTPHelper<ComputationalBlock, ElasticityInterface>;
          //! Index type
          using index_type = typename Traits::index_type;
          //********************************************************************

         public:
          //**Type definitions**************************************************
          //! Model type
          using ElasticityModel = Parent;
          //********************************************************************

         protected:
          //**Type definitions**************************************************
          //! List of initialized variables
          using InitializedVariables =
              tmpl::append<typename Parent::InitializedVariables,
                           typename VariableDefinitions::InitializedVariables>;
          //! List of computed variables
          using ComputedVariables =
              tmpl::append<typename ElasticityModel::ComputedVariables,
                           typename VariableDefinitions::ComputedVariables>;
          //! List of all variables
          using Variables =
              tmpl::append<typename ElasticityModel::Variables,
                           typename VariableDefinitions::Variables>;
          //! Variables from definitions
          using typename VariableDefinitions::InternalCouple;
          using typename VariableDefinitions::InternalLoads;
          using typename VariableDefinitions::InternalStress;
          using typename VariableDefinitions::InternalTorques;
          //********************************************************************

         protected:
          //********************************************************************
          /*!\copydoc ComponentInitializationDocStub
           */
          // Model params may have a block-slice inside, so to be safe
          // we add another template parameter here instead of using
          // ElasticityModelParam<Traits, BlockLike,
          //                      ElasticityModelMetaArgs...>
          template <typename BlockLike, typename CosseratInitializer,
                    typename ElasticityModel>
          static void initialize(
              ElasticityInterface<Traits, BlockLike, ElasticityModel>&
                  this_component,
              CosseratInitializer&& initializer) {
            // 1. Initialize parent
            Parent::initialize(this_component,
                               std::forward<CosseratInitializer>(initializer));

            // 2. Only computed variables here
            tmpl::for_each<typename VariableDefinitions::ComputedVariables>(
                [&this_component](auto v) {
                  using Var = tmpl::type_from<decltype(v)>;
                  using Tag = blocks::parameter_t<Var>;
                  blocks::get<Tag>(this_component.self()) = 0.0;
                });
          }
          //********************************************************************

         public:
          //**CRTP method*******************************************************
          /*!\name CRTP method*/
          //@{
          using CRTP::self;
          //@}
          //********************************************************************

          //**Get methods*******************************************************
          /*!\name Get methods*/
          //@{

#define STAMP_GETTERS(Var, func)                                         \
  inline constexpr decltype(auto) func()& noexcept {                     \
    return blocks::get<::elastica::tags::Var>(self());                   \
  }                                                                      \
  inline constexpr decltype(auto) func() const& noexcept {               \
    return blocks::get<::elastica::tags::Var>(self());                   \
  }                                                                      \
  inline constexpr decltype(auto) func(index_type idx)& noexcept {       \
    return Var::slice(func(), idx);                                      \
  }                                                                      \
  inline constexpr decltype(auto) func(index_type idx) const& noexcept { \
    return Var::slice(func(), idx);                                      \
  }                                                                      \
          //********************************************************************
          /*!\name internal couple methods*/
          //@{
          /*!\brief Gets internal couples of the rod
           */
          STAMP_GETTERS(InternalCouple, get_internal_couple)
          //@}
          //********************************************************************

          //********************************************************************
          /*!\name internal load methods*/
          //@{
          /*!\brief Gets internal loads of the rod
           */
          STAMP_GETTERS(InternalLoads, get_internal_loads)
          //@}
          //********************************************************************

          //********************************************************************
          /*!\name internal stress methods*/
          //@{
          /*!\brief Gets internal stress of the rod
           */
          STAMP_GETTERS(InternalStress, get_internal_stress)
          //@}
          //********************************************************************

          //********************************************************************
          /*!\name internal stress methods*/
          //@{
          /*!\brief Gets internal stress of the rod
           */
          STAMP_GETTERS(InternalTorques, get_internal_torques)
          //@}
          //********************************************************************
#undef STAMP_GETTERS

          //@}
          //********************************************************************

         public:
          //**Dynamic methods***************************************************
          /*!\name Dynamics methods*/
          //@{

          //********************************************************************
          /*!\brief Compute internal dynamics of the model
           *
           * \details
           * Interface function for use by simulator.
           *
           * \return void/None
           *
           */
          void compute_internal_dynamics() &
              COSSERATROD_LIB_NOEXCEPT {
            compute_internal_forces(self());
            compute_internal_torques(self());
          }
          //********************************************************************

          //@}
          //********************************************************************
        };
        //**********************************************************************

      }  // namespace detail

    }  // namespace component

    //**************************************************************************
    /*!\brief Computers internal force from the elasticity model
     * \ingroup cosserat_rod_custom_entries
     *
     * \details
     * Computes internal forces (shear force + damping force) by calling
     * compute_shear_strain() and compute_internal_modeled_loads_impl()
     *
     * Ref: eq (3.8) from RSOS
     *
     * \return void/None
     *
     * \see fill later?
     */
    template <typename Traits, typename ComputationalBlock,
              class ElasticityModel>
    inline void compute_internal_forces(
        component::detail::ElasticityInterface<Traits, ComputationalBlock,
                                               ElasticityModel>& block_like)
        COSSERATROD_LIB_NOEXCEPT {
      // TODO : Refactor after Model is fixed.
      compute_internal_modeled_loads_impl(block_like.self());
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Computers internal couple from the model
     * \ingroup cosserat_rod_custom_entries
     *
     * \details
     * Computes internal couple by calling
     * compute_curvature() and compute_internal_modeled_torques_impl()
     *
     * \return void/None
     */
    // FIXME : remove
    template <typename Traits, typename ComputationalBlock,
              class ElasticityModel>
    inline void compute_internal_torques(
        component::detail::ElasticityInterface<Traits, ComputationalBlock,
                                               ElasticityModel>& block_like)
        COSSERATROD_LIB_NOEXCEPT {
      compute_internal_modeled_torques_impl(block_like.self());
    }
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

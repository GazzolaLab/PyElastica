#pragma once

//******************************************************************************
// Includes
//******************************************************************************
///// Types always first
#include "Systems/CosseratRods/Components/Elasticity/detail/Types.hpp"
/////
#include "Systems/Block.hpp"
#include "Systems/CosseratRods/Components/Elasticity/detail/ExplicitDampingAdapterFacade.hpp"
#include "Systems/CosseratRods/Components/Elasticity/detail/TypeTraits.hpp"
#include "Systems/CosseratRods/Components/helpers.hpp"
#include "Utilities/CRTP.hpp"
#include "Utilities/TMPL.hpp"

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
        /*!\brief Variables corresponding to the explicit damping adapter
         * \ingroup cosserat_rod_component
         *
         * \details
         * ExplicitDampingAdapterVariables contains the definitions of
         * variables used within the Blocks framework for adapting a 1D elastic
         * model with explicit damping.
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         *
         * \see ExplicitDampingAdapter
         */
        template <typename CRT>
        struct ExplicitDampingAdapterVariables {
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
          /*!\brief Variable marking a scalar coefficient for damping forces
           * within the Cosserat rod hierarchy
           */
          struct ForceDampingRate
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::ForceDampingRate,  //
                    typename Traits::DataType::Scalar,   //
                    typename Traits::Place::OnNode> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking a scalar coefficient for damping torques
           * within the Cosserat rod hierarchy
           */
          struct TorqueDampingRate
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::TorqueDampingRate,  //
                    typename Traits::DataType::Scalar,    //
                    typename Traits::Place::OnElement> {};
          //********************************************************************

          //@}
          //********************************************************************

          //**Type definitions**************************************************
          //! List of computed variables
          using ComputedVariables = tmpl::list<>;
          //! List of initialized variables
          using InitializedVariables =
              tmpl::list<ForceDampingRate, TorqueDampingRate>;
          //! List of all variables
          using Variables =
              tmpl::append<InitializedVariables, ComputedVariables>;
          //********************************************************************
        };
        //**********************************************************************

        //======================================================================
        //
        //  CLASS DEFINITION
        //
        //======================================================================

        //**********************************************************************
        /*!\brief Adapter for adding explicit damping to an elasticity component
         * \ingroup cosserat_rod_component
         *
         * \details
         * ExplicitDampingAdapter adapts an Elasticity component for use
         * within Cosserat rods implemented in the Blocks framework. It adds
         * an additional damping force and torque to those arising from the
         * Elasticity model. It assumes that the Elasticity model operates on a
         * 1-D parametrized entity spanning a single spatial dimension (i.e.
         * along center-line coordinates, such as a Cosserat rod
         * data-structure). By design it only binds to classes with signature of
         * a Component.
         *
         * \usage
         * Since ExplicitDampingAdapter is useful for wrapping an
         * ElasticityModel with additional damping, one should use it using the
         * Adapt class as follows:
         *
         * \code
         * template <typename Traits, typename Block>
         * class MyCustomElasticityModel : public
         * Component<MyCustomElasticityModel<Traits, Block>> {
         *   // ...implementation
         * };
         *
         * template <typename Traits, typename Block>
         * class MyCustomElasticityModelWithDamping : public
         * Adapt<MyCustomElasticityModel<CRT, ComputationalBlock>>::
         *      template with<ExplicitDampingAdapter> {};
         * \endcode
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         * \tparam ComputationalBlock The final block which is derived from the
         * current component
         * \tparam ElasticityModelParam The user defined elasticity model which
         * is to be wrapped in the interface type
         *
         * \see ExplicitDampingAdapterVariables
         */
        template <typename CRT, typename ComputationalBlock,
                  template <typename /*CRT*/, typename /* ComputationalBlock*/,
                            typename... /*ElasticityModelMetaArgs*/>
                  class ElasticityModelParam,
                  typename... ElasticityModelMetaArgs>
        class ExplicitDampingAdapter<
            CRT, ComputationalBlock,
            ElasticityModelParam<CRT, ComputationalBlock,
                                 ElasticityModelMetaArgs...>>
            : public ExplicitDampingAdapterFacade<
                  CRT, ComputationalBlock,
                  // if interface passed in by mistake, doesn't derive from the
                  // interface, but rather from the implementation, this way
                  tt::elasticity_component_trait_t<ElasticityModelParam<
                      CRT, ComputationalBlock, ElasticityModelMetaArgs...>>,
                  detail::ExplicitDampingAdapterVariables> {
         private:
          //**Type definitions**************************************************
          //! Traits type
          using Traits = CRT;
          //! The type of the elasticity model
          using ThisModel = ElasticityModelParam<Traits, ComputationalBlock,
                                                 ElasticityModelMetaArgs...>;
          //! Base model (not need this complexity)
          using BaseModel = tt::elasticity_component_trait_t<ThisModel>;
          //! Parent type
          using Parent = ExplicitDampingAdapterFacade<
              CRT, ComputationalBlock, BaseModel,
              detail::ExplicitDampingAdapterVariables>;
          //! This type
          using This =
              ExplicitDampingAdapter<Traits, ComputationalBlock, ThisModel>;
          //********************************************************************

         protected:
          //**Type definitions**************************************************
          //! List of computed variables
          using typename Parent::ComputedVariables;
          //! List of initialized variables
          using typename Parent::InitializedVariables;
          //! List of all variables
          using typename Parent::Variables;
          //********************************************************************

          //**Parent methods****************************************************
          //! Initialize method inherited from parent class
          using Parent::initialize;
          //********************************************************************
        };
        //**********************************************************************

        //======================================================================
        //
        //  CLASS DEFINITION
        //
        //======================================================================

        //**********************************************************************
        /*!\brief Variables corresponding to the explicit damping adapter, but
         * distributed over a rod
         * \ingroup cosserat_rod_component
         *
         * \details
         * ExplicitDampingAdapterVariablesPerRod contains the definitions of
         * variables used within the Blocks framework for adapting a 1D elastic
         * model with explicit damping. Instead of having damping for every
         * grid node/element, it reserves the same damping for an entire rod
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         *
         * \see ExplicitDampingAdapterPerRod
         */
        template <typename CRT>
        struct ExplicitDampingAdapterVariablesPerRod {
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
          /*!\brief Variable marking a scalar coefficient for damping forces
           * within the Cosserat rod hierarchy
           */
          struct ForceDampingCoefficient
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::ForceDampingRate,  //
                    typename Traits::DataType::Scalar,   //
                    typename Traits::Place::OnRod> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking a scalar coefficient for damping torques
           * within the Cosserat rod hierarchy
           */
          struct TorqueDampingCoefficient
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::TorqueDampingRate,  //
                    typename Traits::DataType::Scalar,    //
                    typename Traits::Place::OnRod> {};
          //********************************************************************

          //@}
          //********************************************************************

          //**Type definitions**************************************************
          //! List of computed variables
          using ComputedVariables = tmpl::list<>;
          //! List of initialized variables
          using InitializedVariables =
              tmpl::list<ForceDampingCoefficient, TorqueDampingCoefficient>;
          //! List of all variables
          using Variables =
              tmpl::append<InitializedVariables, ComputedVariables>;
          //********************************************************************
        };
        //**********************************************************************

        //======================================================================
        //
        //  CLASS DEFINITION
        //
        //======================================================================

        //**********************************************************************
        /*!\brief Adapter for adding explicit damping to an elasticity component
         * \ingroup cosserat_rod_component
         *
         * \details
         * ExplicitDampingAdapterPerRod adapts an Elasticity component for use
         * within Cosserat rods implemented in the Blocks framework. It adds
         * an additional damping force and torque to those arising from the
         * Elasticity model. It assumes that the Elasticity model operates on a
         * 1-D parametrized entity spanning a single spatial dimension (i.e.
         * along center-line coordinates, such as a Cosserat rod
         * data-structure). By design it only binds to classes with signature of
         * a Component. Note that instead of having damping for every grid
         * node/element (which is achieved by ExplicitDampingAdapter) it
         * reserves the same damping for an entire rod.
         *
         * \usage
         * Since ExplicitDampingAdapterPerRod is useful for wrapping an
         * ElasticityModel with additional damping, one should use it using the
         * Adapt class as follows:
         *
         * \code
         * template <typename Traits, typename Block>
         * class MyCustomElasticityModel : public
         * Component<MyCustomElasticityModel<Traits, Block>> {
         *   // ...implementation
         * };
         *
         * template <typename Traits, typename Block>
         * class MyCustomElasticityModelWithDampingPerRod : public
         * Adapt<MyCustomElasticityModel<CRT, ComputationalBlock>>::
         *      template with<ExplicitDampingAdapterPerRod> {};
         * \endcode
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         * \tparam ComputationalBlock The final block which is derived from the
         * current component
         * \tparam ElasticityModelParam The user defined elasticity model which
         * is to be wrapped in the interface type
         *
         * \see ExplicitDampingAdapterVariablesPerRod
         */
        template <typename CRT, typename ComputationalBlock,
                  template <typename /*CRT*/, typename /* ComputationalBlock*/,
                            typename... /* ElasticityModelMetaArgs*/>
                  class ElasticityModelParam,
                  typename... ElasticityModelMetaArgs>
        class ExplicitDampingAdapterPerRod<
            CRT, ComputationalBlock,
            ElasticityModelParam<CRT, ComputationalBlock,
                                 ElasticityModelMetaArgs...>>
            : public ExplicitDampingAdapterFacade<
                  CRT, ComputationalBlock,
                  tt::elasticity_component_trait_t<ElasticityModelParam<
                      CRT, ComputationalBlock, ElasticityModelMetaArgs...>>,
                  ExplicitDampingAdapterVariablesPerRod> {
         private:
          //**Type definitions**************************************************
          //! Traits type
          using Traits = CRT;
          //! The type of the elasticity model
          using ThisModel = ElasticityModelParam<Traits, ComputationalBlock,
                                                 ElasticityModelMetaArgs...>;
          //! Base model (not need this complexity)
          using BaseModel = tt::elasticity_component_trait_t<ThisModel>;
          //! Parent type
          using Parent = ExplicitDampingAdapterFacade<
              CRT, ComputationalBlock, BaseModel,
              detail::ExplicitDampingAdapterVariablesPerRod>;
          //! This type
          using This = ExplicitDampingAdapterPerRod<Traits, ComputationalBlock,
                                                    ThisModel>;
          //********************************************************************

         protected:
          //**Type definitions**************************************************
          //! List of computed variables
          using typename Parent::ComputedVariables;
          //! List of initialized variables
          using typename Parent::InitializedVariables;
          //! List of all variables
          using typename Parent::Variables;
          //********************************************************************

          //**Parent methods****************************************************
          //! Initialize method inherited from parent class
          using Parent::initialize;
          //********************************************************************
        };
        //**********************************************************************

      }  // namespace detail

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica

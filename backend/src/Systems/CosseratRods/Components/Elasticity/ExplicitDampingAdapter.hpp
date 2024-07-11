#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cmath>
#include <utility>

///// Types always first
#include "Systems/CosseratRods/Components/Elasticity/Types.hpp"
/////
#include "Systems/CosseratRods/Components/Elasticity/detail/ExplicitDampingAdapter.hpp"
#include "Utilities/Math/Vec3.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Requires.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Adapter for adding explicit damping to an elasticity component
       * \ingroup cosserat_rod_component
       *
       * \details
       * ExplicitDampingAdapter adapts an Elasticity component for use
       * within Cosserat rods implemented in the Blocks framework. It adds
       * an additional damping force and torque to those arising from the
       * Elasticity model. It assumes that the Elasticity model operates on a
       * 1-D parametrized entity spanning a single spatial dimension (i.e.
       * along center-line coordinates, such as a Cosserat rod data-structure).
       * By design it only binds to classes with signature of a Component.
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
       * Note that however, this does not provide the appropriate interface
       * required by the rest of the components within a Cosserat rod hierarchy.
       * To do so, we need to add the ElasticityInterface adapter, like so
       *
       * \code
       * template <typename Traits, typename Block>
       * class MyCustomElasticityModelWithDampingPerRodThatWorks : public
       * Adapt<MyCustomElasticityModel<CRT, ComputationalBlock>>::
       *      template with<ElasticityInterface, ExplicitDampingAdapter>
       *      {};
       * \endcode
       * The ElasticityInterface needs to be the left-most argument to with,
       * else a compilation error is raised.
       *
       * \tparam CRT A valid Cosserat Rod Traits class
       * \tparam ComputationalBlock The final block which is derived from the
       * current component
       * \tparam ElasticityModelParam The user defined elasticity model which
       * is to be wrapped in the interface type
       *
       * \see ExplicitDampingAdapterVariables
       */
      using detail::ExplicitDampingAdapter;
      //************************************************************************

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Adapter for adding explicit damping to an elasticity component
       * \ingroup cosserat_rod_component
       *
       * \details
       * ExplicitDampingAdapterPerRod adapts an Elasticity component for use
       * within Cosserat rods implemented in the Blocks framework. It adds
       * an additional damping force and torque to those arising from the
       * Elasticity model. It assumes that the Elasticity model operates on a
       * 1-D parametrized entity spanning a single spatial dimension (i.e.
       * along center-line coordinates, such as a Cosserat rod data-structure).
       * By design it only binds to classes with signature of a Component.
       * Note that instead of having damping for every grid node/element
       * (which is achieved by ExplicitDampingAdapter) it reserves the same
       * damping for an entire rod.
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
       * Note that however, this does not provide the appropriate interface
       * required by the rest of the components within a Cosserat rod hierarchy.
       * To do so, we need to add the ElasticityInterface adapter, like so
       *
       * \code
       * template <typename Traits, typename Block>
       * class MyCustomElasticityModelWithDampingPerRodThatWorks : public
       * Adapt<MyCustomElasticityModel<CRT, ComputationalBlock>>::
       *      template with<ElasticityInterface, ExplicitDampingAdapterPerRod>
       *      {};
       * \endcode
       * The ElasticityInterface needs to be the left-most argument to with,
       * else a compilation error is raised.
       *
       *
       * \tparam CRT A valid Cosserat Rod Traits class
       * \tparam ComputationalBlock The final block which is derived from the
       * current component
       * \tparam ElasticityModelParam The user defined elasticity model which
       * is to be wrapped in the interface type
       *
       * \see ExplicitDampingAdapterVariablesPerRod
       */
      using detail::ExplicitDampingAdapterPerRod;
      //************************************************************************

      //// should not include any protocols
      // template <
      //     typename CRT,                // Cosserat Rod Traits
      //     typename ComputationalBlock, // Block
      //     template <typename /*CRT*/, typename /* ComputationalBlock*/,
      //     typename...> class ElasticityModelParam, typename...
      //     ElasticityModelMetaArgs>
      // class ExplicitDampingAdapter<
      //     CRT, ComputationalBlock,
      //     ElasticityModelParam<CRT, ComputationalBlock,
      //     ElasticityModelMetaArgs...>> : public detail::ElasticityInterface<
      //           CRT, ComputationalBlock,
      //           detail::ExplicitDampingAdapter<
      //               CRT, ComputationalBlock,
      //               ElasticityModelParam<CRT, ComputationalBlock,
      //                                    ElasticityModelMetaArgs...>>> {
      // private:
      //   using Traits = CRT;
      //   using ThisModel = ElasticityModelParam<Traits, ComputationalBlock,
      //                                          ElasticityModelMetaArgs...>;
      //   using This = ExplicitDampingAdapter<Traits, ComputationalBlock,
      //   ThisModel>; using Implementation =
      //       detail::ExplicitDampingAdapter<Traits, ComputationalBlock,
      //       ThisModel>;
      //   using Parent =
      //       detail::ElasticityInterface<Traits, ComputationalBlock,
      //       Implementation>;
      //
      // protected:
      //   static_assert(std::is_same<ThisModel, ElasticityModel>::value,
      //                 "Invariant failure!");
      //
      //   using Parent::initialize;
      //   using typename Parent::ComputedVariables;
      //   using typename Parent::InitializedVariables;
      //   using typename Parent::Variables;
      // };
      //
      //// should not include any protocols
      // template <
      //     typename CRT,                // Cosserat Rod Traits
      //     typename ComputationalBlock, // Block
      //     template <typename /*CRT*/, typename /* ComputationalBlock*/,
      //     typename...> class ElasticityModelParam, typename...
      //     ElasticityModelMetaArgs>
      // class ExplicitDampingAdapterPerRod<
      //     CRT, ComputationalBlock,
      //     ElasticityModelParam<CRT, ComputationalBlock,
      //     ElasticityModelMetaArgs...>> : public detail::ElasticityInterface<
      //           CRT, ComputationalBlock,
      //           detail::ExplicitDampingAdapterPerRod<
      //               CRT, ComputationalBlock,
      //               ElasticityModelParam<CRT, ComputationalBlock,
      //                                    ElasticityModelMetaArgs...>>> {
      // private:
      //   using Traits = CRT;
      //   using ThisModel = ElasticityModelParam<Traits, ComputationalBlock,
      //                                          ElasticityModelMetaArgs...>;
      //   using This =
      //       ExplicitDampingAdapterPerRod<Traits, ComputationalBlock,
      //       ThisModel>;
      //   using Implementation =
      //       detail::ExplicitDampingAdapterPerRod<Traits, ComputationalBlock,
      //                                            ThisModel>;
      //   using Parent =
      //       detail::ElasticityInterface<Traits, ComputationalBlock,
      //       Implementation>;
      //
      // protected:
      //   static_assert(std::is_same<ThisModel, ElasticityModel>::value,
      //                 "Invariant failure!");
      //
      //   using Parent::initialize;
      //   using typename Parent::ComputedVariables;
      //   using typename Parent::InitializedVariables;
      //   using typename Parent::Variables;
      // };

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica

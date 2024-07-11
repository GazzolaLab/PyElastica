#pragma once

//******************************************************************************
// Includes
//******************************************************************************
///// Types always first
#include "Systems/CosseratRods/Components/Elasticity/Types.hpp"
/////
#include "Systems/CosseratRods/Components/Elasticity/ExplicitDampingAdapter.hpp"
#include "Systems/CosseratRods/Components/Elasticity/Protocols.hpp"
#include "Systems/CosseratRods/Components/Elasticity/WithDiagonalLinearHyperElasticModel.hpp"
#include "Systems/CosseratRods/Components/Elasticity/detail/ElasticityInterface.hpp"
#include "Systems/CosseratRods/Components/helpers.hpp"
#include "Utilities/CRTP.hpp"
#include "Utilities/TMPL.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Elasticity component corresponding to a linear hyper-elastic
       * model with diagonal stress--stain relations
       * \ingroup cosserat_rod_component
       *
       * \details
       * WithExplicitlyDampedDiagonalLinearHyperElasticModel implements the
       * final Elasticity component (with the right interface) for use within
       * Cosserat rods implemented in the Blocks framework. It denotes a
       * hyper-elasic model where the stresses depend on the strains linearly,
       * in a pure diagonal manner (i.e. strains along a principal axis only
       * affect the stresses along the same principal axis, and nothing else).
       * Additionally the modeled loads are damped with a damping factor
       * proportional to the rate of change of velocities. These are based on
       * the following relations:
       *
       * \f[
       * \boldsymbol{\tau}_{\mathcal{L}} =
       * \bv{B}\left(\boldsymbol{\kappa}_{\mathcal{L}}-\boldsymbol{\kappa}^o_{\mathcal{L}}\right)
       * - \nu_{\omega} \boldsymbol{\omega}_{\mathcal{L}}
       * \f]
       * \f[
       * \boldsymbol{n}_{\mathcal{L}} =
       * \bv{S}\left(\boldsymbol{\sigma}_{\mathcal{L}}-\boldsymbol{\sigma}^o_{\mathcal{L}}\right)
       * - \nu_{v} \boldsymbol{v}
       * \f]
       *
       * \note
       * It requires a valid Geometry component declared in the Blocks
       * Hierarchy to ensure it is properly used. Else, a compilation error is
       * thrown.
       *
       * \usage
       * Since WithExplicitlyDampedDiagonalLinearHyperElasticModel is a valid
       * and complete Elasticity component adhering to protocols::Elastic1D, one
       * can use it to declare a CosseratRodPlugin within the @ref blocks
       * framework
       *
       * \code
       * // pre-declare RodTraits, Blocks
       * using CircularCosseratRod = CosseratRodPlugin<RodTraits, Block,
       * // Conforms to protocols::Geometry1D!
       * components::WithCircularCosseratRod,
       * components::WithExplicitlyDampedDiagonalLinearHyperElasticModel>;
       * \endcode
       *
       * \tparam CRT A valid Cosserat Rod Traits class
       * \tparam ComputationalBlock The final block which is derived from the
       * current component
       *
       * \see CosseratRodPlugin
       */
      template <typename CRT, typename ComputationalBlock>
      class WithExplicitlyDampedDiagonalLinearHyperElasticModel
          : public Adapt<detail::WithDiagonalLinearHyperElasticModelImpl<
                CRT, ComputationalBlock>>::
                template with<detail::ElasticityInterface,
                              detail::ExplicitDampingAdapter> {};
      //************************************************************************

      // clang-format off
//******************************************************************************
/*!\brief Documentation stub with tags of  WithExplicitlyDampedDiagonalLinearHyperElasticModel
 * \ingroup cosserat_rod_component
 *
| Elasticity Variables           ||
|--------------------------------|-------------------------------------------------------------------------------|
| On Nodes    (`n_elements+1`)   | elastica::tags::ForceDampingRate, elastica::tags::InternalLoads               |
| On Elements (`n_elements`)     | elastica::tags::InternalTorques, elastica::tags::InternalStress               |
|^                               | elastica::tags::ShearStretchRigidityMatrix, elastica::tags::TorqueDampingRate |
| On Voronoi  (`n_elements - 1`) | elastica::tags::InternalCouple, elastica::tags::BendingTwistRigidityMatrix    |
*/
      template <typename CRT, typename ComputationalBlock>
      using WithExplicitlyDampedDiagonalLinearHyperElasticModelTagsDocsStub =
      WithExplicitlyDampedDiagonalLinearHyperElasticModel<CRT, ComputationalBlock>;
//******************************************************************************
      // clang-format on

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica

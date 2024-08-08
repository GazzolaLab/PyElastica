#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cmath>
#include <utility>

#include "Systems/CosseratRods/Components/Geometry/Protocols.hpp"
#include "Systems/CosseratRods/Components/Geometry/Types.hpp"
///
#include "Systems/CosseratRods/Components/Geometry/detail/CircleCrossSectionOperations.hpp"
#include "Systems/CosseratRods/Components/Geometry/detail/CosseratRodCrossSectionInterface.hpp"
#include "Systems/CosseratRods/Components/helpers.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Geometry component corresponding to a spanwise rod with a
       * circular cross section
       * \ingroup cosserat_rod_component
       *
       * \details
       * WithCircularCosseratRod implements the final Geometry component use
       * within Cosserat rods implemented in the Blocks framework. It denotes a
       * Cosserat rod data-structure with a circular cross section in the
       * lateral direction and spanning a single spatial dimension along the
       * transverse dimension.
       *
       * \usage
       * Since WithCircularCosseratRod is a valid and complete Geometry
       * component adhering to protocols::Geometry1D, one can use it to declare
       * a CosseratRodPlugin within the @ref blocks framework \code
       * // pre-declare RodTraits, Blocks
       * using CircularCosseratRod = CosseratRodPlugin<RodTraits, Block,
       * components::WithCircularCosseratRod>
       * \endcode
       *
       * \tparam CRT A valid Cosserat Rod Traits class
       * \tparam ComputationalBlock The final block which is derived from the
       * current component
       *
       * \see CosseratRodPlugin
       */
      template <typename CRT, typename ComputationalBlock>
      class WithCircularCosseratRod
          : public detail::CosseratRodCrossSectionInterface<
                CRT, ComputationalBlock,
                detail::CircleCrossSectionOperations<CRT>>,
            public ::tt::ConformsTo<protocols::Geometry1D>,
            public GeometryComponent<
                WithCircularCosseratRod<CRT, ComputationalBlock>> {
       private:
        //**Type definitions****************************************************
        //! Traits type
        using Traits = CRT;
        //! Cross section type
        using CrossSection = detail::CircleCrossSectionOperations<Traits>;
        //! This type
        using This = WithCircularCosseratRod<Traits, ComputationalBlock>;
        //! Parent type
        using Parent =
            detail::CosseratRodCrossSectionInterface<Traits, ComputationalBlock,
                                                     CrossSection>;
        //**********************************************************************

        //! sanity check cross section classes
        static_assert(
            cpp17::is_same_v<typename Parent::CrossSection, CrossSection>,
            "Cross section assertion failure, contact the developers!");

       protected:
        //**Parent methods and aliases******************************************
        //! Initialize method inherited from parent class
        using Parent::initialize;
        //! List of computed variables
        using typename Parent::ComputedVariables;
        //! List of initialized variables
        using typename Parent::InitializedVariables;
        //! List of all variables
        using typename Parent::Variables;
        //**********************************************************************
      };
      //************************************************************************

      // clang-format off
//******************************************************************************
/*!\brief Documentation stub with tags of  WithCircularCosseratRod
 * \ingroup cosserat_rod_component
 *
| Geometry Variables             ||
|--------------------------------|----------------------------------------------------------------------------------------------------------------|
| On Nodes    (`n_elements+1`)   | elastica::tags::Position                                                                                       |
| On Elements (`n_elements`)     | elastica::tags::Director,  elastica::tags::ElementDilatation, elastica::tags::ElementDimension,                |
|^                               | elastica::tags::ElementLength, elastica::tags::ElementVolume, elastica::tags::ReferenceElementLength,          |
|^                               | elastica::tags::ReferenceShearStretchStrain, elastica::tags::ShearStretchStrain, elastica::tags::Tangent,      |
| On Voronoi  (`n_elements - 1`) | elastica::tags::Curvature, elastica::tags::ReferenceCurvature, elastica::tags::ReferenceVoronoiLength,         |
|^                               | elastica::tags::VoronoiDilatation, elastica::tags::VoronoiLength                                               |
*/
      template <typename CRT, typename ComputationalBlock>
      using WithCircularCosseratRodTagsDocsStub =
      WithCircularCosseratRod<CRT, ComputationalBlock>;
//******************************************************************************
      // clang-format on

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica

// public RodKinematics<RodTraits>
// Sparse_strategy
// Dense strategy

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/CosseratRods/Traits/PlacementTraits/PlacementTraits.hpp"
#include "Systems/CosseratRods/Traits/PlacementTraits/TypeTraits.hpp"
#include "Systems/CosseratRods/Traits/Types.hpp"

namespace elastica {

  namespace cosserat_rod {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Traits controlling placement of data-structures for Cosserat rods
     * in \elastica
     * \ingroup cosserat_rod_traits
     *
     * \details
     * PlacementTrait is the customization point for altering the placement of
     * data-structures (defined by DataOpsTraits) to be used within the Cosserat
     * rod hierarchy implemented using @ref blocks in \elastica. It defines
     * (domain-specific) types corresponding to placement on a Cosserat rod
     * (such as on the nodes, elements etc.), and is intended for use as a
     * template parameter in CosseratRodTraits.
     *
     * \see elastica::cosserat_rod::CosseratRodTraits
     */
    struct PlacementTrait : private NonCreatable {
      //**Type definitions******************************************************
      //! Tag to place on nodes
      using OnNode = placement_tags::OnNode;
      //! Tag to place on elements
      using OnElement = placement_tags::OnElement;
      //! Tag to place on voronois
      using OnVoronoi = placement_tags::OnVoronoi;
      //! Tag to place on the whole rod rather than at discrete points
      using OnRod = placement_tags::OnRod;
      //! Type of size
      using size_type = typename protocols::commons::size_type;
      //************************************************************************
    };
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

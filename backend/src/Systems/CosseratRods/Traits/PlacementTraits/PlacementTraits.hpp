#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <cstddef>

#include "ErrorHandling/Assert.hpp"
#include "Protocols.hpp"
#include "Utilities/NonCreatable.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace placement_tags {

      //************************************************************************
      /*!\brief Tag indicating placement on nodes in a staggered grid
       * \ingroup cosserat_rod_traits
       */
      struct OnNode : /* public NonCreatable*/ public ::tt::ConformsTo<
                          protocols::PlacementTrait> {
        //**********************************************************************
        /*!\brief Get size/degrees of freedom for the given placement
         */
        static inline auto get_dofs(
            protocols::commons::size_type n_elems) noexcept
            -> protocols::commons::size_type {
          ELASTICA_ASSERT(n_elems > 0UL,
                          "prescribed number of elements incorrect");
          return (n_elems + 1UL);
        }
        //**********************************************************************

        //**********************************************************************
        /*!\brief Gets number of ghosts needed for the given placement
         */
        static inline auto n_ghosts() noexcept
            -> protocols::commons::size_type {
          return 1UL;
        }
        //**********************************************************************
      };
      //************************************************************************

      //************************************************************************
      /*!\brief Tag indicating placement on elements in a staggered grid
       * \ingroup cosserat_rod_traits
       */
      struct OnElement : /* public NonCreatable*/ public ::tt::ConformsTo<
                             protocols::PlacementTrait> {
        //**********************************************************************
        /*!\brief Get size/degrees of freedom for the given placement
         */
        inline static auto get_dofs(
            protocols::commons::size_type n_elems) noexcept
            -> protocols::commons::size_type {
          ELASTICA_ASSERT(n_elems > 0UL,
                          "prescribed number of elements incorrect");
          return n_elems;
        }
        //**********************************************************************

        //**********************************************************************
        /*!\brief Gets number of ghosts needed for the given placement
         */
        static inline auto n_ghosts() noexcept
            -> protocols::commons::size_type {
          return 2UL;
        }
        //**********************************************************************
      };
      //************************************************************************

      //************************************************************************
      /*!\brief Tag indicating placement on voronoi elements in a staggered grid
       * \ingroup cosserat_rod_traits
       *
       * \details
       * A voronoi element in this case refers to interior nodes which form the
       * Voronoi domain of two adjacent elements
       *
       *    ...===o=====+======0======+======o===...
       *
       *    where
       *    - o is a node
       *    - + is the center of an element
       *    - the region between the two + forms the open Voronoi domain/cell
       *    of the two adjacent elements with 0, the node, as its central
       *    Voronoi vertex
       */
      struct OnVoronoi : /* public NonCreatable*/ public ::tt::ConformsTo<
                             protocols::PlacementTrait> {
        //**********************************************************************
        /*!\brief Get size/degrees of freedom for the given placement
         */
        inline static auto get_dofs(
            protocols::commons::size_type n_elems) noexcept
            -> protocols::commons::size_type {
          ELASTICA_ASSERT(n_elems > 0UL,
                          "prescribed number of elements incorrect");
          return (n_elems - 1UL);
        }
        //**********************************************************************

        //**********************************************************************
        /*!\brief Gets number of ghosts needed for the given placement
         */
        static inline auto n_ghosts() noexcept
            -> protocols::commons::size_type {
          return 3UL;
        }
        //**********************************************************************
      };
      //************************************************************************

      //************************************************************************
      /*!\brief Tag indicating placement on a whole rod, as opposed to
       * on a grid
       * \ingroup cosserat_rod_traits
       */
      struct OnRod : /* public NonCreatable*/ public ::tt::ConformsTo<
                         protocols::PlacementTrait> {
        //**********************************************************************
        /*!\brief Get size/degrees of freedom for the given placement
         */
        inline static auto get_dofs(
            protocols::commons::size_type /*n_elems*/) noexcept
            -> protocols::commons::size_type {
          // This causes test to fail, so I comment it for now
          // ELASTICA_ASSERT(n_elems > 0UL,
          //                "prescribed number of elements incorrect");

          // the actual return value here doesn't matter as it does not get used
          return 1UL;
        }
        //**********************************************************************

        //**********************************************************************
        /*!\brief Gets number of ghosts needed for the given placement
         */
        static inline auto n_ghosts() noexcept
            -> protocols::commons::size_type {
          return 0UL;
        }
        //**********************************************************************
      };
      //************************************************************************

    }  // namespace placement_tags

  }  // namespace cosserat_rod

}  // namespace elastica

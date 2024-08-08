#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <cstddef>  // for size_t
#include <utility>  // for declval

#include "Utilities/TypeTraits.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace protocols {

      //========================================================================
      //
      //  TAG DEFINITIONS
      //
      //========================================================================

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */

      //************************************************************************
      /*!\brief Common types for placement tags
       * \ingroup cosserat_rod
       */
      struct commons {
        using size_type = std::size_t;
      };
      //************************************************************************

      /*! \endcond */
      //************************************************************************

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Placement-trait protocol.
       * \ingroup cosserat_rod_traits
       *
       * Class to enforce adherence to a Cosserat Rod Placement-trait protocol.
       * Any valid data trait class within the \elastica library should
       * (publicly) inherit from this class to indicate it qualifies as a
       * data-trait. Only in case a class is derived publicly from this base
       * class, the tt::conforms_to and tt::assert_conforms_to type traits
       * recognizes the class as valid data traits.
       *
       * Requires that a conforming type `ConformingType` has these nested
       * static member functions,
       * \snippet this expected_static_functions
       *
       * \example
       * The following shows an example of minimal conformance to this protocol
       * \snippet PlacementTraits/Test_Protocols.cpp placement_protocol_eg
       */
      struct PlacementTrait {
        //**********************************************************************
        /*! \cond ELASTICA_INTERNAL */
        /*!\brief Auxiliary helper struct for enforcing protocols.
        // \ingroup protocols
        */
        template <typename ConformingType>
        struct test {
         public:
          /// [expected_static_functions]
          using size_type = typename commons::size_type;  // std::size_t
          using dofs_return_type =
              decltype(ConformingType::get_dofs(std::declval<size_type>()));
          static_assert(cpp17::is_same_v<dofs_return_type, size_type>,
                        R"error(
Not a conforming placement trait, doesn't properly implement static function
`commons::size_type get_dofs(commons::size_type)`
)error");

          using n_ghosts_return_type = decltype(ConformingType::n_ghosts());
          static_assert(cpp17::is_same_v<n_ghosts_return_type, size_type>,
                        R"error(
Not a conforming placement trait, doesn't properly implement static function
`commons::size_type n_ghosts()`
)error");

          /// [expected_static_functions]
        };
        /*! \endcond */
        //**********************************************************************
      };
      //************************************************************************

    }  // namespace protocols

  }  // namespace cosserat_rod

}  // namespace elastica

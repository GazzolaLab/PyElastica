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
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Data-trait protocol.
       * \ingroup cosserat_rod_traits
       *
       * Class to enforce adherence to a Cosserat Rod Data-trait protocol. Any
       * valid data trait class within the \elastica library should (publicly)
       * inherit from this class to indicate it qualifies as a data-trait. Only
       * in case a class is derived publicly from this base class, the
       * tt::conforms_to and tt::assert_conforms_to type traits recognizes the
       * class as valid data traits.
       *
       * Requires that a conforming type `ConformingType` has these nested types
       * \snippet this expected_types
       * and these static member functions,
       * \snippet this expected_static_functions
       *
       * \example
       * The following shows an example of minimal conformance to this protocol.
       * With the setup shown here
       * \snippet Mocks/CosseratRodTraits.hpp vector_datatrait
       * we ensure conformance as
       * \snippet DataType/Test_Protocols.cpp datatrait_protocol_eg
       */
      struct DataTrait {
        //**********************************************************************
        /*! \cond ELASTICA_INTERNAL */
        /*!\brief Auxiliary helper struct for enforcing protocols.
        // \ingroup protocols
        */
        template <typename ConformingType>
        struct test {
         public:
          /// [expected_types]
          //**Type definitions**************************************************
          //! The main type for a TaggedTuple
          using data_type = typename ConformingType::type;
          //! The type of slice for the data type
          using slice_type = typename ConformingType::slice_type;
          //! The type of const slice for the data type
          using const_slice_type = typename ConformingType::const_slice_type;
          //! The type of a reference to the underlying data type
          using reference_type = typename ConformingType::reference_type;
          //! The type of const reference to the underlying data type
          using const_reference_type =
              typename ConformingType::const_reference_type;
          //! The type of reference to the slice type
          using reference_slice_type =
              typename ConformingType::reference_slice_type;
          //! The type of const reference to the slice type
          using const_reference_slice_type =
              typename ConformingType::const_reference_slice_type;
          //! The type of a ghost element for the data type
          using ghost_type = typename ConformingType::ghost_type;
          // //! The rank of the the data type
          // using rank = typename ConformingType::rank;
          //********************************************************************
          /// [expected_types]

          /// [expected_static_functions]
          // Function prototypes needing no arguments
          using ghost_value_return_type =
              decltype(ConformingType::ghost_value());
          static_assert(cpp17::is_same_v<ghost_type, ghost_value_return_type>,
                        R"error(
Not a conforming data trait, doesn't properly implement static function
`ghost_type ghost_value()`)error");

          // Function prototypes needing two arguments
          using index = std::size_t;
          using slice_at_index_return_type = decltype(ConformingType::slice(
              std::declval<data_type&>(), std::declval<index>()));
          static_assert(
              cpp17::is_same_v<slice_at_index_return_type, reference_type>,
              R"error(
Not a conforming data trait, doesn't properly implement static function
`auto slice(data_type& data, std::size_t index)`
)error");

          using const_slice_at_index_return_type =
              decltype(ConformingType::slice(std::declval<data_type const&>(),
                                             std::declval<index>()));
          static_assert(cpp17::is_same_v<const_slice_at_index_return_type,
                                         const_reference_type>,
                        R"error(
Not a conforming data trait, doesn't properly implement static function
`auto slice(data_type const& data, std::size_t index)`
)error");

          using slice_of_slice_at_index_return_type =
              decltype(ConformingType::slice(std::declval<slice_type&>(),
                                             std::declval<index>()));
          static_assert(cpp17::is_same_v<slice_of_slice_at_index_return_type,
                                         reference_slice_type>,
                        R"error(
Not a conforming data trait, doesn't properly implement static function
`auto slice(slice_type& slice, std::size_t index)`
)error");

          using slice_of_const_slice_at_index_return_type =
              decltype(ConformingType::slice(std::declval<slice_type const&>(),
                                             std::declval<index>()));
          static_assert(
              cpp17::is_same_v<slice_of_const_slice_at_index_return_type,
                               const_reference_slice_type>,
              R"error(
Not a conforming data trait, doesn't properly implement static function
`auto slice(slice_type const& slice, std::size_t index)`
)error");

          using const_slice_of_const_slice_at_index_return_type =
              decltype(ConformingType::slice(
                  std::declval<const_slice_type const&>(),
                  std::declval<index>()));
          static_assert(
              cpp17::is_same_v<const_slice_of_const_slice_at_index_return_type,
                               const_reference_slice_type>,
              R"error(
Not a conforming data trait, doesn't properly implement static function
`auto slice(const_slice_type const& slice, std::size_t index)`
)error");

          using new_dofs = std::size_t;
          using resize_return_type = decltype(ConformingType::resize(
              std::declval<data_type&>(), std::declval<new_dofs>()));
          static_assert(cpp17::is_same_v<void, resize_return_type>,
                        R"error(
Not a conforming data trait, doesn't properly implement static function
`void resize(data_type& data, std::size_t new_dofs)`)error");

          // Function prototypes needing three arguments
          using start = std::size_t;
          using size = std::size_t;
          using slice_return_type = decltype(ConformingType::slice(
              std::declval<data_type&>(), std::declval<start>(),
              std::declval<size>()));
          static_assert(cpp17::is_same_v<slice_type, slice_return_type>,
                        R"error(
Not a conforming data trait, doesn't properly implement static function
`slice_type slice(data_type& data, std::size_t start, std::size_t size)`)error");

          using const_slice_return_type = decltype(ConformingType::slice(
              std::declval<data_type const&>(), std::declval<start>(),
              std::declval<size>()));
          static_assert(
              cpp17::is_same_v<const_slice_type, const_slice_return_type>,
              R"error(
Not a conforming data trait, doesn't properly implement static function
`const_slice_type slice(data_type const& data, std::size_t start, std::size_t size)`)error");

          /// [expected_static_functions]
        };
        /*! \endcond */
        //**********************************************************************
      };
      //************************************************************************

    }  // namespace protocols

  }  // namespace cosserat_rod

}  // namespace elastica

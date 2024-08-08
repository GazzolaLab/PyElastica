#pragma once

// Vector types
#include <functional>  // ref wrapper
#include <vector>

// assert
#include "ErrorHandling/Assert.hpp"
#include "Systems/CosseratRods/Traits/DataTraits/Protocols.hpp"
#include "Systems/CosseratRods/Traits/DataTraits/Rank.hpp"
#include "Utilities/NoSuchType.hpp"

namespace elastica {

  namespace cosserat_rod {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Tag for a Rank 1 tensor within Cosserat rods in \elastica
     * \ingroup cosserat_rod_traits
     *
     * \details
     * ScalarTag is used within the Cosserat rod components of \elastica to
     * indicate a vector.
     *
     * \tparam RealOrIndex a floating point (or) index type
     */
    template <typename RealOrIndex>
    struct ScalarTag : public ::tt::ConformsTo<protocols::DataTrait> {
      //**Type definitions******************************************************
      //! The main type for a TaggedTuple
      using type = std::vector<RealOrIndex>;
      //! Data type for better readability
      using data_type = type;
      //! The type of slice for the data type
      using slice_type = std::vector<std::reference_wrapper<RealOrIndex>>;
      //! The type of a const slice for the data type
      using const_slice_type =
          const std::vector<std::reference_wrapper<const RealOrIndex>>;
      //! The type of a ghost element for the data type
      using ghost_type = NoSuchType;
      //! The type of a reference to the underlying data type
      using reference_type = RealOrIndex&;
      //! The type of const reference to the underlying data type
      using const_reference_type = RealOrIndex const&;
      //! The type of reference to the slice type
      using reference_slice_type = RealOrIndex&;
      //! The type of const reference to the slice type
      using const_reference_slice_type = RealOrIndex const&;
      //! The rank of the data type
      using rank = Rank<0U>;
      //************************************************************************

      //**Utility functions*****************************************************
      /*!\name Utility functions */
      //@{

      //************************************************************************
      /*!\brief Obtain the ghost value
       *
       * \details
       * Obtains a default value for putting in ghost elements. This function
       * can be overriden in case the user wants to. This should not be a
       * RealOrIndex (because implicit conversion fills entire data-structure
       * with that particular value) but rather a ghost_type object (like a
       * matrix for example)
       */
      static inline constexpr auto ghost_value() noexcept -> ghost_type {
        return {};
      }
      //**********************************************************************

      //************************************************************************
      /*!\brief Obtain slice of the data type at a given index
       *
       * \details
       * Overload to obtain a slice of data at `index`.
       * It is typically used in the block to generate BlockSlices or access
       * indexed data
       *
       * \param data  Slice to be further sliced
       * \param index Index of slicing
       */
      static inline auto slice(data_type& data, std::size_t index)
          -> reference_type {
        return data[index];
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Obtain slice of the const data type at a given index
       *
       * \details
       * Overload to obtain a slice of const data at `index`.
       * It is typically used in the block to generate ConstBlockSlices or
       * access indexed data
       *
       * \param data  Slice to be further sliced
       * \param index Index of slicing
       */
      static inline auto slice(data_type const& data, std::size_t index)
          -> const_reference_type {
        return data[index];
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Obtain a sub-slice of the slice at a given index
       *
       * \details
       * Overload to obtain a subslice of slice at `index`.
       * It is typically used in the block to generate slices that are then
       * filled in by ghost values.
       *
       * \param slice Slice to be further sliced
       * \param index Index of slicing
       */
      static inline auto slice(slice_type& slice, std::size_t index)
          -> reference_type {
        return slice[index];
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Obtain a sub-slice of the const slice at a given index
       *
       * \details
       * Overload to obtain a slice of data at `index`.
       * It is sometimes needed in a Plugin where the context is const, but the
       * data of the underlying structure may well be modified.
       *
       * \param slice Slice to be further sliced
       * \param index Index of slicing
       */
      static inline auto slice(slice_type const& slice, std::size_t index)
          -> const_reference_type {
        return slice[index];
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Obtain sub-slice of the const_slice type at a given index
       *
       * \details
       * Overload to obtain a subslice of const_slice at `index`.
       * It is needed in a Plugin where the slice itself is const, such as a
       * ConstBlockSlice
       *
       * \param slice Slice to be further sliced
       * \param index Index of slicing
       */
      static inline decltype(auto) slice(const_slice_type const& slice,
                                         std::size_t index) {
        return slice[index];
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Resize the data type to contain at least `size` entries
       *
       * \param data The data to be resized
       * \param new_size New size of data
       */
      static inline void resize(data_type& data, std::size_t new_size) {
        ELASTICA_ASSERT(data.size() <= new_size,
                        "Contract violation, block shrinks");
        return data.resize(new_size);
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Obtain slice of the data type through a set of indices
       *
       * \details
       * Overload to obtain a slice of `size` amount from data at `start`.
       * This is typically used in the block to generate slices that are to be
       * filled in by the class hierarchies, when filling in a block
       * incrementally.
       *
       * \param data        Data to be sliced
       * \param start_index Start index
       * \param size        Size of the data
       */
      static inline auto slice(data_type& data, std::size_t start_index,
                               std::size_t size) -> slice_type {
        return slice_type(&data[start_index], &data[start_index + size]);
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Obtain slice of the const data type through a set of indices
       *
       * \details
       * Overload to obtain a slice of `size` amount from const data at `start`.
       * This is typically used in the block to generate slices that are to be
       * filled in by the class hierarchies, when filling in a block
       * incrementally.
       *
       * \param data        Data to be sliced
       * \param start_index Start index
       * \param size        Size of the data
       */
      static inline auto slice(data_type const& data, std::size_t start_index,
                               std::size_t size) -> const_slice_type {
        // unaligned always
        return const_slice_type(&data[start_index], &data[start_index + size]);
      }
      //************************************************************************

      //@}
      //************************************************************************
    };
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

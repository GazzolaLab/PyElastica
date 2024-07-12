#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Configuration/Systems.hpp"  // For index checking
// assert
#include "ErrorHandling/Assert.hpp"
#include "Systems/CosseratRods/Traits/DataType/Protocols.hpp"
#include "Systems/CosseratRods/Traits/DataType/Rank.hpp"

// matrix types
#include <blaze/math/DynamicMatrix.h>
// #include <blaze/math/StaticMatrix.h>
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/views/Column.h>
#include <blaze/math/views/Submatrix.h>

// this comes from the simulator module, which seems like an anti-pattern
// should be in Systems instead
#include "Simulator/Frames.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace elastica {

  namespace cosserat_rod {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Tag for a Rank 2 tensor within Cosserat rods in \elastica
     * \ingroup cosserat_rod_traits
     *
     * \details
     * MatrixTag is used within the Cosserat rod components of \elastica to
     * indicate a vector.
     *
     * \tparam Real a floating point type
     */
    template <typename Real>
    struct MatrixTag : public ::tt::ConformsTo<protocols::DataTrait> {
      //**Type definitions******************************************************
      //! The main type for a TaggedTuple
      using type =
          blaze::DynamicMatrix<Real, blaze::rowMajor,
                               blaze::AlignedAllocator<Real> /*, TagType*/>;
      //! Data type for better readability
      using data_type = type;
      //! The type of slice for the data type
      using slice_type = blaze::Submatrix_<data_type, blaze::unaligned>;
      //! The type of const slice for the data type
      using const_slice_type =
          const blaze::Submatrix_<const data_type, blaze::unaligned>;
      //! The type of a ghost element for the data type
      using ghost_type = blaze::StaticVector<Real, 3UL, blaze::columnVector>;
      //! The type of a reference to the underlying data type
      using reference_type = blaze::Column_<data_type>;
      //! The type of const reference to the underlying data type
      using const_reference_type = const blaze::Column_<const data_type>;
      //! The type of reference to the slice type
      using reference_slice_type =
          blaze::Subvector_<blaze::Column_<data_type>, blaze::unaligned>;
      //! The type of const reference to the slice type
      using const_reference_slice_type =
          const blaze::Subvector_<const blaze::Column_<const data_type>,
                                  blaze::unaligned>;
      //! The rank of the data type
      using rank = Rank<2U>;
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
       * Real (because implicit conversion fills entire data-structure with
       * that particular value) but rather a ghost_type object (like a matrix
       * for example)
       */
      static inline constexpr auto ghost_value() noexcept -> ghost_type {
        constexpr Real zero(0.0);
        return ghost_type{zero, zero, zero};
      }
      //************************************************************************

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
      static inline decltype(auto) slice(data_type& data, std::size_t index) {
        return blaze::column(data, index,
                             blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
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
      static inline decltype(auto) slice(data_type const& data,
                                         std::size_t index) {
        return blaze::column(data, index,
                             blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Obtain subslice of the slice type at a given index
       *
       * \details
       * Overload to obtain a subslice of slice at `index`.
       * It is typically used in the block to generate slices that are then
       * filled in by ghost values.
       *
       * \param slice Slice to be further sliced
       * \param index Index of slicing
       */
      static inline decltype(auto) slice(slice_type& slice, std::size_t index) {
        return blaze::column(slice, index,
                             blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Obtain sub-slice of the const slice type at a given index
       *
       * \details
       * Overload to obtain a subslice of const slice at `index`.
       *
       * It is sometimes needed in a Plugin where the context is const, but the
       * data of the underlying structure may well be modified.
       *
       * \param slice Slice to be further sliced
       * \param index Index of slicing
       */
      static inline decltype(auto) slice(slice_type const& slice,
                                         std::size_t index) {
        return blaze::column(slice, index,
                             blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Obtain subslice of a temporary slice type at a given index
       *
       * \details
       * Overload to obtain a subslice of slice at `index`.
       *
       * It is typically used when lazily generating individual slices from a
       * lazily generated slice : for example blockslice.get_position(i), where
       * get_position() lazily generates a slice, which then gets further
       * sliced.
       *
       * \param slice Slice to be further sliced
       * \param index Index of slicing
       */
      static inline decltype(auto) slice(slice_type&& slice,
                                         std::size_t index) {
        return blaze::column(static_cast<slice_type&&>(slice), index,
                             blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
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
        return blaze::column(slice, index,
                             blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Resize the data type to contain at least `size` entries
       *
       * \param data The data to be resized
       * \param new_size New size of data
       */
      static inline void resize(data_type& data, std::size_t new_size) {
        ELASTICA_ASSERT(data.columns() <= new_size,
                        "Contract violation, block shrinks");
        return data.resize(Frames::Dimension, new_size, true);
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
      static inline decltype(auto) slice(data_type& data,
                                         std::size_t start_index,
                                         std::size_t size) {
        return blaze::submatrix(data, 0UL, start_index, Frames::Dimension, size,
                                blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
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
      static inline decltype(auto) slice(data_type const& data,
                                         std::size_t start_index,
                                         std::size_t size) {
        return blaze::submatrix(data, 0UL, start_index, Frames::Dimension, size,
                                blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
      }
      //************************************************************************

      //@}
      //************************************************************************
    };
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

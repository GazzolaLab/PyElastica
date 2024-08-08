#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Configuration/Systems.hpp"  // For index checking

// assert
#include "ErrorHandling/Assert.hpp"

// Rank
#include "Systems/CosseratRods/Traits/DataType/Protocols.hpp"
#include "Systems/CosseratRods/Traits/DataType/Rank.hpp"

// Tensor types
#include <blaze/math/AlignmentFlag.h>
#include <blaze/math/views/Submatrix.h>
#include <blaze_tensor/math/DynamicTensor.h>
#include <blaze_tensor/math/StaticTensor.h>
#include <blaze_tensor/math/Subtensor.h>
#include <blaze_tensor/math/views/ColumnSlice.h>

// this comes from the simulator module, which seems like an anti-pattern
// should be in Systems instead
#include "Simulator/Frames.hpp"
#include "Utilities/Math/Vec3.hpp"
#include "Utilities/ProtocolHelpers.hpp"

namespace elastica {

  namespace cosserat_rod {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Tag for a Rank 3 tensor within Cosserat rods in \elastica
     * \ingroup cosserat_rod_traits
     *
     * \details
     * TensorTag is used within the Cosserat rod components of \elastica to
     * indicate a tensor.
     *
     * \tparam Real a floating point type
     */
    template <typename Real>
    struct TensorTag : public ::tt::ConformsTo<protocols::DataTrait> {
      //**Type definitions******************************************************
      //! The main type for a TaggedTuple
      using type = blaze::DynamicTensor<Real>;
      //! Data type for better readability
      using data_type = type;
      //! The type of slice for the data type
      using slice_type = blaze::Subtensor_<data_type, blaze::unaligned>;
      //! The type of const slice for the data type
      using const_slice_type =
          const blaze::Subtensor_<const data_type, blaze::unaligned>;
      /*
       * Developer note:
       *   Blaze tensor cant initialize with a matrix type
       *   so the ghosts for now are tensor types which involve
       *   more work than necessary
       *
       * using ghost_type = blaze::DynamicMatrix<real_t, blaze::rowMajor>;
       *        blaze::StaticMatrix<real_t, 3UL, 3UL, blaze::rowMajor>;
       */
      //! The type of a ghost element for the data type
      using ghost_type =
          blaze::StaticMatrix<Real, Frames::Dimension, Frames::Dimension>;
      //! The type of a reference to the underlying data type
      using reference_type = blaze::ColumnSlice_<data_type>;
      //! The type of const reference to the underlying data type
      using const_reference_type = const blaze::ColumnSlice_<const data_type>;
      //! The type of reference to the slice type
      using reference_slice_type =
          blaze::Submatrix_<blaze::ColumnSlice_<data_type>, blaze::unaligned>;
      //! The type of const reference to the slice type
      using const_reference_slice_type =
          const blaze::Submatrix_<const blaze::ColumnSlice_<const data_type>,
                                  blaze::unaligned>;
      //! The rank of the data type
      using rank = Rank<3U>;
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
      static inline auto ghost_value() noexcept -> ghost_type {
        // cannot be constexpr as it involves allocation :/
        constexpr Real zero(0.0);
        constexpr Real identity(1.0);
        return ghost_type{{{identity}, {zero}, {zero}},
                          {{zero}, {identity}, {zero}},
                          {{zero}, {zero}, {identity}}};
      }
      //************************************************************************

      // should really be in the Slice Tag class, but it ends up being
      // confusing as what we are really doing is taking a slice of a
      // slice but its abstracted away
      //        template <typename MT>
      //        static inline constexpr decltype(auto) slice(MT& matrix,
      //                                                     std::size_t i)

      // TODO decltype expression is same as SliceType
      // TODO This is internal function, need not be exposed
      //
      //

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
        return blaze::columnslice(data, index,
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
        return blaze::columnslice(data, index,
                                  blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
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
      static inline decltype(auto) slice(slice_type& slice, std::size_t index) {
        return blaze::columnslice(slice, index,
                                  blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
        // return blaze::subtensor(slice, 0UL, 0UL, index, Frames::Dimension,
        // Frames::Dimension, 1UL);
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Obtain a sub-const slice of the const slice at a given index
       *
       * \details
       * Overload to obtain a subslice of const slice at `index`.
       *
       * It is sometimes needed in a Plugin where the context is const, but the
       * data of the underlying structure may well be modified.
       *
       * \param slice Slice to be further const sliced
       * \param index Index of slicing
       */
      static inline decltype(auto) slice(slice_type const& slice,
                                         std::size_t index) {
        return blaze::columnslice(slice, index,
                                  blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Obtain a sub-slice of the slice at a given index
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
        return blaze::columnslice(static_cast<slice_type&&>(slice), index,
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
        return blaze::columnslice(slice, index,
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
        // (o, m, n)
        return data.resize(Frames::Dimension, Frames::Dimension, new_size,
                           true);
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Obtain slice of the data type through a set of indices
       *
       * \details
       * Overload to obtain a slice of `size` amount from data at `start`.
       * This is typically used in the block to generate slices that are to be
       * filled in by the class hierarchies, when filling in a block
       * incrementally. With the default size, it is typically used in the
       * block to generate slices that are to be filled in by ghost values.
       *
       * \param data        Data to be sliced
       * \param start_index Start index
       * \param size        Size of the data
       */
      static inline decltype(auto) slice(data_type& data,
                                         std::size_t start_index,
                                         std::size_t size) {
        // (page, row, column), (o, m, n)
        return blaze::subtensor(data, 0UL, 0UL, start_index, Frames::Dimension,
                                Frames::Dimension, size,
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
       * incrementally. With the default size, it is typically used in the
       * block to generate slices that are to be filled in by ghost values.
       *
       * \param data        Data to be sliced
       * \param start_index Start index
       * \param size        Size of the data
       */
      static inline decltype(auto) slice(data_type const& data,
                                         std::size_t start_index,
                                         std::size_t size) {
        // (page, row, column), (o, m, n)
        return blaze::subtensor(data, 0UL, 0UL, start_index, Frames::Dimension,
                                Frames::Dimension, size,
                                blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Assign diagonal elements
       *
       * @return
       */
      //************************************************************************
      static inline void diagonal_assign(slice_type& data, std::size_t index,
                                         Vec3 const& diag) {
        auto submatrix = blaze::columnslice(
            data, index, blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
        submatrix(0UL, 0UL) = static_cast<Real>(diag[0UL]);
        submatrix(1UL, 1UL) = static_cast<Real>(diag[1UL]);
        submatrix(2UL, 2UL) = static_cast<Real>(diag[2UL]);
      }

      static inline void diagonal_assign(data_type& data, std::size_t index,
                                         Vec3 const& diag) {
        auto submatrix = blaze::columnslice(
            data, index, blaze::ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK);
        submatrix(0UL, 0UL) = static_cast<Real>(diag[0UL]);
        submatrix(1UL, 1UL) = static_cast<Real>(diag[1UL]);
        submatrix(2UL, 2UL) = static_cast<Real>(diag[2UL]);
      }

      //@}
      //************************************************************************
    };
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <blaze/math/DynamicMatrix.h>
#include <blaze_tensor/math/DynamicTensor.h>
#include <cstdint>  // std::size_t

#include "Systems/States/Expressions/backends/Declarations.hpp"

namespace elastica {

  namespace states {

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    template <typename T>
    inline auto size_backend(
        blaze::DynamicTensor<T> const& tensor_batch) noexcept -> std::size_t {
      return tensor_batch.columns();
    }

    template <typename Type,   // Data type of the matrix
              bool SO,         // Storage order
              typename Alloc,  // Type of the allocator
              typename Tag>    // Type tag
    inline auto size_backend(
        blaze::DynamicMatrix<Type, SO, Alloc, Tag> const& matrix_batch) noexcept
        -> std::size_t {
      return matrix_batch.columns();
    }
    /*! \endcond */
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

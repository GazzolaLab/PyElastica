#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cstdint> // std::size_t

#include <blaze/math/DynamicMatrix.h>
#include <blaze_tensor/math/DynamicTensor.h>

//
// this comes from the simulator module, which seems like an anti-pattern
// should be in Systems instead
#include "Simulator/Frames.hpp"
//
#include "Systems/States/Expressions/backends/Declarations.hpp"

namespace elastica {

  namespace states {

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    // the parts below are repeated from the Block module, but they are almost
    // always never used in the states module since the size is always expected
    // to be
    template <typename T>
    inline auto resize_backend(blaze::DynamicTensor<T>& data,
                               std::size_t new_size) -> void {
      return data.resize(Frames::Dimension, Frames::Dimension, new_size, true);
    }

    template <typename Type,   // Data type of the matrix
              bool SO,         // Storage order
              typename Alloc,  // Type of the allocator
              typename Tag>    // Type tag
    inline auto resize_backend(blaze::DynamicMatrix<Type, SO, Alloc, Tag>& data,
                               std::size_t new_size) -> void {
      return data.resize(Frames::Dimension, new_size, true);
    }
    /*! \endcond */
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

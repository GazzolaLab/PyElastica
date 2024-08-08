#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cstddef> // std::size_t

namespace elastica {

  namespace states {

    // assign version doesnt care about vectorization
    template <typename LHSMatrix, typename RHSMatrix, typename RHSVector>
    auto SO3_primitive_assign(LHSMatrix& lhs, RHSMatrix const& rhs_matrix,
                              RHSVector const& rhs_vector) noexcept -> void;
    // add assign version doesnt care about vectorization
    template <typename LHSMatrix, typename RHSVector>
    auto SO3_primitive_add_assign(LHSMatrix& lhs,
                                  RHSVector const& rhs_vector) noexcept -> void;
    template <typename Type>
    inline auto size_backend(Type const&) noexcept -> std::size_t;
    template <typename Type>
    inline auto resize_backend(Type&, std::size_t) -> void;

  }  // namespace states

}  // namespace elastica

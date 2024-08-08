#include <pybind11/pybind11.h>
#include <blaze/Math.h>

#include "Systems/States/Expressions/backends/blaze/SO3PrimitiveAddAssign/Scalar.hpp"
#include "Systems/States/Expressions/backends/blaze/SO3PrimitiveAddAssign/SIMD.hpp"

#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/InvRotateDivide/Scalar.hpp"
#include "Systems/CosseratRods/Traits/Operations/BlazeBackend/InvRotateDivide/SIMD.hpp"

namespace detail = elastica::cosserat_rod::detail;
namespace states = elastica::states;

using ElasticaVector = ::blaze::DynamicVector<double, ::blaze::columnVector,
                                              ::blaze::AlignedAllocator<double>>;
using ElasticaMatrix = ::blaze::DynamicMatrix<double, ::blaze::rowMajor,
                                              ::blaze::AlignedAllocator<double>>;
using ElasticaTensor = ::blaze::DynamicTensor<double>;

template <states::detail::SO3AddAssignKind T>
void rotate(ElasticaTensor &director_collection, ElasticaMatrix &axis_collection)
{
    states::detail::SO3AddAssign<T>::apply(director_collection, axis_collection);
}

template <detail::InvRotateDivideKind T>
ElasticaMatrix inv_rotate_with_span(ElasticaTensor &director_collection, ElasticaVector &span_vector)
{
    ElasticaMatrix vector_collection(director_collection.rows(), director_collection.columns() - 1);
    detail::InvRotateDivideOp<T>::apply(vector_collection, director_collection, span_vector);
    return vector_collection;
}

// Overloaded function where span_vector is filled with 1
template <detail::InvRotateDivideKind T>
ElasticaMatrix inv_rotate(ElasticaTensor &director_collection)
{
    ElasticaVector span_vector(director_collection.columns(), 1);
    return inv_rotate_with_span<T>(director_collection, span_vector);
}

PYBIND11_MODULE(_rotations, m)
{
    m.doc() = R"pbdoc(
        elasticapp _rotations
        ---------------

        .. currentmodule:: _rotations

        .. autosummary::
           :toctree: _generate

           rotate
           rotate_scalar
           inv_rotate
           inv_rotate_scalar
    )pbdoc";

    m.def("rotate", &rotate<states::detail::SO3AddAssignKind::simd>, R"pbdoc(
        Perform the rotate operation (SIMD Variant).
    )pbdoc");
    m.def("rotate_scalar", &rotate<states::detail::SO3AddAssignKind::scalar>, R"pbdoc(
        Perform the rotate operation (Scalar Variant).
    )pbdoc");

    m.def("inv_rotate", &inv_rotate<detail::InvRotateDivideKind::simd>, R"pbdoc(
        Perform the inverse-rotate operation (SIMD Variant).
    )pbdoc");
    m.def("inv_rotate", &inv_rotate_with_span<detail::InvRotateDivideKind::simd>, R"pbdoc(
        Perform the inverse-rotate operation (SIMD Variant).
        This overload also accepts a vector (as the second vector) to perform elementwise division.
    )pbdoc");

    m.def("inv_rotate_scalar", &inv_rotate<detail::InvRotateDivideKind::scalar>, R"pbdoc(
        Perform the inverse-rotate operation (Scalar Variant).
    )pbdoc");
    m.def("inv_rotate_scalar", &inv_rotate_with_span<detail::InvRotateDivideKind::scalar>, R"pbdoc(
        Perform the inverse-rotate operation (Scalar Variant).
        This overload also accepts a vector (as the second vector) to perform elementwise division.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

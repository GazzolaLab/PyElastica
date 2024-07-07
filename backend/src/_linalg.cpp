#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <blaze/Math.h>

#include "Utilities/Math/BlazeDetail/BlazeLinearAlgebra.hpp"

namespace py = pybind11;

using blaze::DynamicMatrix;
using blaze::DynamicTensor;
using blaze::StaticMatrix;

using ElasticaVector = ::blaze::DynamicVector<double, ::blaze::columnVector,
                                              ::blaze::AlignedAllocator<double>>;
using ElasticaMatrix = ::blaze::DynamicMatrix<double, ::blaze::rowMajor,
                                              ::blaze::AlignedAllocator<double>>;
using ElasticaTensor = ::blaze::DynamicTensor<double>;

ElasticaMatrix difference_kernel(ElasticaMatrix &vector_batch)
{
    return elastica::difference_kernel(vector_batch);
}

ElasticaMatrix batch_matvec(
    ElasticaTensor &matrix_collection, ElasticaMatrix &vector_collection)
{
    ElasticaMatrix blaze_output(matrix_collection.rows(), matrix_collection.columns());
    elastica::batch_matvec(blaze_output, matrix_collection, vector_collection);
    return blaze_output;
}

ElasticaTensor batch_matmul(
    ElasticaTensor &first_matrix_batch, ElasticaTensor &second_matrix_batch)
{
    ElasticaTensor blaze_output(first_matrix_batch.pages(), first_matrix_batch.rows(), first_matrix_batch.columns());
    elastica::batch_matmul(blaze_output, first_matrix_batch, second_matrix_batch);
    return blaze_output;
}

ElasticaMatrix batch_cross(
    ElasticaMatrix &first_vector_batch, ElasticaMatrix &second_vector_batch)
{
    ElasticaMatrix blaze_output(first_vector_batch.rows(), first_vector_batch.columns());
    elastica::batch_cross(blaze_output, first_vector_batch, second_vector_batch);
    return blaze_output;
}

ElasticaVector batch_dot(
    ElasticaMatrix &first_vector_batch, ElasticaMatrix &second_vector_batch)
{
    return elastica::batch_dot(first_vector_batch, second_vector_batch);
}

ElasticaVector batch_norm(ElasticaMatrix &vector_batch)
{
    return elastica::batch_norm(vector_batch);
}

PYBIND11_MODULE(_linalg, m)
{
    m.doc() = R"pbdoc(
        elasticapp _linalg
        ---------------

        .. currentmodule:: _linalg

        .. autosummary::
           :toctree: _generate

           batch_matmul_naive
           batch_matmul_blaze
           difference_kernel
           batch_matvec
           batch_matmul
           batch_cross
           batch_dot
           batch_norm
    )pbdoc";

    m.def("difference_kernel", &difference_kernel, R"pbdoc(
        Vector Difference
    )pbdoc");

    m.def("batch_matvec", &batch_matvec, R"pbdoc(
        Computes a batchwise matrix-vector product given in indical notation:
            matvec_batch{ik} = matrix_batch{ijk} * vector_batch{jk}
    )pbdoc");

    m.def("batch_matmul", &batch_matmul, R"pbdoc(
        Computes a batchwise matrix-matrix product given in indical notation:
            matmul_batch{ilk} = first_matrix_batch{ijk} * second_matrix_batch{jlk}
    )pbdoc");

    m.def("batch_cross", &batch_cross, R"pbdoc(
        Computes a batchwise vector-vector cross product given in indical notation:
            cross_batch{il} = LCT{ijk} * first_vector_batch{jl} * second_vector_batch{kl}
        where LCT is the Levi-Civita Tensor
    )pbdoc");

    m.def("batch_dot", &batch_dot, R"pbdoc(
        Computes a batchwise vector-vector dot product given in indical notation:
            dot_batch{j} = first_vector_batch{ij} * second_vector_batch{ij}
    )pbdoc");

    m.def("batch_norm", &batch_norm, R"pbdoc(
        Computes a batchwise vector L2 norm given in indical notation:
            norm_batch{j} = (vector_batch{ij} * vector_batch{ij})^0.5
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

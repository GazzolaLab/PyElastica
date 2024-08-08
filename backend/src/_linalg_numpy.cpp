#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <blaze/Math.h>

#include "Utilities/Math/BlazeDetail/BlazeLinearAlgebra.hpp"

namespace py = pybind11;

using blaze::DynamicMatrix;
using blaze::DynamicTensor;
using blaze::StaticMatrix;

/* Simple rewrite of elastica._linalg._batch_matmul */
py::array_t<double> batch_matmul_naive(
    py::array_t<double> first_matrix_collection,
    py::array_t<double> second_matrix_collection)
{
    auto s1 = first_matrix_collection.shape(0);
    auto s2 = first_matrix_collection.shape(1);
    auto max_k = first_matrix_collection.shape(2);
    auto output_matrix = py::array_t<double>{{s1, s2, max_k}};

    auto a = first_matrix_collection.unchecked<3>();
    auto b = second_matrix_collection.unchecked<3>();
    auto c = output_matrix.mutable_unchecked<3>();
    for (py::ssize_t i = 0; i < 3; i++)
    {
        for (py::ssize_t j = 0; j < 3; j++)
        {
            for (py::ssize_t m = 0; m < 3; m++)
            {
                for (py::ssize_t k = 0; k < max_k; k++)
                {
                    c(i, m, k) += a(i, j, k) * b(j, m, k);
                }
            }
        }
    }
    return output_matrix;
}

/* blaze implementation of elastica._linalg._batch_matmul */
py::array_t<double> batch_matmul_blaze(
    py::array_t<double> first_matrix_collection,
    py::array_t<double> second_matrix_collection)
{
    auto s1 = first_matrix_collection.shape(0);
    auto s2 = first_matrix_collection.shape(1);
    auto max_k = first_matrix_collection.shape(2);
    auto output_matrix = py::array_t<double>{{s1, s2, max_k}};

    auto a = first_matrix_collection.unchecked<3>();
    auto b = second_matrix_collection.unchecked<3>();
    auto c = output_matrix.mutable_unchecked<3>();

    StaticMatrix<double, 3UL, 3UL> blaze_a;
    StaticMatrix<double, 3UL, 3UL> blaze_b;
    StaticMatrix<double, 3UL, 3UL> blaze_c;
    for (py::ssize_t k = 0; k < max_k; k++)
    {
        for (py::ssize_t i = 0; i < 3; i++)
        {
            for (py::ssize_t j = 0; j < 3; j++)
            {
                blaze_a(i, j) = a(i, j, k);
                blaze_b(i, j) = b(i, j, k);
            }
        }
        blaze_c = blaze_a * blaze_b;
        for (py::ssize_t i = 0; i < 3; i++)
        {
            for (py::ssize_t j = 0; j < 3; j++)
            {
                c(i, j, k) = blaze_c(i, j);
            }
        }
    }

    return output_matrix;
}

py::array_t<double> difference_kernel(py::array_t<double> vector_batch)
{
    const auto v0 = vector_batch.shape(0);
    const auto num_elems = vector_batch.shape(1);

    assert(v0 == 3UL);
    auto output_arr = py::array_t<double>{{v0, num_elems - 1}};

    auto vector_batch_unchecked = vector_batch.unchecked<2>();
    auto output_arr_unchecked = output_arr.mutable_unchecked<2>();

    DynamicMatrix<double> blaze_vector_batch(v0, num_elems);
    for (py::ssize_t j = 0; j < num_elems; j++)
    {
        for (py::ssize_t i = 0; i < 3; i++)
        {
            blaze_vector_batch(i, j) = vector_batch_unchecked(i, j);
        }
    }
    auto blaze_output = elastica::difference_kernel(blaze_vector_batch);
    for (py::ssize_t j = 0; j < num_elems - 1; j++)
    {
        for (py::ssize_t i = 0; i < 3; i++)
        {
            output_arr_unchecked(i, j) = blaze_output(i, j);
        }
    }

    return output_arr;
}

py::array_t<double> batch_matvec(
    py::array_t<double> matrix_collection, py::array_t<double> vector_collection)
{
    const auto v0 = vector_collection.shape(0);
    const auto num_elems = vector_collection.shape(1);
    assert(v0 == 3UL);

    assert(matrix_collection.shape(0) == 3UL);
    assert(matrix_collection.shape(1) == 3UL);
    assert(matrix_collection.shape(2) == num_elems);

    auto output_arr = py::array_t<double>{{v0, num_elems}};

    auto matrix_collection_unchecked = matrix_collection.unchecked<3>();
    auto vector_collection_unchecked = vector_collection.unchecked<2>();
    auto output_arr_unchecked = output_arr.mutable_unchecked<2>();

    DynamicTensor<double> blaze_matrix_collection(v0, v0, num_elems);
    DynamicMatrix<double> blaze_vector_collection(v0, num_elems);
    for (py::ssize_t k = 0; k < num_elems; k++)
    {
        for (py::ssize_t j = 0; j < 3; j++)
        {
            for (py::ssize_t i = 0; i < 3; i++)
            {
                blaze_matrix_collection(i, j, k) = matrix_collection_unchecked(i, j, k);
            }
            blaze_vector_collection(j, k) = vector_collection_unchecked(j, k);
        }
    }

    DynamicMatrix<double> blaze_output(v0, num_elems);
    elastica::batch_matvec(blaze_output, blaze_matrix_collection, blaze_vector_collection);
    for (py::ssize_t j = 0; j < num_elems; j++)
    {
        for (py::ssize_t i = 0; i < 3; i++)
        {
            output_arr_unchecked(i, j) = blaze_output(i, j);
        }
    }

    return output_arr;
}

py::array_t<double> batch_matmul(
    py::array_t<double> first_matrix_batch, py::array_t<double> second_matrix_batch)
{
    const auto m0 = first_matrix_batch.shape(0);
    const auto m1 = first_matrix_batch.shape(1);
    const auto num_elems = first_matrix_batch.shape(2);
    assert(m0 == 3UL);
    assert(m1 == 3UL);

    assert(second_matrix_batch.shape(0) == 3UL);
    assert(second_matrix_batch.shape(1) == 3UL);
    assert(second_matrix_batch.shape(2) == num_elems);

    auto output_arr = py::array_t<double>{{m0, m1, num_elems}};

    auto first_matrix_batch_unchecked = first_matrix_batch.unchecked<3>();
    auto second_matrix_batch_unchecked = second_matrix_batch.unchecked<3>();
    auto output_arr_unchecked = output_arr.mutable_unchecked<3>();

    DynamicTensor<double> blaze_first_matrix_batch(m0, m1, num_elems);
    DynamicTensor<double> blaze_second_matrix_batch(m0, m1, num_elems);
    for (py::ssize_t k = 0; k < num_elems; k++)
    {
        for (py::ssize_t j = 0; j < 3; j++)
        {
            for (py::ssize_t i = 0; i < 3; i++)
            {
                blaze_first_matrix_batch(i, j, k) = first_matrix_batch_unchecked(i, j, k);
                blaze_second_matrix_batch(i, j, k) = second_matrix_batch_unchecked(i, j, k);
            }
        }
    }
    DynamicTensor<double> blaze_output(m0, m1, num_elems);
    elastica::batch_matmul(blaze_output, blaze_first_matrix_batch, blaze_second_matrix_batch);
    for (py::ssize_t k = 0; k < num_elems; k++)
    {
        for (py::ssize_t j = 0; j < 3; j++)
        {
            for (py::ssize_t i = 0; i < 3; i++)
            {
                output_arr_unchecked(i, j, k) = blaze_output(i, j, k);
            }
        }
    }
    return output_arr;
}

py::array_t<double> batch_cross(
    py::array_t<double> first_vector_batch, py::array_t<double> second_vector_batch)
{
    const auto v0 = first_vector_batch.shape(0);
    const auto num_elems = first_vector_batch.shape(1);
    assert(v0 == 3UL);

    assert(second_vector_batch.shape(0) == 3UL);
    assert(second_vector_batch.shape(1) == num_elems);

    auto output_arr = py::array_t<double>{{v0, num_elems}};

    auto first_vector_batch_unchecked = first_vector_batch.unchecked<2>();
    auto second_vector_batch_unchecked = second_vector_batch.unchecked<2>();
    auto output_arr_unchecked = output_arr.mutable_unchecked<2>();

    DynamicMatrix<double> blaze_first_vector_batch(v0, num_elems);
    DynamicMatrix<double> blaze_second_vector_batch(v0, num_elems);
    for (py::ssize_t j = 0; j < num_elems; j++)
    {
        for (py::ssize_t i = 0; i < 3; i++)
        {
            blaze_first_vector_batch(i, j) = first_vector_batch_unchecked(i, j);
            blaze_second_vector_batch(i, j) = second_vector_batch_unchecked(i, j);
        }
    }

    DynamicMatrix<double> blaze_output(v0, num_elems);
    elastica::batch_cross(blaze_output, blaze_first_vector_batch, blaze_second_vector_batch);
    for (py::ssize_t j = 0; j < num_elems; j++)
    {
        for (py::ssize_t i = 0; i < 3; i++)
        {
            output_arr_unchecked(i, j) = blaze_output(i, j);
        }
    }

    return output_arr;
}

py::array_t<double> batch_dot(
    py::array_t<double> first_vector_batch, py::array_t<double> second_vector_batch)
{
    const auto v0 = first_vector_batch.shape(0);
    const auto num_elems = first_vector_batch.shape(1);
    assert(v0 == 3UL);

    assert(second_vector_batch.shape(0) == 3UL);
    assert(second_vector_batch.shape(1) == num_elems);

    auto output_arr = py::array_t<double>{num_elems};

    auto first_vector_batch_unchecked = first_vector_batch.unchecked<2>();
    auto second_vector_batch_unchecked = second_vector_batch.unchecked<2>();
    auto output_arr_unchecked = output_arr.mutable_unchecked<1>();

    DynamicMatrix<double> blaze_first_vector_batch(v0, num_elems);
    DynamicMatrix<double> blaze_second_vector_batch(v0, num_elems);
    for (py::ssize_t j = 0; j < num_elems; j++)
    {
        for (py::ssize_t i = 0; i < 3; i++)
        {
            blaze_first_vector_batch(i, j) = first_vector_batch_unchecked(i, j);
            blaze_second_vector_batch(i, j) = second_vector_batch_unchecked(i, j);
        }
    }

    auto blaze_output = elastica::batch_dot(blaze_first_vector_batch, blaze_second_vector_batch);
    for (py::ssize_t j = 0; j < num_elems; j++)
    {
        output_arr_unchecked(j) = blaze_output[j];
    }

    return output_arr;
}

py::array_t<double> batch_norm(py::array_t<double> vector_batch)
{
    const auto v0 = vector_batch.shape(0);
    const auto num_elems = vector_batch.shape(1);

    assert(v0 == 3UL);
    
    auto output_arr = py::array_t<double>{num_elems};

    auto vector_batch_unchecked = vector_batch.unchecked<2>();
    auto output_arr_unchecked = output_arr.mutable_unchecked<1>();

    DynamicMatrix<double> blaze_vector_batch(v0, num_elems);
    for (py::ssize_t j = 0; j < num_elems; j++)
    {
        for (py::ssize_t i = 0; i < 3; i++)
        {
            blaze_vector_batch(i, j) = vector_batch_unchecked(i, j);
        }
    }

    auto blaze_output = elastica::batch_norm(blaze_vector_batch);
    for (py::ssize_t j = 0; j < num_elems; j++)
    {
        output_arr_unchecked(j) = blaze_output[j];
    }

    return output_arr;
}

PYBIND11_MODULE(_linalg_numpy, m)
{
    m.doc() = R"pbdoc(
        elasticapp _linalg_numpy
        ---------------

        .. currentmodule:: _linalg_numpy

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

    m.def("batch_matmul_naive", &batch_matmul_naive, R"pbdoc(
        This is batch matrix matrix multiplication function. Only batch
        of 3x3 matrices can be multiplied.
    )pbdoc");

    m.def("batch_matmul_blaze", &batch_matmul_blaze, R"pbdoc(
        This is batch matrix matrix multiplication function. Only batch
        of 3x3 matrices can be multiplied.
    )pbdoc");

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

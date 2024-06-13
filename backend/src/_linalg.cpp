#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <blaze/Math.h>

namespace py = pybind11;

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

PYBIND11_MODULE(_linalg, m)
{
    m.doc() = R"pbdoc(
        elasticapp _linalg
        ---------------

        .. currentmodule:: _linalg

        .. autosummary::
           :toctree: _generate

           batch_matmul_naive
    )pbdoc";

    m.def("batch_matmul_naive", &batch_matmul_naive, R"pbdoc(
        This is batch matrix matrix multiplication function. Only batch
        of 3x3 matrices can be multiplied.
    )pbdoc");

    m.def("batch_matmul_blaze", &batch_matmul_blaze, R"pbdoc(
        This is batch matrix matrix multiplication function. Only batch
        of 3x3 matrices can be multiplied.
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
# #!/usr/bin/env python3
# """ Test scripts for linear algebra helpers in elastica
# """
# # System imports
# import numpy as np
# import pytest
# from numpy.testing import assert_allclose
#
# # from elastica._linalg import (
# #     _batch_matvec,
# #     _batch_cross,
# #     _batch_matmul,
# #     levi_civita_tensor,
# # )
#
# try:
#     from elastica._linalg import (
#         _batch_matvec,
#         _batch_matmul,
#         _batch_cross,
#         _batch_vec_oneD_vec_cross,
#         _batch_dot,
#         _batch_norm,
#         _batch_product_i_k_to_ik,
#         _batch_product_i_ik_to_k,
#         _batch_product_k_ik_to_ik,
#         _batch_vector_sum,
#         _batch_matrix_transpose,
#     )
#
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_matvec(blocksize):
#         input_matrix_collection = np.random.randn(3, 3, blocksize)
#         input_vector_collection = np.random.randn(3, blocksize)
#
#         test_vector_collection = _batch_matvec(
#             input_matrix_collection, input_vector_collection
#         )
#
#         correct_vector_collection = [
#             np.dot(input_matrix_collection[..., i], input_vector_collection[..., i])
#             for i in range(blocksize)
#         ]
#         correct_vector_collection = np.array(correct_vector_collection).T
#
#         assert_allclose(test_vector_collection, correct_vector_collection)
#
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_matmul(blocksize):
#         input_first_matrix_collection = np.random.randn(3, 3, blocksize)
#         input_second_matrix_collection = np.random.randn(3, 3, blocksize)
#
#         test_matrix_collection = _batch_matmul(
#             input_first_matrix_collection, input_second_matrix_collection
#         )
#
#         correct_matrix_collection = np.empty((3, 3, blocksize))
#         for i in range(blocksize):
#             correct_matrix_collection[..., i] = np.dot(
#                 input_first_matrix_collection[..., i],
#                 input_second_matrix_collection[..., i],
#             )
#
#         assert_allclose(test_matrix_collection, correct_matrix_collection)
#
#     # TODO : Generalize to two dimensions
#     @pytest.mark.parametrize("dim", [3])
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_cross(dim, blocksize):
#         input_first_vector_collection = np.random.randn(dim, blocksize)
#         input_second_vector_collection = np.random.randn(dim, blocksize)
#
#         test_vector_collection = _batch_cross(
#             input_first_vector_collection, input_second_vector_collection
#         )
#         correct_vector_collection = np.cross(
#             input_first_vector_collection,
#             input_second_vector_collection,
#             axisa=0,
#             axisb=0,
#         ).T
#
#         assert_allclose(test_vector_collection, correct_vector_collection)
#
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_vec_oneD_vec_cross(blocksize):
#         input_first_vector_collection = np.random.randn(3, blocksize)
#         input_second_vector = np.random.randn(3)
#
#         test_vector_collection = _batch_vec_oneD_vec_cross(
#             input_first_vector_collection, input_second_vector
#         )
#
#         correct_vector_collection = np.cross(
#             input_first_vector_collection, input_second_vector, axisa=0, axisb=0,
#         ).T
#
#         assert_allclose(test_vector_collection, correct_vector_collection)
#
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_dot(blocksize):
#         input_first_vector_collection = np.random.randn(3, blocksize)
#         input_second_vector_collection = np.random.randn(3, blocksize)
#
#         test_vector_collection = _batch_dot(
#             input_first_vector_collection, input_second_vector_collection
#         )
#
#         correct_vector_collection = np.einsum(
#             "ij,ij->j", input_first_vector_collection, input_second_vector_collection
#         )
#
#         assert_allclose(test_vector_collection, correct_vector_collection)
#
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_norm(blocksize):
#         input_first_vector_collection = np.random.randn(3, blocksize)
#
#         test_vector_collection = _batch_norm(input_first_vector_collection)
#
#         correct_vector_collection = np.sqrt(
#             np.einsum(
#                 "ij,ij->j", input_first_vector_collection, input_first_vector_collection
#             )
#         )
#
#         assert_allclose(test_vector_collection, correct_vector_collection)
#
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_product_i_k_to_ik(blocksize):
#         input_first_vector_collection = np.random.randn(3)
#         input_second_vector_collection = np.random.randn(blocksize)
#
#         test_vector_collection = _batch_product_i_k_to_ik(
#             input_first_vector_collection, input_second_vector_collection
#         )
#
#         correct_vector_collection = np.einsum(
#             "i,j->ij", input_first_vector_collection, input_second_vector_collection
#         )
#
#         assert_allclose(test_vector_collection, correct_vector_collection)
#
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_product_i_ik_to_k(blocksize):
#         input_first_vector_collection = np.random.randn(3)
#         input_second_vector_collection = np.random.randn(3, blocksize)
#
#         test_vector_collection = _batch_product_i_ik_to_k(
#             input_first_vector_collection, input_second_vector_collection
#         )
#
#         correct_vector_collection = np.einsum(
#             "i,ij->j", input_first_vector_collection, input_second_vector_collection
#         )
#
#         assert_allclose(test_vector_collection, correct_vector_collection)
#
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_product_k_ik_to_ik(blocksize):
#         input_first_vector_collection = np.random.randn(blocksize)
#         input_second_vector_collection = np.random.randn(3, blocksize)
#
#         test_vector_collection = _batch_product_k_ik_to_ik(
#             input_first_vector_collection, input_second_vector_collection
#         )
#
#         correct_vector_collection = np.einsum(
#             "j,ij->ij", input_first_vector_collection, input_second_vector_collection
#         )
#
#         assert_allclose(test_vector_collection, correct_vector_collection)
#
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_vector_sum(blocksize):
#         input_first_vector_collection = np.random.randn(3, blocksize)
#         input_second_vector_collection = np.random.randn(3, blocksize)
#
#         test_vector_collection = _batch_vector_sum(
#             input_first_vector_collection, input_second_vector_collection
#         )
#
#         correct_vector_collection = (
#             input_first_vector_collection + input_second_vector_collection
#         )
#
#         assert_allclose(test_vector_collection, correct_vector_collection)
#
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_matrix_transpose(blocksize):
#         input_matrix_collection = np.random.randn(3, 3, blocksize)
#
#         test_matrix_collection = _batch_matrix_transpose(input_matrix_collection)
#
#         correct_matrix_collection = np.einsum("ijk->jik", input_matrix_collection)
#
#         assert_allclose(test_matrix_collection, correct_matrix_collection)
#
#
# except ImportError:
#     from elastica._linalg import (
#         _batch_matvec,
#         _batch_cross,
#         _batch_matmul,
#         levi_civita_tensor,
#     )
#
#     # NOTE : Testing Levi-Civita only for commonly used cases of two and three dimensions
#     @pytest.mark.parametrize("dim", [2, 3])
#     def test_levi_civita_first_index_product(dim):
#         lct = levi_civita_tensor(dim)
#         if dim == 2:
#             contraction_operator = "ij,ik->jk"
#             correct_tensor = np.eye(dim)
#         else:
#             contraction_operator = "imn,jmn->ij"
#             correct_tensor = 2 * np.eye(dim)
#
#         test_tensor = np.einsum(contraction_operator, lct, lct)
#
#         assert_allclose(test_tensor, correct_tensor)
#
#     @pytest.mark.parametrize("dim", [2, 3])
#     def test_levi_civita_inner_product(dim):
#         lct = levi_civita_tensor(dim)
#
#         if dim == 2:
#             contraction_operator = "ij,ij->"
#             correct_scalar = 2.0
#         else:
#             contraction_operator = "ijk,ijk->"
#             correct_scalar = 6.0
#
#         test_scalar = np.einsum(contraction_operator, lct, lct)
#
#         assert_allclose(test_scalar, correct_scalar)
#
#     @pytest.mark.parametrize("dim", [2, 3])
#     def test_levi_civita_correctness(dim):
#         test_tensor = levi_civita_tensor(dim)
#
#         if dim == 2:
#             correct_tensor = np.array([[0.0, 1.0], [-1.0, 0.0]])
#         else:
#             correct_tensor = np.zeros((3, 3, 3))
#             correct_tensor[0] = np.array(
#                 [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
#             )
#             correct_tensor[1] = np.array(
#                 [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
#             )
#             correct_tensor[2] = np.array(
#                 [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -0.0, 0.0]]
#             )
#
#         assert_allclose(test_tensor, correct_tensor)
#
#     # @pytest.mark.parametrize("dim", [2, 3])
#     @pytest.mark.parametrize("dim", [3])
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_matvec(dim, blocksize):
#         for iteration in range(3):
#             input_matrix_collection = np.random.randn(dim, dim, blocksize)
#             input_vector_collection = np.random.randn(dim, blocksize)
#
#             test_vector_collection = _batch_matvec(
#                 input_matrix_collection, input_vector_collection
#             )
#
#             correct_vector_collection = [
#                 np.dot(input_matrix_collection[..., i], input_vector_collection[..., i])
#                 for i in range(blocksize)
#             ]
#             correct_vector_collection = np.array(correct_vector_collection).T
#
#             assert_allclose(test_vector_collection, correct_vector_collection)
#
#     # @pytest.mark.parametrize("dim", [2, 3])
#     @pytest.mark.parametrize("dim", [3])
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_matmul(dim, blocksize):
#         for iteration in range(3):
#             input_first_matrix_collection = np.random.randn(dim, dim, blocksize)
#             input_second_matrix_collection = np.random.randn(dim, dim, blocksize)
#
#             test_matrix_collection = _batch_matmul(
#                 input_first_matrix_collection, input_second_matrix_collection
#             )
#
#             correct_matrix_collection = np.empty((dim, dim, blocksize))
#             for i in range(blocksize):
#                 correct_matrix_collection[..., i] = np.dot(
#                     input_first_matrix_collection[..., i],
#                     input_second_matrix_collection[..., i],
#                 )
#
#             assert_allclose(test_matrix_collection, correct_matrix_collection)
#
#     # TODO : Generalize to two dimensions
#     @pytest.mark.parametrize("dim", [3])
#     @pytest.mark.parametrize("blocksize", [8, 32])
#     def test_batch_cross(dim, blocksize):
#         for iteration in range(3):
#             input_first_vector_collection = np.random.randn(dim, blocksize)
#             input_second_vector_collection = np.random.randn(dim, blocksize)
#
#             test_vector_collection = _batch_cross(
#                 input_first_vector_collection, input_second_vector_collection
#             )
#             correct_vector_collection = np.cross(
#                 input_first_vector_collection,
#                 input_second_vector_collection,
#                 axisa=0,
#                 axisb=0,
#             ).T
#
#             assert_allclose(test_vector_collection, correct_vector_collection)

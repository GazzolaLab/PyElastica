#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "block.h"
#include "block_view.h"
#include "traits.h"
#include "api.h"
#include "operations.h"
#include "cosserat_rod_system.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>
#include <chrono>
#include <Eigen/Core>

// Include OpenMP headers if threading is enabled
#ifdef ELASTICAPP_USE_THREADING
#include <omp.h>
#endif

namespace py = pybind11;

namespace elasticapp {

#ifdef ELASTICAPP_GIL_RELEASE
#define ELASTICAPP_GIL_RELEASE_SCOPE() py::gil_scoped_release gil_release;
#else
#define ELASTICAPP_GIL_RELEASE_SCOPE()
#endif

// Helper to find a variable type by name at runtime
// Iterates through SystemType::Variables tuple and finds matching name
template<SystemModel SystemType, typename VariablesTuple, std::size_t Index>
auto get_variable_by_name_impl(BlockView<SystemType>& view, const std::string& var_name) {
    using CurrentVar = std::tuple_element_t<Index, VariablesTuple>;

    // Check if current variable's name matches
    if (var_name == std::string(CurrentVar::name)) {
        return view.template get<CurrentVar>();
    }

    // Recurse to next variable if not last
    if constexpr (Index + 1 < std::tuple_size_v<VariablesTuple>) {
        return get_variable_by_name_impl<SystemType, VariablesTuple, Index + 1>(view, var_name);
    }

    // If we've exhausted all variables, throw error
    throw std::runtime_error("Unknown variable name: " + var_name);
}

// Helper function to map string variable names to types and call get<VariableType>()
// This allows Python to use block.at(index).get("position") syntax
template<SystemModel SystemType>
auto get_variable_by_name(BlockView<SystemType>& view, const std::string& var_name) {
    using VariablesTuple = typename SystemType::Variables;

    if constexpr (std::tuple_size_v<VariablesTuple> > 0) {
        return get_variable_by_name_impl<SystemType, VariablesTuple, 0>(view, var_name);
    } else {
        throw std::runtime_error("System has no variables");
    }
}

// Helper to find a variable type by name at runtime for Block
// Iterates through SystemType::Variables tuple and finds matching name
template<typename BlockType, typename VariablesTuple, std::size_t Index>
auto get_block_variable_by_name_impl(BlockType& block, const std::string& var_name) {
    using CurrentVar = std::tuple_element_t<Index, VariablesTuple>;

    // Check if current variable's name matches
    if (var_name == std::string(CurrentVar::name)) {
        return block.template get<CurrentVar>();
    }

    // Recurse to next variable if not last
    if constexpr (Index + 1 < std::tuple_size_v<VariablesTuple>) {
        return get_block_variable_by_name_impl<BlockType, VariablesTuple, Index + 1>(block, var_name);
    }

    // If we've exhausted all variables, throw error
    throw std::runtime_error("Unknown variable name: " + var_name);
}

// Helper function to map string variable names to types and call get<VariableType>() for Block
// This allows Python to use block.get("position") syntax
template<typename BlockType>
auto get_block_variable_by_name(BlockType& block, const std::string& var_name) {
    using VariablesTuple = typename BlockType::Variables;

    if constexpr (std::tuple_size_v<VariablesTuple> > 0) {
        return get_block_variable_by_name_impl<BlockType, VariablesTuple, 0>(block, var_name);
    } else {
        throw std::runtime_error("System has no variables");
    }
}

// Helper to reset ghost for a variable by name
template<typename BlockType, typename VariablesTuple, std::size_t Index>
void reset_ghost_for_variable_by_name_impl(BlockType& block, const std::string& var_name) {
    using CurrentVar = std::tuple_element_t<Index, VariablesTuple>;

    // Check if current variable's name matches
    if (var_name == std::string(CurrentVar::name)) {
        block.template reset_ghost_for_variable<CurrentVar>();
        return;
    }

    // Recurse to next variable if not last
    if constexpr (Index + 1 < std::tuple_size_v<VariablesTuple>) {
        reset_ghost_for_variable_by_name_impl<BlockType, VariablesTuple, Index + 1>(block, var_name);
    } else {
        throw std::runtime_error("Unknown variable name: " + var_name);
    }
}

// Helper function to reset ghost for a variable by name
template<typename BlockType>
void reset_ghost_for_variable_by_name(BlockType& block, const std::string& var_name) {
    using VariablesTuple = typename BlockType::Variables;

    if constexpr (std::tuple_size_v<VariablesTuple> > 0) {
        reset_ghost_for_variable_by_name_impl<BlockType, VariablesTuple, 0>(block, var_name);
    } else {
        throw std::runtime_error("System has no variables");
    }
}

// Helper to convert Eigen Block view to numpy array
// For Scalar variables (rows == 1), returns a 1D array instead of 2D
// For Matrix variables (rows == 9), returns a 3D array (3, 3, N) directly
template<typename BlockExpr>
py::object block_to_numpy(BlockExpr&& block_expr, py::object parent) {
    // Evaluate the expression to get dimensions
    auto rows = static_cast<py::ssize_t>(block_expr.rows());
    auto cols = static_cast<py::ssize_t>(block_expr.cols());

    // Get actual strides from the Eigen Block expression
    // For Eigen Blocks, innerStride() is the stride between elements in the same row/column
    // and outerStride() is the stride between rows/columns depending on storage order
    // For column-major: innerStride() = 1 (between rows), outerStride() = underlying_rows (between columns)
    // For row-major: innerStride() = 1 (between columns), outerStride() = underlying_cols (between rows)
    auto inner_stride = static_cast<py::ssize_t>(block_expr.innerStride() * sizeof(double));
    auto outer_stride = static_cast<py::ssize_t>(block_expr.outerStride() * sizeof(double));

    // For Scalar variables (rows == 1), return as 1D array
    if (rows == 1) {
        // For 1D array, stride is the column stride
        py::ssize_t stride;
        if constexpr (IsColMajor) {
            // Column-major: stride between columns is outer_stride
            stride = outer_stride;
        } else {
            // Row-major: stride between columns is inner_stride
            stride = inner_stride;
        }

        return py::array_t<double>(
            {cols},
            {stride},
            const_cast<double*>(block_expr.data()),
            parent  // Keep parent object alive
        );
    }

    // For Matrix variables (rows == 9), return as 3D array (3, 3, N)
    // Matrix variables represent 3x3 matrices stored as flattened 9-element vectors
    // Storage order: [m00, m10, m20, m01, m11, m21, m02, m12, m22] (column-major)
    if (rows == 9) {
        // For (3, 3, N) view from (9, N) column-major:
        // Mapping: arr_3d[a, b, c] -> arr_2d[a*3 + b, c]
        // Strides:
        //   - Page stride (a dimension): 3 * row_stride (to skip 3 rows)
        //   - Row stride (b dimension): row_stride (to skip 1 row)
        //   - Col stride (c dimension): col_stride (to skip 1 column)
        py::ssize_t page_stride, row_stride_3d, col_stride_3d;
        if constexpr (IsColMajor) {
            // Column-major: row_stride = inner_stride, col_stride = outer_stride
            page_stride = 3 * inner_stride;  // Stride for first 3x3 dimension (skip 3 rows)
            row_stride_3d = inner_stride;     // Stride between rows in 3x3 (skip 1 row)
            col_stride_3d = outer_stride;    // Stride between columns (N dimension)
        } else {
            // Row-major: row_stride = outer_stride, col_stride = inner_stride
            page_stride = 3 * outer_stride;
            row_stride_3d = outer_stride;
            col_stride_3d = inner_stride;
        }

        // Use py::buffer_info for 3D arrays with custom strides
        std::vector<py::ssize_t> shape = {3, 3, cols};
        std::vector<py::ssize_t> strides = {page_stride, row_stride_3d, col_stride_3d};
        py::buffer_info buf_info(
            const_cast<double*>(block_expr.data()),
            sizeof(double),
            py::format_descriptor<double>::format(),
            3,
            shape,
            strides
        );
        return py::array_t<double>(buf_info, parent);
    }

    // For Vector variables (rows == 3), return as 2D array
    // For numpy, strides are in bytes and represent the step size for each dimension
    // For column-major (Eigen default): row_stride = inner_stride, col_stride = outer_stride
    // For row-major: row_stride = outer_stride, col_stride = inner_stride
    py::ssize_t row_stride, col_stride;
    if constexpr (IsColMajor) {
        // Column-major: stride between rows is inner_stride, between columns is outer_stride
        row_stride = inner_stride;
        col_stride = outer_stride;
    } else {
        // Row-major: stride between rows is outer_stride, between columns is inner_stride
        row_stride = outer_stride;
        col_stride = inner_stride;
    }

    // Create numpy array view (non-owning) with correct strides
    return py::array_t<double>(
        {rows, cols},
        {row_stride, col_stride},
        const_cast<double*>(block_expr.data()),
        parent  // Keep parent object alive
    );
}

PYBIND11_MODULE(_memory_block, m) {
    m.doc() = R"pbdoc(
        Elasticapp BlockCosseratRod module
        ----------------------------------------

        Provides BlockCosseratRod class for 2D array management with Eigen backend.
    )pbdoc";

    // Forward declare BlockRodSystemView so it can be used as a return type
    using BlockRodSystemViewType = BlockRodSystem::View;

    // Thread management functions (only available when threading is enabled)
    #ifdef ELASTICAPP_USE_THREADING
    // Disable Eigen's internal threading to prevent oversubscription with OpenMP
    // We use OpenMP for explicit parallelization, so Eigen should use single-threaded operations
    Eigen::setNbThreads(1);

    m.def("set_num_threads", [](int num_threads) {
        if (num_threads > 0) {
            omp_set_num_threads(num_threads);
            // Ensure Eigen uses single thread to avoid oversubscription
            Eigen::setNbThreads(1);
        }
    },
    R"pbdoc(
        Set the number of OpenMP threads to use for parallel operations.

        Args:
            num_threads: Number of threads to use (must be > 0).
                        If 0 or negative, OpenMP will use its default
                        (typically all available CPU cores).

        Note:
            This affects all subsequent parallel regions in the code.
            The environment variable OMP_NUM_THREADS can also be used
            to control thread count.
    )pbdoc",
    py::arg("num_threads"));

    m.def("get_num_threads", []() {
        return omp_get_num_threads();
    },
    R"pbdoc(
        Get the current number of threads in the current parallel region.

        Returns:
            int: Number of threads (returns 1 if called outside a parallel region).
    )pbdoc");

    m.def("get_max_threads", []() {
        return omp_get_max_threads();
    },
    R"pbdoc(
        Get the maximum number of threads that can be used.

        Returns:
            int: Maximum number of threads available.
    )pbdoc");

    m.def("get_thread_num", []() {
        return omp_get_thread_num();
    },
    R"pbdoc(
        Get the current thread number (0 to num_threads-1).

        Returns:
            int: Current thread number (returns 0 if called outside a parallel region).
    )pbdoc");
    #endif // ELASTICAPP_USE_THREADING

    // BlockRodSystem class
    py::class_<BlockRodSystem>(m, "BlockRodSystem")
        .def(py::init([](py::object n_elems_per_rod_obj) {
            std::vector<std::size_t> n_elems_per_rod;

            // Handle list, tuple, or numpy array
            if (py::isinstance<py::list>(n_elems_per_rod_obj)) {
                py::list lst = n_elems_per_rod_obj.cast<py::list>();
                for (auto item : lst) {
                    n_elems_per_rod.push_back(item.cast<std::size_t>());
                }
            } else if (py::isinstance<py::tuple>(n_elems_per_rod_obj)) {
                py::tuple tup = n_elems_per_rod_obj.cast<py::tuple>();
                for (auto item : tup) {
                    n_elems_per_rod.push_back(item.cast<std::size_t>());
                }
            } else if (py::isinstance<py::array>(n_elems_per_rod_obj)) {
                py::array arr = n_elems_per_rod_obj.cast<py::array>();
                auto buf = arr.request();
                if (buf.ndim != 1) {
                    throw std::runtime_error("numpy array must be 1-dimensional");
                }
                // Properly convert numpy array elements to std::size_t
                // Handle different integer types safely
                n_elems_per_rod.reserve(buf.size);
                if (buf.itemsize == sizeof(std::int32_t)) {
                    auto* ptr = static_cast<std::int32_t*>(buf.ptr);
                    for (py::ssize_t i = 0; i < buf.size; ++i) {
                        n_elems_per_rod.push_back(static_cast<std::size_t>(ptr[i]));
                    }
                } else if (buf.itemsize == sizeof(std::int64_t)) {
                    auto* ptr = static_cast<std::int64_t*>(buf.ptr);
                    for (py::ssize_t i = 0; i < buf.size; ++i) {
                        n_elems_per_rod.push_back(static_cast<std::size_t>(ptr[i]));
                    }
                } else if (buf.itemsize == sizeof(std::size_t)) {
                    auto* ptr = static_cast<std::size_t*>(buf.ptr);
                    n_elems_per_rod.assign(ptr, ptr + buf.size);
                } else {
                    // Fallback: iterate and cast each element
                    for (py::ssize_t i = 0; i < buf.size; ++i) {
                        py::object item = arr.attr("__getitem__")(i);
                        n_elems_per_rod.push_back(item.cast<std::size_t>());
                    }
                }
            } else {
                throw std::runtime_error("n_elems_per_rod must be a list, tuple, or numpy array");
            }

            return new BlockRodSystem(n_elems_per_rod);
        }),
        R"pbdoc(
            Create a BlockRodSystem from list of element counts per rod.

            Args:
                n_elems_per_rod: List, tuple, or numpy array of integers representing
                                 number of elements in each rod
        )pbdoc",
        py::arg("n_elems_per_rod"))
        .def_property_readonly("n_systems", [](const BlockRodSystem& block) {
            return block.n_systems();
        },
        R"pbdoc(
            Number of systems (rods) in the block.
        )pbdoc")
        .def_property_readonly("shape", [](const BlockRodSystem& block) {
            auto shape = block.shape();
            return py::make_tuple(shape.first, shape.second);
        },
        R"pbdoc(
            Get the shape of the block as (depth, width).

            Returns:
                tuple: (depth, width) tuple representing the block dimensions
        )pbdoc")
        .def("as_ref", [](const BlockRodSystem& block) {
            return block_to_numpy(block.data(), py::cast(block));
        },
        R"pbdoc(
            Get a numpy array view of the entire block data.

            Returns:
                numpy.ndarray: A writable numpy array view into the block's data.
                The array does not own the data.
        )pbdoc",
        py::keep_alive<0, 1>())
        .def("system_start_index", [](const BlockRodSystem& block, std::size_t index) {
            return block.system_start_index(index);
        },
        R"pbdoc(
            Get the starting column index for a specific rod.

            Args:
                index: Index of the rod

            Returns:
                int: Starting column index for the rod in the block
        )pbdoc",
        py::arg("index"))
        .def("at", [](BlockRodSystem& block, std::size_t index) -> BlockRodSystemViewType {
            return std::move(block.at(index));
        },
        R"pbdoc(
            Get a view for a specific rod.

            Args:
                index: Index of the rod

            Returns:
                BlockRodSystemView: View object for accessing variables of this rod
        )pbdoc",
        py::arg("index"), py::return_value_policy::reference_internal)
        .def("get", [](BlockRodSystem& block, const std::string& var_name) {
            auto block_expr = get_block_variable_by_name(block, var_name);
            return block_to_numpy(block_expr, py::cast(block));
        },
        R"pbdoc(
            Get a variable by name as a numpy array view across all rods.

            Args:
                var_name: Name of the variable (e.g., "position", "velocity", "director")

            Returns:
                numpy.ndarray: A writable numpy array view into the variable's data
                across all rods. The array does not own the data.
        )pbdoc",
        py::arg("var_name"), py::keep_alive<0, 1>())
        .def("compute_internal_forces_and_torques", [](BlockRodSystem& block, double time) {
            ELASTICAPP_GIL_RELEASE_SCOPE();
            block.compute_internal_forces_and_torques(time);
        },
        R"pbdoc(
            Compute internal forces and torques for all rods in the block.

            This operation computes the internal forces and torques based on the
            current state of the rods (positions, velocities, etc.).

            Args:
                time: Current simulation time.
        )pbdoc",
        py::arg("time"))
        .def("compute_strains", [](BlockRodSystem& block, double time) {
            ELASTICAPP_GIL_RELEASE_SCOPE();
            block.compute_strains(time);
        },
        R"pbdoc(
            Compute strains for all rods in the block.

            This operation computes the shear/stretch strains and bending/twist strains
            based on the current state of the rods.

            Args:
                time: Current simulation time (included for API compatibility, not used in implementation).
        )pbdoc",
        py::arg("time"))
        .def("update_accelerations", [](BlockRodSystem& block, double time) {
            ELASTICAPP_GIL_RELEASE_SCOPE();
            block.update_accelerations(time);
        },
        R"pbdoc(
            Update accelerations based on forces and torques.

            This operation updates the acceleration variables based on the
            computed forces and torques.

            Args:
                time: Current simulation time (included for API compatibility, not used in implementation).
        )pbdoc",
        py::arg("time"))
        .def("zeroed_out_external_forces_and_torques", [](BlockRodSystem& block, double time) {
            ELASTICAPP_GIL_RELEASE_SCOPE();
            block.zeroed_out_external_forces_and_torques(time);
        },
        R"pbdoc(
            Zero out external forces and torques for all rods.

            This operation sets all external forces and torques to zero,
            typically called at the beginning of each time step.

            Args:
                time: Current simulation time (included for API compatibility, not used in implementation).
        )pbdoc",
        py::arg("time"))
        .def("update_kinematics", [](BlockRodSystem& block, double time, double prefac) {
            ELASTICAPP_GIL_RELEASE_SCOPE();
            block.update_kinematics(prefac);
        },
        R"pbdoc(
            Update kinematics (position, director) for all rods.

            This operation updates the kinematic variables based on velocity and omega.
            Updates: position += prefac * velocity, director = R(prefac * omega) @ director

            Args:
                time: Current time (for compatibility with Python interface, not used in C++)
                prefac: Integration prefactor (e.g., time step dt)
        )pbdoc",
        py::arg("time"), py::arg("prefac"))
        .def("update_dynamics", [](BlockRodSystem& block, double time, double prefac) {
            ELASTICAPP_GIL_RELEASE_SCOPE();
            block.update_dynamics(prefac);
        },
        R"pbdoc(
            Update dynamics (velocity, omega) for all rods.

            This operation updates the dynamic variables based on acceleration and alpha.
            Updates: velocity += prefac * acceleration, omega += prefac * alpha

            Args:
                time: Current time (for compatibility with Python interface, not used in C++)
                prefac: Integration prefactor (e.g., time step dt)
        )pbdoc",
        py::arg("time"), py::arg("prefac"))
        .def_property_readonly("ghost_nodes_idx", [](const BlockRodSystem& block) {
            auto indices = block.ghost_nodes_idx();
            // Convert to numpy array (pybind11 will handle the conversion automatically)
            return py::cast(indices);
        },
        R"pbdoc(
            Get indices of ghost nodes between rods.

            Returns:
                numpy.ndarray: An array of ghost node indices (length: n_rods - 1).
                The array does not own the data.
        )pbdoc",
        py::keep_alive<0, 1>())
        .def_property_readonly("ghost_elems_idx", [](const BlockRodSystem& block) {
            auto indices = block.ghost_elems_idx();
            // Convert to numpy array (pybind11 will handle the conversion automatically)
            return py::cast(indices);
        },
        R"pbdoc(
            Get indices of ghost elements between rods.

            Returns:
                numpy.ndarray: An array of ghost element indices (length: 2 * (n_rods - 1)).
                The array does not own the data.
        )pbdoc",
        py::keep_alive<0, 1>())
        .def_property_readonly("ghost_voronoi_idx", [](const BlockRodSystem& block) {
            auto indices = block.ghost_voronoi_idx();
            // Convert to numpy array (pybind11 will handle the conversion automatically)
            return py::cast(indices);
        },
        R"pbdoc(
            Get indices of ghost voronoi nodes between rods.

            Returns:
                numpy.ndarray: An array of ghost voronoi indices (length: 3 * (n_rods - 1)).
                The array does not own the data.
        )pbdoc",
        py::keep_alive<0, 1>())
        .def("reset_ghost_for_variable", [](BlockRodSystem& block, const std::string& var_name) {
            // Helper to reset ghost for a variable by name
            reset_ghost_for_variable_by_name(block, var_name);
        },
        R"pbdoc(
            Reset ghost values for a specific variable by name.

            Args:
                var_name: Name of the variable (e.g., "position", "velocity", "director")
        )pbdoc",
        py::arg("var_name"))
        .def("reset_ghost", [](BlockRodSystem& block) {
            block.reset_ghost();
        },
        R"pbdoc(
            Reset ghost values for all variables.

            This operation sets all ghost node/element/voronoi values to their
            default ghost_value as defined in each variable type.
        )pbdoc");

    // BlockRodSystemView class
    py::class_<BlockRodSystemViewType>(m, "BlockRodSystemView")
        .def_property_readonly("shape", [](const BlockRodSystemViewType& view) {
            auto shape = view.shape();
            return py::make_tuple(shape.first, shape.second);
        },
        R"pbdoc(
            Get the shape of the view as (depth, width).

            Returns:
                tuple: (depth, width) tuple representing the view dimensions
        )pbdoc")
        .def("as_ref", [](const BlockRodSystemViewType& view) {
            return block_to_numpy(view.data(), py::cast(view));
        },
        R"pbdoc(
            Get a numpy array view of the entire view data.

            Returns:
                numpy.ndarray: A writable numpy array view into the view's data.
                The array does not own the data.
        )pbdoc",
        py::keep_alive<0, 1>())
        .def("get", [](BlockRodSystemViewType& view, const std::string& var_name) {
            auto block_expr = get_variable_by_name(view, var_name);
            return block_to_numpy(block_expr, py::cast(view));
        },
        R"pbdoc(
            Get a variable by name as a numpy array view.

            Args:
                var_name: Name of the variable (e.g., "position", "velocity", "director")

            Returns:
                numpy.ndarray: A writable numpy array view into the variable's data.
                The array does not own the data.
        )pbdoc",
        py::arg("var_name"), py::keep_alive<0, 1>());

}
} // namespace elasticapp

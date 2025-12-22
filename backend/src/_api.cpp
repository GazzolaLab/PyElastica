#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "block.h"
#include "block_view.h"
#include "traits.h"
#include "api.h"
#include "operations.h"
#include "cosserat_rod_system.h"
#include <string>
#include <stdexcept>

namespace py = pybind11;

namespace elasticapp {

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

// Helper to convert Eigen Block view to numpy array
template<typename BlockExpr>
py::array_t<double> block_to_numpy(BlockExpr&& block_expr, py::object parent) {
    // Evaluate the expression to get dimensions
    auto rows = static_cast<py::ssize_t>(block_expr.rows());
    auto cols = static_cast<py::ssize_t>(block_expr.cols());

    // Compute strides based on storage order
    auto strides = compute_strides(
        static_cast<std::size_t>(rows),
        static_cast<std::size_t>(cols)
    );

    // Create numpy array view (non-owning)
    return py::array_t<double>(
        {rows, cols},
        {static_cast<py::ssize_t>(strides.first),
         static_cast<py::ssize_t>(strides.second)},
        block_expr.data(),
        parent  // Keep parent object alive
    );
}

PYBIND11_MODULE(_memory_block, m) {
    m.doc() = R"pbdoc(
        Elasticapp BlockCosseratRod module
        ----------------------------------------

        Provides BlockCosseratRod class for 2D array management with Eigen backend.
    )pbdoc";

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
                auto* ptr = static_cast<std::size_t*>(buf.ptr);
                n_elems_per_rod.assign(ptr, ptr + buf.size);
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
        .def("as_ref", [](BlockRodSystem& block) {
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
        .def("at", [](BlockRodSystem& block, std::size_t index) {
            return block.at(index);
        },
        R"pbdoc(
            Get a view for a specific rod.

            Args:
                index: Index of the rod

            Returns:
                BlockView: View object for accessing variables of this rod
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
        .def("compute_internal_forces_and_torques", [](BlockRodSystem& block) {
            block.compute_internal_forces_and_torques();
        },
        R"pbdoc(
            Compute internal forces and torques for all rods in the block.

            This operation computes the internal forces and torques based on the
            current state of the rods (positions, velocities, etc.).
        )pbdoc")
        .def("update_accelerations", [](BlockRodSystem& block) {
            block.update_accelerations();
        },
        R"pbdoc(
            Update accelerations based on forces and torques.

            This operation updates the acceleration variables based on the
            computed forces and torques.
        )pbdoc")
        .def("zeroed_out_external_forces_and_torques", [](BlockRodSystem& block) {
            block.zeroed_out_external_forces_and_torques();
        },
        R"pbdoc(
            Zero out external forces and torques for all rods.

            This operation sets all external forces and torques to zero,
            typically called at the beginning of each time step.
        )pbdoc")
        .def("update_kinematics", [](BlockRodSystem& block) {
            block.update_kinematics();
        },
        R"pbdoc(
            Update kinematics (position, velocity, etc.) for all rods.

            This operation updates the kinematic variables based on the
            current state and time integration.
        )pbdoc")
        .def("update_dynamics", [](BlockRodSystem& block) {
            block.update_dynamics();
        },
        R"pbdoc(
            Update dynamics (forces, torques, etc.) for all rods.

            This operation updates the dynamic variables including forces
            and torques based on the current state.
        )pbdoc");

    // BlockView class
    using BlockViewType = BlockRodSystem::View;
    py::class_<BlockViewType>(m, "BlockView")
        .def_property_readonly("shape", [](const BlockViewType& view) {
            auto shape = view.shape();
            return py::make_tuple(shape.first, shape.second);
        },
        R"pbdoc(
            Get the shape of the view as (depth, width).

            Returns:
                tuple: (depth, width) tuple representing the view dimensions
        )pbdoc")
        .def("get", [](BlockViewType& view, const std::string& var_name) {
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

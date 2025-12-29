#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../api.h"
#include "api.h"
#include "collision/collision_system.h"

namespace py = pybind11;

PYBIND11_MODULE(_collision, m) {
    m.doc() = R"pbdoc(
        Elasticapp Collision module
        ---------------------------

        Provides collision detection and resolution for Cosserat rods using
        the Discrete Element Method (DEM).
    )pbdoc";

    using namespace elasticapp;
    using namespace elasticapp::collision;
    using namespace elasticapp::collision::physics;

    // NoInteraction model class (for testing)
    py::class_<physics::NoInteraction>(m, "NoInteraction")
        .def(py::init<>(),
            R"pbdoc(
                Initialize a NoInteraction collision physics model.

                This model returns zero force for all contacts, useful for testing
                collision detection without applying forces.

                Args:
                    None (no parameters required)
            )pbdoc");

    // LinearSpringDashpot model class
    py::class_<physics::LinearSpringDashpot>(m, "LinearSpringDashpot")
        .def(py::init<double, double, double>(),
            R"pbdoc(
                Initialize a LinearSpringDashpot collision physics model.

                Args:
                    k_normal: Normal spring constant for repulsion force
                    eta_normal: Normal damping coefficient (also used for tangential damping)
                    friction: Static friction coefficient
            )pbdoc",
            py::arg("k_normal") = 1.0,
            py::arg("eta_normal") = 0.1,
            py::arg("friction") = 0.5)
        .def(py::init<double, double, double, double>(),
            R"pbdoc(
                Initialize a LinearSpringDashpot collision physics model with explicit tangential damping.

                Args:
                    k_normal: Normal spring constant for repulsion force
                    eta_normal: Normal damping coefficient
                    eta_tangential: Tangential damping coefficient
                    friction: Static friction coefficient
            )pbdoc",
            py::arg("k_normal"),
            py::arg("eta_normal"),
            py::arg("eta_tangential"),
            py::arg("friction"))
        .def(py::init<double, double, double, double, double>(),
            R"pbdoc(
                Initialize a LinearSpringDashpot collision physics model with explicit tangential spring and damping.

                Args:
                    k_normal: Normal spring constant for repulsion force
                    eta_normal: Normal damping coefficient
                    k_tangential: Tangential spring constant
                    eta_tangential: Tangential damping coefficient
                    friction: Static friction coefficient
            )pbdoc",
            py::arg("k_normal"),
            py::arg("eta_normal"),
            py::arg("k_tangential"),
            py::arg("eta_tangential"),
            py::arg("friction"));

    // CollisionSystem class (using DefaultCollisionSystem)
    // Support both LinearSpringDashpot and NoInteraction models
    py::class_<DefaultCollisionSystem>(m, "CollisionSystem")
        .def(py::init<const physics::LinearSpringDashpot&, std::size_t>(),
            R"pbdoc(
                Initialize a CollisionSystem with a LinearSpringDashpot physics model.

                Uses default policies: HashGrid (coarse), MaxContacts (fine), UnionFind (batching).

                Args:
                    model: The LinearSpringDashpot collision physics model
                    detect_every: Perform coarse detection every N steps (default: 1, meaning every step)
            )pbdoc",
            py::arg("model"),
            py::arg("detect_every") = 1)
        .def(py::init<const physics::NoInteraction&, std::size_t>(),
            R"pbdoc(
                Initialize a CollisionSystem with a NoInteraction physics model.

                Uses default policies: HashGrid (coarse), MaxContacts (fine), UnionFind (batching).
                Useful for testing collision detection without applying forces.

                Args:
                    model: The NoInteraction collision physics model
                    detect_every: Perform coarse detection every N steps (default: 1, meaning every step)
            )pbdoc",
            py::arg("model"),
            py::arg("detect_every") = 1)
        .def("detect_every", &DefaultCollisionSystem::detect_every,
            R"pbdoc(
                Get the detect_every parameter.

                Returns:
                    int: Number of steps between coarse detection calls
            )pbdoc")
        .def("set_detect_every", &DefaultCollisionSystem::set_detect_every,
            R"pbdoc(
                Set the detect_every parameter.

                Args:
                    detect_every: Number of steps between coarse detection calls
            )pbdoc",
            py::arg("detect_every"))
        .def("resolve", &DefaultCollisionSystem::resolve,
            R"pbdoc(
                Resolve collisions for a BlockRodSystem.

                This method performs the full collision detection and resolution pipeline:
                1. Data extraction from Block
                2. Coarse collision detection (HashGrid)
                3. Fine collision detection (MaxContacts)
                4. Contact batching (UnionFind)
                5. Contact resolution (LinearSpringDashpot)
                6. Force application to ExternalForces

                Args:
                    system: The BlockRodSystem to resolve collisions for
            )pbdoc",
            py::arg("system"));
}

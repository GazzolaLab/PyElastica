#include <pybind11/pybind11.h>
#include "version.h"

namespace py = pybind11;

PYBIND11_MODULE(version, m) {
    m.doc() = R"pbdoc(
        Elasticapp version module
        -------------------------

        Provides version information for the elasticapp package.
    )pbdoc";

    // Version function
    m.def("version", &elasticapp::version, R"pbdoc(
        Return the current version of elasticapp.

        Returns:
            str: The version string
    )pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = elasticapp::version();
#else
    m.attr("__version__") = "dev";
#endif
}

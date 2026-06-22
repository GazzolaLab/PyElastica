#ifndef ELASTICAPP_VERSION_H
#define ELASTICAPP_VERSION_H

#include <string>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace elasticapp {

inline std::string version() {
    // VERSION_INFO is defined by CMake and stringified here
    return std::string(MACRO_STRINGIFY(VERSION_INFO));
}

} // namespace elasticapp

#endif // ELASTICAPP_VERSION_H

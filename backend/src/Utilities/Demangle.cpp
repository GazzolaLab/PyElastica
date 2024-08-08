//******************************************************************************
// Includes
//******************************************************************************

#include "Utilities/Demangle.hpp"

/// \cond
// #define DEMANGLE_USE_BOOST
/// \endcond

#if defined(DEMANGLE_USE_BOOST)
#include <boost/core/demangle.hpp>
#elif defined(__clang__) || defined(__GNUG__)
#include <cxxabi.h>
#endif

#if defined(DEMANGLE_USE_BOOST)
std::string demangle(char const* name) { return boost::core::demangle(name); }
#elif defined(__clang__) || defined(__GNUG__)
// else we inline some logic of boost::core::demangle
std::string demangle(char const* name) {
  int status(-1);
  char* demangled_name = abi::__cxa_demangle(name, nullptr, nullptr, &status);
  // status == 0 is good
  std::string class_name(status ? name : demangled_name);
  // Be a good boy and free
  std::free(demangled_name);  // NOLINT
  return class_name;
}
#else
std::string demangle(char const* name) { return name; }
#endif

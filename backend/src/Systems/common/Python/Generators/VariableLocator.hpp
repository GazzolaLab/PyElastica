#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <pybind11/pybind11.h>
//

namespace py_bindings {

  template <typename T>
  struct VariableLocator {
    constexpr auto data() & noexcept -> T& { return item_; }
    constexpr auto data() const& noexcept -> T const& { return item_; }
    T& item_;
  };

}  // namespace py_bindings

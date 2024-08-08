#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/Block/TypeTraits.hpp"
#include "Utilities/TMPL.hpp"
//
#include <type_traits>

namespace py_bindings {

  namespace tt {

    //**************************************************************************
    /*!\brief Filters out variables based on UnwantedVariableTags
     * \ingroup python_bindings
     *
     * \tparam UnwantedVariableTags Tags that should not be bound/exposed to the
     * python side
     */
    template <typename UnwantedVariableTags>
    struct VariableFilter {
      template <typename T>
      using type = tmpl::any<UnwantedVariableTags,
                             std::is_same<tmpl::_1, ::blocks::parameter_t<T>>>;
    };
    //**************************************************************************

  }  // namespace tt

}  // namespace py_bindings

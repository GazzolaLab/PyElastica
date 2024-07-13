// A modified version of
// https://github.com/joboccara/NamedType/blob/master/named_type_impl.hpp
// customized for elastica needs
// See https://raw.githubusercontent.com/joboccara/NamedType/master/LICENSE

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Types.hpp"

namespace named_type {

  // Clang-Tidy: Declaration uses identifier '_DefaultNamedTypeParameterTag_',
  // which is a reserved identifier
  struct _DefaultNamedTypeParameterTag_ {}; /* NOLINT */

}  // namespace named_type

#pragma once

namespace named_type {

  //////////////////////////////////////////////////////////////////////////////
  //
  // Forward declarations of named types
  //
  //////////////////////////////////////////////////////////////////////////////
  struct _DefaultNamedTypeParameterTag_;

  template <typename T, typename Tag, template <typename> class... Affordances>
  class NamedType;

}  // namespace named_type

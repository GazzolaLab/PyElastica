#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <string>

namespace convert_case {

  struct PascalCase {};
  struct SnakeCase {};

  template <typename Case_>
  struct From {
    using Case = Case_;
  };

  template <typename Case_>
  struct To {
    using Case = Case_;
  };

  using FromPascalCase = From<PascalCase>;
  using ToSnakeCase = To<SnakeCase>;

  std::string convert(std::string const& input_str, From<PascalCase> /*meta*/,
                      To<SnakeCase> /*meta*/);

}  // namespace convert_case

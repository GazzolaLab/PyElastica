#include "ConvertCase.hpp"

#include <algorithm>
#include <cctype>

namespace convert_case {

  // Convert lowerPascalCase and UpperPascalCase strings to
  // lower_with_underscore.
  std::string convert(std::string const& input_str, From<PascalCase> /*meta*/,
                      To<SnakeCase> /*meta*/) {
    std::size_t const siz(input_str.size());
    if (not siz)
      return "";

    std::string str(1, std::tolower(static_cast<unsigned char>(input_str[0])));
    str.reserve(siz);

    // First place underscores between contiguous lower and upper case letters.
    // For example, `_LowerPascalCase` becomes `_Lower_Pascal_Case`.
    for (auto it = std::cbegin(input_str) + 1; it != std::cend(input_str);
         ++it) {
      if (std::isupper(*it) && *(it - 1) != '_')
        str += "_";
      str += *it;
    }

    // Then convert it to lower case.
    std::transform(std::cbegin(str), std::cend(str), std::begin(str),
                   [](unsigned char c) { return std::tolower(c); });

    return str;
  }

}  // namespace convert_case

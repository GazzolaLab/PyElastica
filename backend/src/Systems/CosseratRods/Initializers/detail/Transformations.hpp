#pragma once

#include <stdexcept>
#include <string>
#include <tuple>
///
#include "Systems/CosseratRods/BlockInitializer.hpp"
//
#include "Utilities/MakeString.hpp"  // for more contextual error messages
// #include "Utilities/PrettyType.hpp"  // for more contextual error messages

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      template <typename Tag, typename Tuple>
      inline auto pass_as_wrapped(Tuple const& cache) noexcept {
        return blocks::initialize<typename Tag::TagType>(
            // do a copy here of tuple of initializers
            [value = (std::get<Tag>(cache).value())](
                std::size_t /* dummy index parameter*/) ->
            typename Tag::type { return value; });
      }

      template <typename Tuple, typename... ApplyTags>
      inline auto pass_as_wrapped(Tuple const& cache,
                                  tmpl::list<ApplyTags...> /* meta*/) noexcept {
        // forward_as_tuple makes references to temporary, so make tuple
        // instead. The move constructors should then kick in.
        return std::make_tuple(pass_as_wrapped<ApplyTags>(cache)...);
      }

      template <typename Tuple, typename... ApplyTags>
      inline auto check_lower_bounds(Tuple const& cache,
                                     tmpl::list<ApplyTags...> /* meta*/) {
        using L = tmpl::list<ApplyTags...>;
        tmpl::for_each<L>([&](auto v) {
          using ApplyTag = tmpl::type_from<decltype(v)>;
          if (std::get<ApplyTag>(cache).value() <= ApplyTag ::lower_bound()) {
            throw std::domain_error(std::string(
                MakeString{}
                // << pretty_type::name<ApplyTag>() << " requested is too small!"
                << ApplyTag::name() << " requested is too small!"
                << "\n"
                // need to stringify apply help here, else
                // constexpr symbols are not found when launching
                // case-studies from python
                << "Help : " << std::string(ApplyTag::help) << "\n"
                << "Provide a value greater than : " << ApplyTag ::lower_bound()
                << "\n"));
          }
        });
      }

    }  // namespace detail

  }  // namespace cosserat_rod

}  // namespace elastica

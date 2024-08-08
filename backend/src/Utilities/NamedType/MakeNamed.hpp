// A modified version of
// https://github.com/joboccara/NamedType/blob/master/named_type_impl.hpp
// customized for elastica needs
// See https://raw.githubusercontent.com/joboccara/NamedType/master/LICENSE

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <type_traits>
#include <utility>

#include "Traits.hpp"
#include "Types.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

namespace named_type {

  //****************************************************************************
  /*!\brief Makes a named parameter with the given value
  //
  // \example
  // Say we have a strong type called `Arithmetic` below
  // \snippet Test_MakeNamed.cpp arithmetic_example
  //
  // then we can use make_named() to create strong types as follows
  // \snippet Test_MakeNamed.cpp make_arithmetic_example
  //
  // \tparam StrongType The named parameter type
  // \param value Value to initialize StrongType with
  // \return Instance of StrongType
  */
  template <template <typename> class StrongType, typename T>
  constexpr auto make_named(T&& value) noexcept(
      noexcept(StrongType<std::decay_t<T>>(std::forward<T>(value))))
      -> StrongType<std::decay_t<T>> {
    using type = std::decay_t<T>;
    return StrongType<type>(std::forward<T>(value));
  }
  //****************************************************************************

  namespace detail {
    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Auxiliary helper class for make_default
    // \ingroup named_type
    */
    template <typename Tag, template <typename> class... Affordances>
    struct NameHelper {
      template <typename T>
      using type = NamedType<T, Tag, Affordances...>;
    };
    /*! \endcond */
    //**************************************************************************

  }  // namespace detail

  //****************************************************************************
  /*!\brief Makes a named parameter with a default tag
  //
  // \example
  // \snippet TODO
  //
  // \tparam StrongType The named parameter type
  // \param value Value to initialize StrongType with
  // \return Instance of NamedType with a  default tag
   */
  template <typename Tag, template <typename> class... Affordances,
            template <typename>
            class R = detail::NameHelper<Tag, Affordances...>::template type,
            typename T
            // DONE : This Requires fails for the use cases in wrapped_at
            // because we have no way of checking the existence of an operator()
            // without the function signature when its templated. Hence we
            // comment it out for now. Fixed with trait is_callable.
            ,
            Requires<cpp17::conjunction_v<typename Traits<
                Affordances>::template type<std::decay_t<T>>...>> = nullptr>
  constexpr decltype(auto) make_named(T&& value) noexcept(
      noexcept(make_named<R>(std::forward<T>(value)))) {
    return make_named<R>(std::forward<T>(value));
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Makes a named parameter with a default tag
  //
  // \example
  // \snippet TODO
  //
  // \tparam StrongType The named parameter type
  // \param value Value to initialize StrongType with
  // \return Instance of NamedType with a  default tag
   */
  template <template <typename> class... Affordances, typename T>
  constexpr decltype(auto) make_named_with_default(T&& value) noexcept(
      noexcept(make_named<_DefaultNamedTypeParameterTag_, Affordances...>(
          std::forward<T>(value)))) {
    return make_named<_DefaultNamedTypeParameterTag_, Affordances...>(
        std::forward<T>(value));
  }
  //****************************************************************************

}  // namespace named_type

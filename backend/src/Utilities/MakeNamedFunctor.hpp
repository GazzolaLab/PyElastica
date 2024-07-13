#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Utilities/NamedType.hpp"
#include "Utilities/Overloader.hpp"

namespace named_type {

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Makes a named parameter with a given functor for module generics
  //
  // \details
  // DRY for making named interfaces for generic objects across all \elastica
  // modules such as Connections, Constraints
  //
  // \tparam StrongType The named parameter type
  //
  // \param func Functor to initialize StrongType with
  // \return Instance of StrongType wrapped with the functor
  //
  // \see make_value_constraint(), make_rate_constraint()
  */
  template <template <typename> class StrongType, typename F>
  constexpr inline decltype(auto) make_named_functor(F&& func) noexcept(
      noexcept(::named_type::make_named<StrongType>(std::forward<F>(func)))) {
    return ::named_type::make_named<StrongType>(std::forward<F>(func));
  }
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Makes a named parameter with given functors for module generics
  //
  // \details
  // DRY for making named interfaces for generic objects across all \elastica
  // modules such as Connections, Constraints. Here we consider the overloaded
  // set of all possible functors passed in, and return a Strongly typed
  // interface.
  //
  // \tparam StrongType The named parameter type
  //
  // \param f First overload functor to initialize StrongType with
  // \param g First overload functor to initialize StrongType with
  // \param funcs... Other overload functors to initialize StrongType with
  // \return Instance of StrongType wrapped with the functor
  //
  // \see make_value_constraint(), make_rate_constraint()
  */
  template <template <typename> class StrongType, typename F, typename G,
            typename... Funcs>
  constexpr inline decltype(auto)
  make_named_functor(F&& f, G&& g, Funcs&&... funcs) noexcept(
      noexcept(::named_type::make_named<StrongType>(
          make_overloader(std::forward<F>(f), std::forward<G>(g),
                          std::forward<Funcs>(funcs)...)))) {
    return ::named_type::make_named<StrongType>(make_overloader(
        std::forward<F>(f), std::forward<G>(g), std::forward<Funcs>(funcs)...));
  }
  /*! \endcond */
  //****************************************************************************

}  // namespace named_type

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <type_traits>

#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits/ArraySize.hpp"
#include "Utilities/TypeTraits/Cpp17.hpp"
#include "Utilities/TypeTraits/Cpp20.hpp"
#include "Utilities/TypeTraits/HasEquivalence.hpp"
#include "Utilities/TypeTraits/InvokeResult.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/TypeTraits/IsCallable.hpp"
#include "Utilities/TypeTraits/IsConvertibleToA.hpp"
#include "Utilities/TypeTraits/IsDefined.hpp"
#include "Utilities/TypeTraits/IsDetected.hpp"
#include "Utilities/TypeTraits/IsInvokable.hpp"
#include "Utilities/TypeTraits/IsIterable.hpp"
#include "Utilities/TypeTraits/IsMaplike.hpp"
#include "Utilities/TypeTraits/IsStdArray.hpp"
#include "Utilities/TypeTraits/IsStdArrayOfSize.hpp"
#include "Utilities/TypeTraits/IsStreamable.hpp"
#include "Utilities/TypeTraits/Void.hpp"

//==============================================================================
//
//  DOXYGEN DOCUMENTATION
//
//==============================================================================

//******************************************************************************
/*!\defgroup type_traits_group Type Traits
 * \ingroup utils
 * \brief A collection of type traits for metaprogramming
 *
 * \details
 * Also includes C++17 and C++20 additions to the standard library.
 */
//******************************************************************************

//******************************************************************************
/*!\brief Contains all typetraits functionality
// \ingroup type_traits_group
*/
namespace tt {}
//******************************************************************************

namespace tt {

  /// \ingroup type_traits_group
  /// \brief Given a set of templated types, returns `void`
  ///
  /// \details
  /// Given a list of templated types, returns `void`. This is useful for
  /// controlling name lookup resolution via partial template specialization.
  ///
  /// \usage
  /// For any set of templated types `template <...> Ti`,
  /// \code
  /// using result = cpp17::void_t<T0, T1, T2, T3>;
  /// \endcode
  ///
  /// \metareturns
  /// void
  ///
  /// \semantics
  /// For any set of types `Ti`,
  /// \code
  /// using result = void;
  /// \endcode
  ///
  /// \example
  /// \snippet Utilities/Test_TypeTraits.cpp void_t_example
  /// \see std::void_t
  /// \tparam Ts the set of types
  template <template <class...> class...>
  using templated_void_t = void;

  // Count number of types in the tagged tuple
  namespace detail {
    // Check whether a type is contained in the parameter pack
    template <typename ToBeChecked, typename... Args>
    struct IsContained;

    // Not contained
    template <typename ToBeChecked>
    struct IsContained<ToBeChecked> : std::false_type {};

    //
    template <typename ToBeChecked, typename Head, typename... Args>
    struct IsContained<ToBeChecked, Head, Args...> {
      constexpr static bool value = std::is_same<ToBeChecked, Head>::value or
                                    IsContained<ToBeChecked, Args...>::value;
    };
  }  // namespace detail

  template <typename T, typename... Args>
  constexpr bool is_contained_v = detail::IsContained<T, Args...>::value;

  namespace detail {
    template <typename... Args>
    struct IsAllUnique;

    template <typename ToBeChecked, typename... Args>
    struct IsAllUnique<ToBeChecked, Args...> {
      constexpr static bool value = not is_contained_v<ToBeChecked, Args...> and
                                    IsAllUnique<Args...>::value;
    };

    // No repeats when its the final argument
    template <typename FinalArg>
    struct IsAllUnique<FinalArg> : std::true_type {};

  }  // namespace detail

  template <typename... Args>
  constexpr bool is_all_unique_v = detail::IsAllUnique<Args...>::value;

  namespace detail {
    /*
     * Taken from
     * https://stackoverflow.com/a/34672753
     * But is similar in spirit to Walter Brown's talks.
     * Needs to publicly inherit!!!
     */
    template <template <typename...> class Base, typename Derived>
    struct IsBaseOfTemplateImpl {
      template <typename... Ts>
      static constexpr std::true_type check(const Base<Ts...>*);
      static constexpr std::false_type check(...);
      static constexpr bool value =
          decltype(check(std::declval<Derived*>()))::value;
    };
  }  // namespace detail

  template <template <typename...> class Base, typename Derived>
  constexpr bool is_templated_base_of_v =
      detail::IsBaseOfTemplateImpl<Base, Derived>::value;

  template <template <typename...> class Base, typename Derived>
  using is_templated_base_of_t =
      cpp17::bool_constant<is_templated_base_of_v<Base, Derived>>;

  // @{
  /// \ingroup type_traits
  /// \brief Check if type `T` has operator<<(`S`, `T`) defined.
  ///
  /// \details
  /// Inherits from std::true_type if the type `T` has operator<<(`S`, `T`)
  /// defined for a stream `S`, otherwise inherits from std::false_type
  ///
  /// \usage
  /// For any type `T` and stream type `S`,
  /// \code
  /// using result = tt::is_streamable<S, T>;
  /// \endcode
  ///
  /// \metareturns
  /// cpp17::bool_constant
  ///
  /// \semantics
  /// If the type `T` has operator<<(`S`, `T`) defined for stream `S`, then
  /// \code
  /// typename result::type = std::true_type;
  /// \endcode
  /// otherwise
  /// \code
  /// typename result::type = std::false_type;
  /// \endcode
  ///
  /// \example
  /// \snippet Utilities/Test_TypeTraits.cpp is_streamable_example
  /// \see std::cout std::ifstream std::sstream std::ostream
  /// \tparam S the stream type, e.g. std::stringstream or std::ostream
  /// \tparam T the type we want to know if it has operator<<
  namespace detail {
    template <typename T>
    struct is_cvref : cpp17::conjunction<std::is_const<T>, std::is_reference<T>,
                                         std::is_volatile<T>> {};
  }  // namespace detail

  /// \see is_streamable
  template <typename T>
  constexpr bool is_cvref_v = detail::is_cvref<T>::value;

  /// \see is_streamable
  template <typename T>
  using is_cvref_t = typename detail::is_cvref<T>::type;
  // @}
}  // namespace tt

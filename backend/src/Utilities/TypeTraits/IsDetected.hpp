#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

#include "Utilities/NoSuchType.hpp"
#include "Utilities/TypeTraits/Void.hpp"

namespace tt {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Default result type for the detected_t type trait.
  // \ingroup type_traits_group
  */
  using ::NoSuchType;
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Auxiliary helper struct for the is_detected type trait.
  // \ingroup type_traits_group
  */
  template <typename Default, typename AlwaysVoid,
            template <typename...> class OP, typename... Ts>
  struct Detector {
    using value_type = std::false_type;
    using type = Default;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of the Detector class template for a successful
  // evaluation of \a OP.
  // \ingroup type_traits_group
  */
  template <typename Default, template <typename...> class OP, typename... Ts>
  struct Detector<Default, cpp17::void_t<OP<Ts...> >, OP, Ts...> {
    using value_type = std::true_type;
    using type = OP<Ts...>;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*!\brief Compile time detection of type properties.
  // \ingroup type_traits_group
  //
  // \details
  // The detected_t type trait determines if the given `Op` can be evaluated
  // for the given types `Ts...`. In case `Op` can be successfully evaluated,
  // detected_or evaluates to `Op<Ts...>`. Otherwise it evaluates to
  // `Default`
  //
  // \usage
  // For a type `Default`, any template type `Op`, and types `Ts...`
     \code
     using result = tt::detected_or<Default, Op, Ts...>;
     \endcode
  //
  // \metareturns
  // either `Op<Ts..>` if `Op<Ts...>` is well-formed, else returns `Default`
  //
  // \example
  // \snippet Test_IsDetected.cpp detected_or_t_example
  //
  // \tparam Default Default type to fall back to
  // \tparam Op Meta-type to check
  // \tparam Ts... Type of arguments to be passed onto Op
  //
  // \see detected_or_t
  */
  template <typename Default, template <typename...> class Op, typename... Ts>
  using detected_or = Detector<Default, cpp17::void_t<>, Op, Ts...>;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Evaluates the type upon applying `Op` for the given types `Ts...`
  // \ingroup type_traits_group
  //
  // \details
  // The detected_t type trait determines if the given `Op` can be evaluated
  // for the given types `Ts...`. In case `Op` can be successfully evaluated,
  // detected_or evaluates to `Op<Ts...>`. Otherwise it evaluates to
  // `NoSuchType`
  //
  // \usage
  // For any template type `Op`, and types `Ts...`
     \code
     using result = tt::detected_t<Op, Ts...>;
     \endcode
  //
  // \metareturns
  // either `Op<Ts..>` if `Op<Ts...>` is well-formed, else returns `NoSuchType`
  //
  // \example
  // \snippet Test_IsDetected.cpp detected_t_example
  //
  // \tparam Op Meta-type to check
  // \tparam Ts... Type of arguments to be passed onto Op
  //
  // \see detected_or
  */
  template <template <typename...> class Op, typename... Ts>
  using detected_t = typename detected_or<NoSuchType, Op, Ts...>::type;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for `detected_or`.
  // \ingroup type_traits_group
  //
  // \details
  // The detected_or_t variable template provides a convenient shortcut to
  // access the nested type `type` of `detected_or`, used as follows.
  //
  // \usage
  // Given the type `Default`, the template `Op` and the two types `T1` and
  // `T2` the following two statements are identical:
     \code
     using type1 = typename tt::detected_or<Default,Op,T1,T2>::type;
     using type2 = tt::detected_or_t<Default,Op,T1,T2>;
     \endcode
  // as demonstrated through this example
  //
  // \example
  // \snippet Test_IsDetected.cpp detected_or_t_example
  //
  // \tparam Default Default type to fall back to
  // \tparam Op Meta-type to check
  // \tparam Ts... Type of arguments to be passed onto Op
  //
  // \see detected_or, detected_t
  */
  template <typename Default, template <typename...> class Op, typename... Ts>
  using detected_or_t = typename detected_or<Default, Op, Ts...>::type;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Detect if `OP` can be evaluated for the given types `Ts...`
  // \ingroup type_traits_group
  //
  // \details
  // Determines if the given `OP` can be evaluated for the given types `Ts...`.
  // In case `OP` can be successfully evaluated, is_detected is an alias for
  // std::true_type. Otherwise it is an alias for std::false_type.
  //
  // \usage
  // For any template type `Op`, and types `Ts...`
     \code
     using result = tt::is_detected<Op, Ts...>;
     \endcode
  //
  // \metareturns
  // cpp17::bool_constant
  //
  // \semantics
  // For any template type `Op` that can potentially be evaluated with the
  // types `Ts...`
     \code
     result = std::true_type;
     \endcode
  // otherwise
     \code
     result = std::false_type;
     \endcode
  //
  // \example
  // \snippet Test_IsDetected.cpp is_detected_example
  //
  // \tparam Op Meta-type to check
  // \tparam Ts... Type of arguments to be passed onto Op
  //
  // \see detected_or
  */
  template <template <typename...> class Op, typename... Ts>
  using is_detected = typename detected_or<NoSuchType, Op, Ts...>::value_type;
  //****************************************************************************

  //****************************************************************************
  /*!\brief Auxiliary variable template for eager check of `is_detected`
  // \ingroup type_traits_group
  //
  // \details
  // The is_detected_v variable template provides a convenient shortcut to
  // access the nested value `value` of `is_detected`, used as follows.
  //
  // \usage
  // Given the template `Op` and the two types `T1` and `T2` the
  // following two statements are identical:
     \code
     constexpr bool value1 = tt::is_detected<Op,T1,T2>::value;
     constexpr bool value2 = tt::is_detected_v<Op,T1,T2>;
     \endcode
  // as demonstrated through this example
  //
  // \example
  // \snippet Test_IsDetected.cpp is_detected_v_example
  //
  // \tparam Op Meta-type to check
  // \tparam Ts... Type of arguments to be passed onto Op
  //
  // \see is_detected
  */
  template <template <typename...> class Op, typename... Ts>
  constexpr bool is_detected_v = is_detected<Op, Ts...>::value;
  //****************************************************************************

}  // namespace tt

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

namespace tt {

  namespace detail {
    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    // reduce instantiations using nested structs
    template <template <typename...> class T>
    struct is_convertible_to_a {
      template <class U, class... Args>
      static constexpr decltype(
          static_cast<T<Args...> const&>(std::declval<U>()), std::true_type{})
          test(const T<Args...>&);

      template <class U>
      static constexpr std::false_type test(...);
    };
    /*! \endcond */
    //**************************************************************************
  }  // namespace detail

  // https://stackoverflow.com/questions/22592419/checking-if-a-class-inherits-from-any-template-instantiation-of-a-template

  //****************************************************************************
  /*!\brief Check if type `U` is convertible to a template specialization of `T`
   * \ingroup type_traits_group
   *
   * \requires `U` is a class template
   * \effects If `T` is convertible to a template specialization of `U`, then
   * inherits from std::true_type, otherwise inherits from std::false_type
   *
   * \usage
   * For any type `T` and class template `U`
   * \code
   * using result = tt::is_convertible_to_a<U, T>;
   * \endcode
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * If the type `T` is a template specialization of the type `U`, or can be
   * converted to a template specialization of the type `U` (i.e. by deriving
   * from a template specialization of `U`), then
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Test_IsConvertibleToA.cpp is_convertible_to_a_example
   *
   * \tparam T type to check
   * \tparam U the type that T might be a template specialization of, or might
   * be derived from a template specialization of
   *
   * \see ::tt::is_a
   */
  template <template <typename...> class U, typename T>
  struct is_convertible_to_a
      : public decltype(detail::is_convertible_to_a<U>::template test<T>(
            std::declval<T>())) {};
  //****************************************************************************

}  // namespace tt

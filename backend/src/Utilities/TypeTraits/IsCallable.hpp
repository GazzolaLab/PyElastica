#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

namespace tt {
  //
  // Taken from https://stackoverflow.com/a/35762494
  //

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Check if type `T` has a call operator
  // \ingroup type_traits_group
  //
  // \details
  // The is_callable type trait inherits from std::true_type if `T` has the
  // call operator, operator() defined, otherwise inherits from std::false_type.
  //
  // \usage
  // For any type `T`
     \code
     using result = tt::is_callable<T>;
     \endcode
  //
  // \metareturns
  // cpp17::bool_constant
  //
  // \semantics
  // If the type `T` is callable, then
     \code
     typename result::type = std::true_type;
     \endcode
  // otherwise
     \code
     typename result::type = std::false_type;
     \endcode
  //
  // \example
  // \snippet Test_IsCallable.cpp is_callable_example
  // \tparam T type to check
  */
  template <typename T, typename U = void>
  struct is_callable {
    static bool constexpr value =
        std::conditional_t<std::is_class<std::remove_reference_t<T>>::value,
                           is_callable<std::remove_reference_t<T>, int>,
                           std::false_type>::value;
  };
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of is_callable for scenarios
  // \ingroup type_traits_group
  */
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args...), U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T (*)(Args...), U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T (&)(Args...), U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args..., ...), U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T (*)(Args..., ...), U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T (&)(Args..., ...), U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args...) const, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args...) volatile, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args...) const volatile, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args..., ...) const, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args..., ...) volatile, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args..., ...) const volatile, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args...)&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args...) const&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args...) volatile&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args...) const volatile&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args..., ...)&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args..., ...) const&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args..., ...) volatile&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args..., ...) const volatile&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args...)&&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args...) const&&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args...) volatile&&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args...) const volatile&&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args..., ...)&&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args..., ...) const&&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args..., ...) volatile&&, U> : std::true_type {};
  template <typename T, typename U, typename... Args>
  struct is_callable<T(Args..., ...) const volatile&&, U> : std::true_type {};

  template <typename T>
  struct is_callable<T, int> {
   private:
    using YesType = char (&)[1];
    using NoType = char (&)[2];

    struct Fallback {
      void operator()();
    };

    struct Derived : T, Fallback {};

    template <typename U, U>
    struct Check;

    template <typename>
    static YesType Test(...);

    template <typename C>
    static NoType Test(Check<void (Fallback::*)(), &C::operator()>*);

   public:
    static bool const constexpr value =
        sizeof(Test<Derived>(0)) == sizeof(YesType);
  };
  /*! \endcond */
  //****************************************************************************

}  // namespace tt

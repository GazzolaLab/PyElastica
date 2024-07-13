#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

namespace cpp17 {

  //****************************************************************************
  /*!\brief Marks a type as const
   * \ingroup utils
   *
   * \details
   * Forms lvalue reference to const type of `t`
   *
   * \example
   * \snippet Test_AsConst.cpp as_const_eg
   *
   * \param t Parameter to make as const
   * \return const T
   */
  template <class T>
  constexpr auto as_const(T& t) noexcept -> std::add_const_t<T>& {
    return t;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\cond ELASTICA_INTERNAL */
  /*!\brief Marks a type as const
   * \ingroup utils
   *
   * \details
   *  const rvalue reference overload is deleted to disallow rvalue arguments
   */
  template <class T>
  void as_const(const T&&) = delete;
  /*! \endcond */
  //****************************************************************************

}  // namespace cpp17

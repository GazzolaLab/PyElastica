#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

namespace tt {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Check whether a given type `Derived` is derived from
   * 'CRTPBaseTemplate' using the CRTP pattern
   * \ingroup type_traits_group
   *
   * \details
   * Inherits from std::true_type if `Derived` is derived from
   * 'CRTPBaseTemplate' otherwise inherits from std::false_type.
   *
   * \usage
   * For any base template `CRTPBaseTemplate` and type `C`,
   * \code
   * using result = IsCRTP<CRTPBaseTemplate, C>;
   * \endcode
   *
   * \metareturns
   * cpp17::bool_constant
   *
   * \semantics
   * If the type `C` is publicly derived from CRTPBaseTemplate<C>, then
   * \code
   * typename result::type = std::true_type;
   * \endcode
   * otherwise
   * \code
   * typename result::type = std::false_type;
   * \endcode
   *
   * \example
   * \snippet Test_IsCRTP.cpp is_crtp_example
   *
   * \tparam CRTPBaseTemplate The CRTP base class
   * \tparam Derived The CRTP derived class
   * \tparam Args Optional Meta-args to CRTP base, if any
   *
   * \note std::is_convertible is used in the following type aliases as it
   * will not match private base classes (unlike std::is_base_of)
   *
   * \see CRTPHelper
   */
  template <template <typename...> class CRTPBaseTemplate, typename Derived,
            typename... Args>
  struct IsCRTP
      : public std::is_convertible<Derived*,
                                   CRTPBaseTemplate<Derived, Args...>*> {};
  //****************************************************************************

}  // namespace tt

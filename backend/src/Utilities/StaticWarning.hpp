#pragma once

namespace elastica {

  //****************************************************************************
  /*!\brief Prints a contextual warning at compile-time
   * \ingroup utils
   */
  template <bool B>
  constexpr void static_warning(const char*) noexcept;
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Prints a contextual warning at compile-time
   * \ingroup utils
   *
   * \details
   * Specialization for a passing check, does not print anything
   */
  template <>
  constexpr void static_warning<true>(const char*) noexcept {}
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Prints a contextual warning at compile-time
   * \ingroup utils
   *
   * \details
   * Specialization for a failed check, prints error message
   */
  template <>
  [[deprecated]] constexpr void static_warning<false>(
      const char* const) noexcept {}
  /*! \endcond */
  //****************************************************************************
}  // namespace elastica

//******************************************************************************
/*!\brief A compile-time warning message similar to static_assert
// \ingroup utils
*/
#define STATIC_WARNING(pred, msg) ::elastica::static_warning<pred>(msg)
//******************************************************************************

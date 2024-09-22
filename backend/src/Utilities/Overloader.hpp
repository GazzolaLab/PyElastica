// Reused from SpECTRE : https://spectre-code.org/
// Distributed under the MIT License.
// See LICENSE.txt for details.
// First came across it here : https://gist.github.com/dabrahams/3779345

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <utility>

#include "Utilities/ForceInline.hpp"

#if defined(__clang__) || __GNUC__ > 5
#define OVERLOADER_CONSTEXPR constexpr
#else
#define OVERLOADER_CONSTEXPR
#endif

//******************************************************************************
/*! \cond ELASTICA_INTERNAL */
namespace overloader_detail {
  struct no_such_type;
}  // namespace overloader_detail
/*! \endcond */
//******************************************************************************

//******************************************************************************
/*!\brief Facilities for overloading lambdas, useful for lambda-SFINAE
 * \ingroup UtilitiesGroup
 *
 * \snippet Utilities/Test_Overloader.cpp overloader_example
 */
template <class... Fs>
class Overloader;
//******************************************************************************

//******************************************************************************
/*! \cond ELASTICA_INTERNAL */
/*!\brief Specializations for Overloader
 * \ingroup UtilitiesGroup
 */
template <class F1, class F2, class F3, class F4, class F5, class F6, class F7,
          class F8, class... Fs>
class Overloader<F1, F2, F3, F4, F5, F6, F7, F8, Fs...>
    : public F1,
      public F2,
      public F3,
      public F4,
      public F5,
      public F6,
      public F7,
      public F8,
      public Overloader<Fs...> {
 public:
  OVERLOADER_CONSTEXPR Overloader(F1 f1, F2 f2, F3 f3, F4 f4, F5 f5, F6 f6,
                                  F7 f7, F8 f8, Fs... fs)
      : F1(std::move(f1)),
        F2(std::move(f2)),
        F3(std::move(f3)),
        F4(std::move(f4)),
        F5(std::move(f5)),
        F6(std::move(f6)),
        F7(std::move(f7)),
        F8(std::move(f8)),
        Overloader<Fs...>(std::move(fs)...) {}

  using F1::operator();
  using F2::operator();
  using F3::operator();
  using F4::operator();
  using F5::operator();
  using F6::operator();
  using F7::operator();
  using F8::operator();
  using Overloader<Fs...>::operator();
};
/*! \endcond */
//******************************************************************************

//******************************************************************************
/*! \cond ELASTICA_INTERNAL */
/*!\brief Specializations for Overloader
 * \ingroup UtilitiesGroup
 */
template <class F1, class F2, class F3, class F4, class... Fs>
class Overloader<F1, F2, F3, F4, Fs...>
    : public F1, public F2, public F3, public F4, public Overloader<Fs...> {
 public:
  OVERLOADER_CONSTEXPR Overloader(F1 f1, F2 f2, F3 f3, F4 f4, Fs... fs)
      : F1(std::move(f1)),
        F2(std::move(f2)),
        F3(std::move(f3)),
        F4(std::move(f4)),
        Overloader<Fs...>(std::move(fs)...) {}

  using F1::operator();
  using F2::operator();
  using F3::operator();
  using F4::operator();
  using Overloader<Fs...>::operator();
};
/*! \endcond */
//******************************************************************************

//******************************************************************************
/*! \cond ELASTICA_INTERNAL */
/*!\brief Specializations for Overloader
 * \ingroup UtilitiesGroup
 */
template <class F1, class F2, class... Fs>
class Overloader<F1, F2, Fs...>
    : public F1, public F2, public Overloader<Fs...> {
 public:
  OVERLOADER_CONSTEXPR Overloader(F1 f1, F2 f2, Fs... fs)
      : F1(std::move(f1)),
        F2(std::move(f2)),
        Overloader<Fs...>(std::move(fs)...) {}

  using F1::operator();
  using F2::operator();
  using Overloader<Fs...>::operator();
};
/*! \endcond */
//******************************************************************************

//******************************************************************************
/*! \cond ELASTICA_INTERNAL */
/*!\brief Specializations for Overloader
 * \ingroup UtilitiesGroup
 */
template <class F>
class Overloader<F> : public F {
 public:
  explicit OVERLOADER_CONSTEXPR Overloader(F f) : F(std::move(f)) {}

  using F::operator();
};
/*! \endcond */
//******************************************************************************

//******************************************************************************
/*! \cond ELASTICA_INTERNAL */
/*!\brief Specializations for Overloader
 * \ingroup UtilitiesGroup
 */
template <>
class Overloader<> {
 public:
  using type = Overloader;
  ELASTICA_ALWAYS_INLINE void operator()(
      const overloader_detail::no_such_type& /*unused*/) noexcept {}
};
/*! \endcond */
//******************************************************************************

/*!
 * \ingroup UtilitiesGroup
 * \brief Create `Overloader<Fs...>`, see Overloader for details
 */
template <class... Fs>
OVERLOADER_CONSTEXPR Overloader<Fs...> make_overloader(Fs... fs) {
  return Overloader<Fs...>{std::move(fs)...};
}

#undef OVERLOADER_CONSTEXPR

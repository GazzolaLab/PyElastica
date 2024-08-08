#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <array>
#include <cstdint>
#include <sstream>
#include <memory>
#include <algorithm>

#include "Utilities/Invoke.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/NonCopyable.hpp"

namespace elastica {

  namespace detail {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Base type for memoizing a compact set of integral indices
    // \ingroup utils
    */
    struct numeric_at_base_t : NonCopyable {
      //**Constructors**********************************************************
      /*!\name Constructors */
      //@{

      //************************************************************************
      /*!\brief Default constructor.
       */
      numeric_at_base_t() = default;
      //************************************************************************

      //************************************************************************
      /*!\brief Move constructor.
       */
      numeric_at_base_t(numeric_at_base_t&&) = default;
      //@}
      //************************************************************************

      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      virtual ~numeric_at_base_t() = default;
      //@}
      //************************************************************************

      //**Utility***************************************************************
      /*!\name Utility methods */
      //@{

      //************************************************************************
      /*!\brief Gets the indices buffer
       */
      virtual std::size_t* data() noexcept = 0;
      //************************************************************************

      //************************************************************************
      /*!\brief Gets the number of indices
       */
      virtual std::size_t size() const noexcept = 0;
      //************************************************************************

      //@}
      //************************************************************************
    };
    /*! \endcond */
    //**************************************************************************

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief A compact set of integral indices known at compile-time
    // \ingroup utils
    */
    template <std::uint8_t N>
    struct numeric_at_t final : numeric_at_base_t {
      //**Type definitions******************************************************
      //! Type of array for integral indices
      using type = std::array<std::size_t, N>;
      //************************************************************************

      //**Constructors**********************************************************
      /*!\name Constructors */
      //@{

      //************************************************************************
      /*!\brief Array constructor.
      //
      // \param a array of integer indices
      */
      explicit constexpr numeric_at_t(type a) noexcept : args{a} {};
      //************************************************************************

      //************************************************************************
      /*!\brief Move constructor.
       */
      numeric_at_t(numeric_at_t&& other) noexcept
          : args{std::move(other.args)} {};
      //************************************************************************

      //@}
      //************************************************************************

      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~numeric_at_t() override = default;
      //@}
      //**********************************************************************

      //**Utility***************************************************************
      /*!\name Utility methods */
      //@{

      //************************************************************************
      /*!\brief Gets the indices buffer
       */
      std::size_t* data() noexcept override { return args.data(); }
      //************************************************************************

      //************************************************************************
      /*!\brief Gets the number of indices
       */
      std::size_t size() const noexcept override { return N; }
      //************************************************************************

      //@}
      //************************************************************************

      //**Member variables******************************************************
      //! The indices
      type args;
      //************************************************************************
    };
    /*! \endcond */
    //**************************************************************************

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief A compact set of integral indices only known at runtime
    // \ingroup utils
    */
    struct runtime_numeric_at_t final : numeric_at_base_t {
      //**Type definitions******************************************************
      //! Type of storage for integral indices
      using type = std::vector<std::size_t>;
      //************************************************************************

      //**Constructors**********************************************************
      /*!\name Constructors */
      //@{

      //************************************************************************
      /*!\brief vector constructor.
      //
      // \param a vector of integer indices
      */
      explicit runtime_numeric_at_t(type&& a) : args(std::move(a)){};
      //************************************************************************

      //************************************************************************
      /*!\brief Move constructor.
       */
      runtime_numeric_at_t(runtime_numeric_at_t&& other) noexcept
          : args(std::move(other.args)){};
      //************************************************************************

      //@}
      //************************************************************************

      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~runtime_numeric_at_t() override = default;
      //@}
      //**********************************************************************

      //**Utility***************************************************************
      /*!\name Utility methods */
      //@{

      //************************************************************************
      /*!\brief Reserves space in the underlying storage
       *
       * \param size size to reserve
       */
      void reserve(std::size_t size) { args.reserve(size); }
      //************************************************************************

      //************************************************************************
      /*!\brief Gets the indices buffer
       */
      std::size_t* data() noexcept override { return args.data(); }
      //************************************************************************

      //************************************************************************
      /*!\brief Gets the number of indices
       */
      std::size_t size() const noexcept override { return args.size(); }
      //@}
      //************************************************************************

      //**Member variables******************************************************
      //! The runtime indices
      type args{};
      //************************************************************************
    };
    /*! \endcond */
    //**************************************************************************

  }  // namespace detail

  //============================================================================
  //
  //  UTILITY FUNCTIONS
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Constructs a compact set of integral indices from parameters
   * \ingroup utils
   *
   * \example
   * \snippet Test_NumericAt.cpp numeric_at_example
   *
   *
   * \param x An integral index
   * \param ...indices Other integral indices
  */
  template <typename... Ints>
  inline constexpr auto numeric_at(std::size_t x, Ints... ints)
      -> detail::numeric_at_t<sizeof...(Ints) + 1UL> {
    return detail::numeric_at_t<sizeof...(Ints) + 1UL>{make_array(x, ints...)};
    //      sizeof...(Ints) > 0 ? make_array(x, ints...) : make_array<1UL,
    //      int>(x); return detail::at_t<sizeof...(Ints) + 1UL>{
    //          make_array<sizeof...(Ints) + 1UL, int>(x, ints...)};
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Sorts and checks duplicates within a compact set of integral indices
   * \ingroup utils
   *
   * \details
   * Sorts and checks for duplicates. If duplicates exist, an exception is
   * raised
   *
   * \example
   * \snippet Test_NumericAt.cpp scd_example
   *
   * \param ats Numeric indices to be sorted and checked
   * \return ats
   */
  template <std::uint8_t N>
  inline auto sort_and_check_duplicates(detail::numeric_at_t<N>&& ats)
      -> detail::numeric_at_t<N> {
    std::sort(std::begin(ats.args), std::end(ats.args));
    if (std::adjacent_find(std::begin(ats.args), std::end(ats.args)) !=
        std::end(ats.args)) {
      throw std::logic_error("Repeated elements in the set");
    }
    return std::move(ats);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Sorts and checks duplicates within a compact set of integral indices
   * \ingroup utils
   *
   * \details
   * Runtime equivalent of sort_and_check_duplicates()
   *
   * \param ats A unique pointer to numeric indices to be sorted and checked
   * \return ats
   */
  template <typename D>
  inline auto sort_and_check_duplicates(
      std::unique_ptr<detail::numeric_at_base_t, D>&& ats)
      -> std::unique_ptr<detail::numeric_at_base_t, D> {
    std::sort(ats->data(), ats->data() + ats->size());
    if (std::adjacent_find(ats->data(), ats->data() + ats->size()) !=
        (ats->data() + ats->size())) {
      throw std::logic_error("Repeated elements in the set");
    }
    return std::move(ats);
  }
  //****************************************************************************

  /////////// The following functions are no longer used ///////////////////////
  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*
  template <std::uint8_t N>
  auto sanitize(detail::numeric_at_t<N> const& index_set,
                std::size_t n_dofs) noexcept -> detail::numeric_at_t<N> {
    // there may be overflow here, but its less likely
    auto bound = [nd = static_cast<int>(n_dofs)](int x) -> int {
      return x < 0 ? (nd + x) : x;
    };
    return index_apply<N>([&bound, &args = index_set.args](auto... Is) {
      return numeric_at(bound(std::get<Is>(args))...);
    });
  }

  template <std::uint8_t N>
  auto sanitize(detail::numeric_at_t<N>&& index_set,
                std::size_t n_dofs) noexcept -> detail::numeric_at_t<N> {
    auto bound = [n_dofs = static_cast<int>(n_dofs)](int& x) -> void {
      x = x < 0 ? (n_dofs + x) : x;
    };
    index_apply<N>([&bound, &args = index_set.args](auto... Is) {
      return EXPAND_PACK_LEFT_TO_RIGHT(bound(std::get<Is>(args)));
    });
    // force move construction, C++17 should fix this by automatic NRVO
    return std::move(index_set);
  }

  template <typename D>
  auto sanitize(std::unique_ptr<detail::numeric_at_base_t, D>&& index_set,
                std::size_t n_dofs) noexcept
      -> std::unique_ptr<detail::numeric_at_base_t, D> {
    auto bound = [n_dofs = static_cast<int>(n_dofs)](int& x) -> void {
      x = x < 0 ? (n_dofs + x) : x;
    };
    std::for_each(index_set->data(), index_set->data() + index_set->size(),
                  bound);
    // force move construction, C++17 should fix this by automatic NRVO
    return std::move(index_set);
  }

  template <std::uint8_t N>
  auto check_range(detail::numeric_at_t<N>&& sanitized, std::size_t n_dofs)
      -> detail::numeric_at_t<N> {
    // < 0 or == SIZE_MAX, then report error
    // there may be overflow here, but its less likely
    std::ostringstream s_stream;
    auto checker = [&s_stream,
                    n_dofs = static_cast<int>(n_dofs)](int x) -> bool {
      bool failed = (x < 0) || (x >= n_dofs);
      if (failed)
        s_stream << x << ", ";
      return failed;
    };

    bool failed = false;
    /////// compile-time equivalent
    for (const auto& x : sanitized.args) {
      failed |= checker(x);
    }

    if (failed) {
      s_stream << "as the number of degrees of freedom is only " << n_dofs;
      throw std::range_error(s_stream.str());
    }

    // force move construction, C++17 should fix this by automatic NRVO
    return std::move(sanitized);
  }

  //    template <std::uint8_t N>
  //    auto check_range(detail::at_t<N>&& sanitized, std::size_t n_dofs)
  //        -> detail::at_t<N> {
  template <typename D>
  auto check_range(std::unique_ptr<detail::numeric_at_base_t, D>&& sanitized,
                   std::size_t n_dofs)
      -> std::unique_ptr<detail::numeric_at_base_t, D> {
    // < 0 or == SIZE_MAX, then report error
    // there may be overflow here, but its less likely
    std::ostringstream s_stream;
    auto checker = [&s_stream,
                    n_dofs = static_cast<int>(n_dofs)](int const x) -> bool {
      bool failed = (x < 0) || (x >= n_dofs);
      if (failed)
        s_stream << x << ", ";
      return failed;
    };

    // runtime replacement
    const bool failed = [&]() {
      bool int_failed = false;
      auto accumulator = [&int_failed, &checker](int const x) {
        int_failed |= checker(x);
      };
      std::for_each(sanitized->data(), sanitized->data() + sanitized->size(),
                    accumulator);
      return int_failed;
    }();

    if (failed) {
      s_stream << "as the number of degrees of freedom is only " << n_dofs;
      throw std::range_error(s_stream.str());
    }

    // force move construction, C++17 should fix this by automatic NRVO
    return std::move(sanitized);
  }

  template <std::uint8_t N>
  auto sanitize_and_check(detail::numeric_at_t<N> const& index_set,
                          std::size_t n_dofs) -> detail::numeric_at_t<N> {
    return check_range(sanitize(index_set, n_dofs), n_dofs);
  }

  template <std::uint8_t N>
  auto sanitize_and_check(detail::numeric_at_t<N>&& index_set,
                          std::size_t n_dofs) -> detail::numeric_at_t<N> {
    return check_range(sanitize(std::move(index_set), n_dofs), n_dofs);
  }

  // runtime support
  template <typename D>
  auto sanitize_and_check(
      std::unique_ptr<detail::numeric_at_base_t, D>&& index_set,
      std::size_t n_dofs) -> std::unique_ptr<detail::numeric_at_base_t, D> {
    return check_range(sanitize(std::move(index_set), n_dofs), n_dofs);
  }
  */
  /*! \endcond */
  //****************************************************************************

}  // namespace elastica

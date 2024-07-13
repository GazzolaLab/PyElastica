#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

#include "ErrorHandling/Assert.hpp"
#include "Utilities/NonCopyable.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits/Cpp17.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace elastica {

  namespace detail {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief A numeric range of indices
    // \ingroup utils
    //
    // \details
    // Range models a range of indices on which one can iterate upon or evaluate
    // in case of floating point ranges.
    //
    // \tparam F The arithmetic type of the range
    */
    template <typename F, typename Step = F>
    struct NumericRangeImpl : public NonCopyable {
      //**Constructors**********************************************************
      /*!\name Constructors */
      //@{

      //************************************************************************
      /*!\brief Default constructor.
       */
      constexpr NumericRangeImpl(F st, F so, Step se) noexcept(
          std::is_nothrow_move_constructible<F>::value)
          : start_(std::move(st)), stop_(std::move(so)), step_(std::move(se)){};
      //************************************************************************

      //************************************************************************
      /*!\brief Move constructor.
       */
      constexpr NumericRangeImpl(NumericRangeImpl&& other) noexcept(
          std::is_nothrow_move_constructible<F>::value)
          : start_(std::move(other.start_)),
            stop_(std::move(other.stop_)),
            step_(std::move(other.step_)) {}
      //************************************************************************

      //@}
      //************************************************************************

      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~NumericRangeImpl() = default;
      //@}
      //************************************************************************

      //**Member variables******************************************************
      // makes little sense to modify a range after creation, hence declared
      // const
      //! The start index of range
      F const start_;
      //! The stop index of range
      F const stop_;
      //! The step of range
      Step const step_;
      //************************************************************************
    };
    /*! \endcond */
    //**************************************************************************

    //==========================================================================
    //
    //  ALIAS DECLARATIONS
    //
    //==========================================================================

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Alias declaration for an integer NumericRangeImpl.
    // \ingroup utils
    */
    using NumericRange = NumericRangeImpl<std::size_t, std::size_t>;
    /*! \endcond */
    //**************************************************************************

  }  // namespace detail

  namespace tt {
    //**************************************************************************
    /*!\brief Checks whether a type models \elastica NumericRange
    // \ingroup utils type_traits
    //
    // \example
    // \snippet Test_NumericRange.cpp is_numeric_range_example
    */
    template <typename T>
    struct IsNumericRange : ::tt::is_a<::elastica::detail::NumericRangeImpl, T> {};
    //**************************************************************************
  }  // namespace tt

  namespace detail {

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Idiomatic way to construct a half-open numeric range
    //
    // \details
    // The range is closed from left and open from right and models
    // \f$ [a, b) \f$, where \f$ a \f$ is the start and \f$ b \f$ is the stop.
    //
    // \example
    // \snippet Test_NumericRange.cpp numeric_range_example
    //
    // \param start Start of the range
    // \param stop Stop of the range
    // \param step Step of the range
    */
    template <typename F, typename Step>
    inline constexpr auto numeric_range(
        F start, F stop,
        Step step = 1) noexcept(noexcept(NumericRangeImpl<F, Step>{
        start, stop, step})) -> NumericRangeImpl<F, Step> {
      return detail::NumericRangeImpl<F, Step>{start, stop, step};
    }
    /*! \endcond */
    //**************************************************************************

  }  // namespace detail

  //============================================================================
  //
  //  UTILITY FUNCTIONS
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Idiomatic way to construct a half-open numeric range
  //
  // \details
  // The range is closed from left and open from right and models
  // \f$ [a, b) \f$, where \f$ a \f$ is the start and \f$ b \f$ is the stop.
  //
  // \example
  // \snippet Test_NumericRange.cpp numeric_range_example
  //
  // \param start Start of the range
  // \param stop Stop of the range
  // \param step Step of the range
  //
  // \note
  // - while the name is more verbose, it differentiates itself from interface::
  // range().
  // - The parameters are restricted to C++ integral but non-boolean types
  */
  template <typename F, typename Step = int>
  inline decltype(auto) numeric_range(F start, F stop, Step step = 1) {
    static_assert(
        std::is_integral<F>::value && not std::is_same<F, bool>::value,
        "Type must be an arithmetic, non-boolean type");
    static_assert(
        std::is_integral<Step>::value && not std::is_same<Step, bool>::value,
        "Step must be an arithmetic, non-boolean type");

    ELASTICA_ASSERT(start != stop, "Range start and end cannot be equal");
    ELASTICA_ASSERT(step != Step(0), "Step size cannot be zero");
    const bool has_positive_step(step > Step(0));
    ELASTICA_ASSERT(((has_positive_step && start <= stop) ||
                     (!has_positive_step && start >= stop)),
                    "Step moves away from end");
    // switch arguments here so that we can work only with size_t in forcing
    // etc.
    return has_positive_step
               ? detail::numeric_range(start, stop, static_cast<F>(step))
               : detail::numeric_range(stop, start, static_cast<F>(-step));
  }
  //****************************************************************************

  //  template <typename F>
  //  auto sanitize(detail::NumericRangeImpl<F> const& index_range, std::size_t
  //  n_dofs) noexcept
  //  -> detail::NumericRangeImpl<F>{
  //    // there may be overflow here, but its less likely
  //    auto bound = [nd = static_cast<int>(n_dofs)](int x) -> int {
  //      return x < 0 ? (nd + x) : x;
  //    };
  //    return index_apply<N>([&bound, &args = index_set.args](auto... Is) {
  //      return at(bound(std::get<Is>(args))...);
  //    });
  //  }
}  // namespace elastica

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <cstddef>    // size_t
#include <stdexcept>  // exception

#include "ErrorHandling/Assert.hpp"
#include "Utilities/NumericRange.hpp"

namespace elastica {

  namespace detail {

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    template <typename T>
    struct ARangeGenerator {
      // use only with arithmetic types
      using dtype = T;
      using step_type = T;

      ARangeGenerator(T start, T step, std::size_t end_index)
          : range_(std::move(start), T(), std::move(step)),
            end_index_(end_index) {}

      inline auto operator()(std::size_t index) const -> dtype {
        if (index >= end_index_) {
          throw std::out_of_range("Requested index out of range");
        }
        return range_.start_ + static_cast<dtype>(index) * range_.step_;
      }

      NumericRangeImpl<dtype, step_type> range_;
      std::size_t end_index_;
    };
    /*! \endcond */
    //**************************************************************************

  }  // namespace detail

  //****************************************************************************
  /*!\brief Returns a generator producing evenly spaced numbers over a specified
   * interval.
   * \ingroup utils
   *
   * \details
   * Returns `num` evenly spaced samples, calculated over the interval
   * [`start`, `stop`]. The endpoint of the interval can optionally be excluded.
   *
   * See https://numpy.org/doc/stable/reference/generated/numpy.linspace.html
   * for more details.
   *
   * \example
   * \snippet Test_Generators.cpp linspace_eg
   *
   * \param start The starting value of the sequence.
   * \param stop
   * The end value of the sequence, unless endpoint is set to false. In that
   * case, the sequence consists of all but the last of `num + 1` evenly spaced
   * samples, so that stop is excluded. Note that the step size changes when
   * endpoint is false.
   * \param num Number of samples to generate. Default is 50. Must be
   * non-negative.
   * \param end_point If true, stop is the last sample. Otherwise, it is not
   * included. Default is true.
   * \return A generator producing samples in the closed interval
   * [`start`, `stop`] or the half-open interval [`start`, `stop`) (depending on
   * whether endpoint is `true` or `false`).
   */
  template <typename dtype>
  auto linspace_generator(dtype start, dtype stop, std::size_t num = 50UL,
                          bool end_point = true)
      -> detail::ARangeGenerator<dtype> {
    // do all the checking here.
    using RT = detail::ARangeGenerator<dtype>;

    // always an error
    if (num == 0UL) {
      throw std::logic_error("Cannot create values with no indices");
    }

    if (num == 1UL) {
      // gives only the start point so step doesnt matter
      return RT{std::move(start), dtype{}, num};
    }

    // what is <= in this context?
    // if (stop <= start) {
    //   ELASTICA_ASSERT("stop value must be greater than the start value.");
    // }

    // num >= 1UL here
    std::size_t step_factor = end_point ? num - 1UL : num;
    dtype step = (stop - start) / step_factor;  // bug in min-size release clang
    return RT{std::move(start), std::move(step), num};
  }
  //****************************************************************************

}  // namespace elastica

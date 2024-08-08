#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace brigand {

  //****************************************************************************
  /*!\brief Transform any typelist into a tuples::TaggedTuple
   * \ingroup type_traits
   *
   * \details
   * Transform any typelist (such as a `tmpl::list`, `std::tuple` or even a
   * tuples::TaggedTuple) into a tuples::TaggedTuple
   *
   * \usage
   * For any typelist called `L`
   * \code
   * using result = tmpl::as_tagged_tuple<L>;
   * \endcode
   * \metareturns
   * tuples::TaggedTuple
   *
   * \semantics
   * If the type `L` is a typelist with types `Types...`
   * \code
   * result = tuples::TaggedTuple<Types...>;
   * \endcode
   *
   * \example
   * \snippet Test_AsTaggedTuple.cpp as_tagged_tuple_eg
   *
   * \tparam L type list to be converted to a tagged tuple
   */
  template <typename L>
  using as_tagged_tuple = wrap<L, tuples::tagged_tuple_wrapper>;
  //****************************************************************************

}  // namespace brigand

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <utility>

#include "Utilities/NonCopyable.hpp"

/*
 * Developer note
 *
 * One can use the NamedType utility to the same effect, but by design we
 * disallow implicit conversion there. Here the point is to make these
 * Parameters play nice with the Options Parsing
 *
 * Something like this
 * template <typename T>
 * using Parameter = NamedType<T, T, NonCopyable,
 * ImplicitlyConvertibleTo<T>::templ>;
 * should be enough
 */
namespace elastica {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief A keyword argument parameter for cleaner client interfaces
   * \ingroup OptionParsingGroup
   */
  template <typename T>  // The value type
  struct Parameter /*: ::elastica::NonCopyable*/ {
    //**Type definitions********************************************************
    //! Parameter type
    using type = T;
    //**************************************************************************

    //==========================================================================
    //
    //  CONSTRUCTORS and DESTRUCTOR
    //
    //==========================================================================
    // not marked explicit for moving into
    // Clang-Tidy: Single-argument constructors must be marked explicit to avoid
    // unintentional implicit conversions
    Parameter() = delete;
    /*explicit*/ Parameter(type t) : t_(std::move(t)){}; /* NOLINT */
    // /*explicit*/ Parameter(type&& t) : t_(std::move(t)){}; /* NOLINT */
    ~Parameter() = default;
    // Parameter(Parameter&& other) noexcept : t_(std::move(other.t_)) {}
    //**************************************************************************

    //==========================================================================
    //
    //  CONVERSION OPERATORS
    //
    //==========================================================================
    explicit operator type&() & { return t_; }
    explicit operator type() && { return std::move(t_); }
    //**************************************************************************

    //==========================================================================
    //
    //  NON-MUTATING VALUE OPERATOR
    //
    //==========================================================================
    inline auto value() const -> type { return t_; }
    //**************************************************************************

    //==========================================================================
    //
    //  MEMBERS
    //
    //==========================================================================
    //! The value of the parameter
    type t_;
  };
  //****************************************************************************

}  // namespace Options

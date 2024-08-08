// A modified version of
// https://github.com/joboccara/NamedType/blob/master/named_type_impl.hpp
// customized for elastica needs
// See https://raw.githubusercontent.com/joboccara/NamedType/master/LICENSE

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <tuple>
#include <type_traits>
#include <utility>

#include "DefaultTag.hpp"
#include "Types.hpp"

namespace named_type {

  //****************************************************************************
  /*!\brief Class for naming types with affordances
  // \ingroup named_type
  //
  // \example
  // \snippet TODO
  //
  // \tparam T Any value type
  // \tparam Tag Tag type for naming
  // \tparam Affordances... Actions afforded by the underlying value
  //
  // \note
  // To enable reference semantics, use std::ref
  */
  template <typename T, typename Tag, template <typename> class... Affordances>
  class NamedType : public Affordances<NamedType<T, Tag, Affordances...>>... {
   public:
    //**Type definitions********************************************************
    //! Type of the underlying stored value
    using UnderlyingType = T;
    //! Type of the tag parameter
    using TagType = Tag;
    //! Type of the current instance
    using This = NamedType<T, Tag, Affordances...>;
    //! Tuple of all affordances
    using AffordanceTuple = std::tuple<void, Affordances<This>...>;
    //**************************************************************************

   private:
#define REPEAT_2(M, N) M(N) M(N + 1)
#define REPEAT_4(M, N) REPEAT_2(M, N) REPEAT_2(M, N + 2)
#define REPEAT_8(M, N) REPEAT_4(M, N) REPEAT_4(M, N + 4)
#define FRIEND(N)                                                      \
  friend std::tuple_element_t<std::min(N + 1, sizeof...(Affordances)), \
                              AffordanceTuple>;
    // Declare friendship with each requested module so that they have
    // access to private functions
    REPEAT_8(FRIEND, 0UL)

#undef FRIEND
#undef REPEAT_8
#undef REPEAT_4
#undef REPEAT_2

   public:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{

    //**************************************************************************
    /*!\brief The default constructor.
     *
     * \param value reference to the underlying callable, copied into func
     */
    explicit constexpr NamedType(T const& value) noexcept(
        std::is_nothrow_copy_constructible<T>::value)
        : Affordances<This>()..., value_(value){};
    //**************************************************************************

    //**************************************************************************
    /*!\brief The default constructor.
     *
     * \param f temporary value of the underlying callable, moved into func
     */
    explicit constexpr NamedType(T&& value) noexcept(
        std::is_nothrow_move_constructible<T>::value)
        : Affordances<This>()..., value_(std::move(value)){};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Copy constructor.
     *
     */
    constexpr NamedType(NamedType const& other) noexcept(
        std::is_nothrow_copy_constructible<T>::value)
        : Affordances<This>(other)..., value_(other.value_){};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Move constructor.
     *
     */
    constexpr NamedType(NamedType&& other) noexcept
        : Affordances<This>(std::move(other))...,
          value_(std::move(other.value_)){};
    //**************************************************************************
    //@}
    //**************************************************************************

   private:
    //**************************************************************************
    /*!\brief swap function
    //
    */
    friend void swap(NamedType& first, NamedType& second) noexcept {
      using std::swap;
      swap(first.value_, second.value_);
    }
    //**************************************************************************

   public:
    //**Assignment Operators****************************************************
    /*!\name Assignment Operators */
    //@{
    //**************************************************************************
    /*!\brief Copy/move assignment using the copy/move-and-swap idiom, maybe
    // inefficient.
    //
    */
    NamedType& operator=(NamedType other) noexcept {
      swap(*this, other);
      return *this;
    };
    //**************************************************************************
    //@}
    //**************************************************************************

    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~NamedType() = default;
    //@}
    //**************************************************************************

   private:
    //**Access function*********************************************************
    /*!\name Access functions */
    //@{

    //**************************************************************************
    /*!\brief Access to the underlying value
    //
    // \return Mutable lvalue reference to the underlying value
    */
    constexpr T& get_value() & noexcept { return value_; }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to the underlying value
    //
    // \return Constant lvalue reference to the underlying value
    */
    constexpr T const& get_value() const& noexcept { return value_; }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to the underlying value
    //
    // \return Mutable rvalue reference to the underlying value
    */
    constexpr T&& get_value() && noexcept { return static_cast<T&&>(value_); }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to the underlying value
    //
    // \return Const rvalue reference to the underlying value
    */
    constexpr T const&& get_value() const&& noexcept {
      return static_cast<T const&&>(value_);
    }
    //**************************************************************************

    //@}
    //**************************************************************************

   private:
    //**Member variables********************************************************
    T value_;  //!< Underlying value
    //**************************************************************************
  };
  //****************************************************************************

}  // namespace named_type

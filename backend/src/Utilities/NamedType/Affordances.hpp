// A modified version of
// https://github.com/joboccara/NamedType/blob/master/named_type_impl.hpp
// customized for elastica needs
// See https://raw.githubusercontent.com/joboccara/NamedType/master/LICENSE

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <type_traits>

#include "CRTP.hpp"
#include "Traits.hpp"
#include "Types.hpp"
#include "Utilities/TypeTraits/IsCallable.hpp"

namespace named_type {

  //****************************************************************************
  /*!\brief Makes a NamedType gettable
  // \ingroup named_type
  //
  // \example
  // \snippet TODO
  //
  // \tparam T Any NamedType
  */
  template <typename T>
  class Gettable : public CRTPHelper<T, Gettable> {
   protected:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{
    //**************************************************************************
    /*!\brief The default constructor.
     */
    Gettable() = default;
    //**************************************************************************
    //@}
    //**************************************************************************

   private:
    //**Type definitions********************************************************
    using CRTP = CRTPHelper<T, Gettable>;  //<! CRTP Type
    using CRTP::self;                      //<! CRTP methods
    //**************************************************************************

   public:
    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~Gettable() = default;
    //@}
    //**************************************************************************

    //**Get method**************************************************************
    /*!\brief Access to the underlying value
    //
    // returns mutable reference to the stored value, see NamedType::get_value()
    */
    inline constexpr decltype(auto) get() & noexcept {
      return self().get_value();
    }
    //**************************************************************************

    //**Get method**************************************************************
    /*!\brief Access to the underlying value
    //
    // returns constant reference to the stored value, see
    // NamedType::get_value()
    */
    inline constexpr decltype(auto) get() const& noexcept {
      return self().get_value();
    }
    //**************************************************************************

    //**Get method**************************************************************
    /*!\brief Access to the underlying value
    //
    // returns mutable reference to the stored value, see NamedType::get_value()
    */
    inline constexpr decltype(auto) get() && noexcept {
      return static_cast<T&&>(self()).get_value();
    }
    //**************************************************************************

    //**Get method**************************************************************
    /*!\brief Access to the underlying value
    //
    // returns constant reference to the stored value,
    // see NamedType::get_value()
    */
    inline constexpr decltype(auto) get() const&& noexcept {
      return static_cast<T const&&>(self()).get_value();
    }
    //**************************************************************************
  };
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of Traits for a Gettable type
  // \ingroup named_type
  */
  template <>
  struct Traits<Gettable> {
    // no requirements on T
    template <typename T>
    using type = std::true_type;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*!\brief Makes a NamedType non copyable
  // \ingroup named_type
  //
  // \example
  // \snippet TODO
  //
  // \tparam T Any NamedType
  */
  template <typename T>
  class NonCopyable : public CRTPHelper<T, NonCopyable> {
   protected:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{

    //**************************************************************************
    /*!\brief The default constructor.
     */
    NonCopyable() noexcept = default;
    //**************************************************************************

    //==========================================================================
    //
    //  DELETED CONSTRUCTORS
    //
    //==========================================================================
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
    //**************************************************************************

    //**************************************************************************
    /*!\brief The move constructor and assinments
     */
    NonCopyable(NonCopyable&&) noexcept = default;
    NonCopyable& operator=(NonCopyable&&) noexcept = default;
    //**************************************************************************

    //@}
    //**************************************************************************

   private:
    //**Type definitions********************************************************
    using CRTP = CRTPHelper<T, NonCopyable>;  //<! CRTP Type
    using CRTP::self;                         //<! CRTP methods
    //**************************************************************************

   public:
    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~NonCopyable() = default;
    //@}
    //**************************************************************************
  };
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of Traits for a NonCopyable type
  // \ingroup named_type
  */
  template <>
  struct Traits<NonCopyable> {
    // no requirements on T
    template <typename T>
    using type = std::true_type;
  };
  /*! \endcond */
  //****************************************************************************

  //****************************************************************************
  /*!\brief Makes a NamedType with the operator()
  // \ingroup named_type
  //
  // \example
  // \snippet TODO
  //
  // \tparam T Any NamedType
  */
  template <typename T>
  class Callable : public CRTPHelper<T, Callable> {
   protected:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{

    //**************************************************************************
    /*!\brief The default constructor.
     */
    Callable() = default;
    //**************************************************************************
    //@}
    //**************************************************************************

   private:
    //**Type definitions********************************************************
    using CRTP = CRTPHelper<T, Callable>;  //<! CRTP Type
    using CRTP::self;                      //<! CRTP methods
    //**************************************************************************

   public:
    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~Callable() = default;
    //@}
    //**************************************************************************

    //**Call method*************************************************************
    /*!\brief Access to the underlying callable's function
    //
    // returns the operator() with a wrapped syntax
    */
    template <typename... Args>
    decltype(auto) operator()(Args&&... args) const
        noexcept(noexcept(self().get_value()(std::forward<Args>(args)...))) {
      return self().get_value()(std::forward<Args>(args)...);
    }
    //**************************************************************************
  };
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialization of Traits for a Callable type
  // \ingroup named_type
  */
  template <>
  struct Traits<Callable> {
    template <typename T>
    using type = ::tt::is_callable<T>;
  };
  /*! \endcond */
  //****************************************************************************

}  // namespace named_type

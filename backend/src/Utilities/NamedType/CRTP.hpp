// Thanks :
// https://github.com/joboccara/NamedType/blob/master/include/NamedType/crtp.hpp

#pragma once

#include "Utilities/ForceInline.hpp"  // introduces dependency

namespace named_type {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief CRTP helper class
  // \ingroup util
  //
  // CRTPHelper eases coding CRTP classes in \elastica by providing a common
  // interface to the underlying data type, using the self() methods
  //
  // \example
  // \snippet Test_CRTP.cpp crtp_example
  //
  // \tparam Derived The final derived class
  // \tparam Mixin The interface mixed-in class, templated to prevent diamond
  // hierarchies with multiple inheritance. Constraint : At least one of the
  // templated types needs to be Derived
  */
  template <typename Derived,
            template <typename /* Derived */, typename...> class Mixin>
  class CRTPHelper {
   protected:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{

    //**************************************************************************
    /*!\brief The default constructor.
     */
    CRTPHelper() = default;
    //**************************************************************************
    //@}
    //**************************************************************************

   private:
    //**Type definitions********************************************************
    //! Type of the bottom level derived class
    using Self = Derived;
    //! Reference type of the bottom level derived class
    using Reference = Self&;
    //! const reference type of the bottom level derived class
    using ConstReference = Self const&;
    //**************************************************************************

   protected:
    //**Self method*************************************************************
    /*!\brief Access to the underlying derived
    //
    // \return Mutable reference to the underlying derived
    //
    // Safely down-casts this module to the underlying derived type, using
    // the Curiously Recurring Template Pattern (CRTP).
    */
    ELASTICA_ALWAYS_INLINE constexpr auto self() & noexcept -> Reference {
      return static_cast<Reference>(*this);
    }
    //**************************************************************************

    //**Self method*************************************************************
    /*!\brief Access to the underlying derived
    //
    // \return Const reference to the underlying derived
    //
    // Safely down-casts this module to the underlying derived type, using
    // the Curiously Recurring Template Pattern (CRTP).
    */
    ELASTICA_ALWAYS_INLINE constexpr auto self() const& noexcept
        -> ConstReference {
      return static_cast<ConstReference>(*this);
    }
    //**************************************************************************

   public:
    //**Deleted Self method*****************************************************
    // They can never be called anyway since they do not participate in overload
    // resolution (since the `this` pointer is always a & type)
    constexpr auto self() && = delete;
    constexpr auto self() const&& = delete;
    //**************************************************************************
  };
  //****************************************************************************

}  // namespace named_type

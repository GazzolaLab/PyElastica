#pragma once

namespace elastica {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Base class for non-copyable class instances.
  // \ingroup util
  //
  // The NonCopyable class is intended to work as a base class for non-copyable
  // classes. Both the copy constructor and the copy assignment operator are
  // deleted in order to prohibit copy operations of the derived classes.
  //
  // \note
  // It is not necessary to publicly derive from this class. It is sufficient to
  // derive privately to prevent copy operations on the derived class.
  //
  // \example
     \code
       class A : private NonCopyable
       { ... };
     \endcode
  // https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Non-copyable_Mixin
  //
  // \see NonCreatable
  */
  class NonCopyable {
   public:
    //==========================================================================
    //
    //  DELETED CONSTRUCTORS
    //
    //==========================================================================
    NonCopyable(const NonCopyable&) = delete;
    NonCopyable& operator=(const NonCopyable&) = delete;
    //**************************************************************************

   protected:
    //==========================================================================
    //
    //  CONSTRUCTORS and DESTRUCTOR
    //
    //==========================================================================
    inline NonCopyable() noexcept = default;
    inline ~NonCopyable() = default;
    NonCopyable(NonCopyable&&) noexcept = default;
    NonCopyable& operator=(NonCopyable&&) noexcept = default;
    //**************************************************************************
  };
  //****************************************************************************

}  // namespace elastica

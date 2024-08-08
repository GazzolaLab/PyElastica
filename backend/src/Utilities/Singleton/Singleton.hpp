//==============================================================================
/*!
 *  Original file pe/util/singleton/Singleton.h
 *  Original brief Header file for the Singleton class
 *
 *  Original Copyright (C) 2009 Klaus Iglberger
 *
 *  This file is part of pe.
 *
 *  pe is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 *  pe is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along with
 * pe. If not, see <http://www.gnu.org/licenses/>.
 */
//==============================================================================

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

///
#include "Utilities/Singleton/Types.hpp"
///
#include "Utilities/Invoke.hpp"
#include "Utilities/NonCopyable.hpp"
#include "Utilities/Singleton/Dependency.hpp"
#include "Utilities/Singleton/TypeTraits.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace elastica {

  namespace tt {

    //**************************************************************************
    /*!\brief Type of a Singleton Storage
    // \ingroup singleton
    //
    // \tparam T : Type to check
     */
    template <typename T>
    struct SingletonStorage {
      using type = std::shared_ptr<T>;
    };
    //**************************************************************************

    //**************************************************************************
    template <typename T>
    struct AddSingletonStorage : public IsSingleton<T> {
      using IsSingleton<T>::value;
      static_assert(value, "Not a singleton");
      // TODO : Is there a way to use a common
      using type = typename SingletonStorage<T>::type;
    };
    //**************************************************************************

  }  // namespace tt

//******************************************************************************
/*!\brief Helper for establishing friendships with the base Singleton
 * \ingroup singleton
 *
 * \details
 * Helper macro for establishing friendships with the base Singleton and other
 * helpers (type traits, dependencies)
 *
 * \usage
 * While defining the class deriving from Singleton, use this macro as shown
 * below. The access specifier does not matter (the macro can be placed within
 * the private, protected or public context.
   \code
   class MySingleton : private Singleton<MySingleton>{
        //  ...details...
        BEFRIEND_SINGLETON;
   };
   \endcode
 * which hides the details of Singleton implementation from the user.
 *
 * \note
 * We add Dependency to the list too, as dependencies rely
 * on instantiating the dependent singleton derived class using the static
 * instance method()
 */
#define BEFRIEND_SINGLETON                                             \
  template <typename, class...>                                        \
  friend class ::elastica::Singleton;                                  \
  template <typename, typename, bool>                                  \
  friend struct ::elastica::tt::singleton_detail::HasCyclicDependency; \
  template <typename>                                                  \
  friend struct ::elastica::tt::IsSingleton;                           \
  template <typename>                                                  \
  friend class ::elastica::Dependency

  //****************************************************************************
  /*!\brief An abstract Singleton
  // \ingroup singleton
  //
  // \details
  // Implements the abstract Singleton pattern in \elastica
  //
  // \example
  // \snippet Singleton/Test_Singleton.cpp singleton_example
  //
  // \tparam T Type to be made singleton
  // \tparam Deps... Dependencies of T
   */
  template <class T, class... Deps>
  class Singleton : private NonCopyable {
   public:
    //**Type definitions********************************************************
    //! Type of the current singleton class
    using SingletonType = Singleton<T, Deps...>;
    //! Typelist of dependencies
    using Dependencies = tmpl::list<Deps...>;
    //! Storage of the current type T
    using SingletonStorageType = typename tt::SingletonStorage<T>::type;
    //! Storage of dependencies of the current type T
    using DependencyStorage = tmpl::as_tuple<
        tmpl::transform<Dependencies, tt::SingletonStorage<tmpl::_1>>>;
    //**************************************************************************

    //**Get functions***********************************************************
    /*!\brief Number of dependencies
    //
    // \return the number of dependencies of the current type T
    */
    constexpr static std::size_t get_number_of_dependencies() noexcept {
      return tmpl::size<Dependencies>::value;
    }
    //**************************************************************************

    //**Access functions********************************************************
    /*!\brief Access to the underlying singleton
    //
    // \return Handle to the sole singleton of type T
    */
    static SingletonStorageType instance() {
      static SingletonStorageType object(new T());
      return object;
    }
    //**************************************************************************

   protected:
    //==========================================================================
    //
    //  CONSTRUCTOR
    //
    //==========================================================================

    //**Constructor*************************************************************
    /*!\name Constructor */
    //@{
    Singleton() {
      // Fill with instances
      // idiomatic but need to use shared_ptr
      cpp17::apply(
          [](auto&... dependency) {
            auto assign = [](auto& dep) {
              // ref -> type
              using PointerToDependency = std::decay_t<decltype(dep)>;
              // shared ptr -> element
              dep = PointerToDependency::element_type::instance();
            };
            EXPAND_PACK_LEFT_TO_RIGHT(assign(dependency));
          },
          dependencies_);

      static_assert(
          cpp17::conjunction_v<tt::IsSingleton<T>, tt::IsSingleton<Deps>...>,
          "Type does not derive from Singleton");
      static_assert(!tt::has_cyclic_dependency<Deps...>,
                    "Cyclic dependency detected!");
    };
    //@}
    //**************************************************************************

    //**Destructor**************************************************************
    // No explicitly declared destructor.
    //**************************************************************************

   private:
    // Allowing friend access to IsSingleton to access
    // Singleton type even when Singleton is privately/publicly derived.
    template <typename>
    friend struct tt::IsSingleton;

    ///////// std::tuple<std::shared_ptr<Dependencies>...>
    //! Store handles to all dependencies
    DependencyStorage dependencies_;
  };
  //****************************************************************************

}  // namespace elastica

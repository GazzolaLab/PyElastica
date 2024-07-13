#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <type_traits>

///
#include "Utilities/Singleton/Types.hpp"
///
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/Cpp17.hpp"

namespace elastica {

  namespace tt {

    //**************************************************************************
    /*!\brief Checks for a Singleton Type
    // \ingroup singleton
    //
    // \details
    // Checks whether a type `T` is a \elastica Singleton type.
    //
    // \tparam T : Type to check
    //
    // \example
    // \snippet Singleton/Test_TypeTraits.cpp is_singleton_example
    //
    // Developer note:
    // The following is a perfectly valid implementation for IsSingleton,
    // however it creates issues with friend scope.
       \code
         template <typename T>
         struct IsSingleton : public std::is_base_of<typename
         T::SingletonType,T> {};
       \endcode
    // IsSingleton will be a friend of all classes that use our Singleton, using
    // the BEFRIEND_SINGLETON macro. The issue arises because we try and derive
    // directly from is_base_of, which is not a friend of T. Rather, we push the
    // is_base_of inside the definition of the struct, where friendship rules
    // apply.
    */
    template <typename T>
    struct IsSingleton {
      constexpr static bool value =
          std::is_base_of<typename T::SingletonType, T>::value;
    };
    //**************************************************************************

    namespace singleton_detail {

      //========================================================================
      //
      // RESOLVING CYCLIC DEPENDENCIES
      //
      //========================================================================

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Helper to determine whether there are cyclic dependencies
      // within a list of dependencies
      // \ingroup singleton
      //
      // \tparam T Type to be checked for cyclic lifetime
      // \tparam TL Type list of checked lifetime dependencies
      */
      template <typename T, typename TL,
                bool C>  // auxiliary param indicating a cyclic dependency
      struct HasCyclicDependency;
      /*! \endcond */
      //************************************************************************

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Specialization for T in TL
      // \ingroup singleton
      //
      // \details
      // Specialization in case T is contained in TL, which implies a
      // cyclic dependency, hence deriving from true type
      //
      // \tparam T Type to be checked for cyclic lifetime
      // \tparam TL Type list of checked lifetime dependencies
      */
      template <typename T, typename TL>
      struct HasCyclicDependency<T, TL, true> : std::true_type {};
      /*! \endcond */
      //************************************************************************

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Auxiliary helper to determine whether there are cyclic
      // dependencies within a list of dependencies
      // \ingroup singleton
      //
      // \tparam TL Type list of checked lifetime dependencies
      // \tparam DL Type list of lifetime dependencies checked for
      */
      template <typename TL, typename DL>
      struct HasCyclicDependencyHelper;
      /*! \endcond */
      //************************************************************************

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Specialization for T in TL
      // \ingroup singleton
      //
      // \tparam T Type to be checked for cyclic lifetime
      // \tparam TL Type list of checked lifetime dependencies
      */
      template <typename T, typename TL,
                bool C =
                    ::tmpl::list_contains_v<TL, T>>  // Is T contained in TL?
      struct HasCyclicDependency {
        using ETL = tmpl::push_back<TL, T>;  // Extended Type List to be checked
        static constexpr bool value =
            HasCyclicDependencyHelper<ETL, typename T::Dependencies>::value;
      };
      /*! \endcond */
      //************************************************************************

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Specialization for a typelist of dependencies
      // \ingroup singleton
      //
      // \tparam TL Type list of checked lifetime dependencies
      // \tparam DL Type list of lifetime dependencies checked for
      */
      template <typename TL, template <typename...> class DL,
                typename... D>  // Dependencies contained in the Type List
      struct HasCyclicDependencyHelper<TL, DL<D...>> {
        static constexpr bool value = cpp17::disjunction_v<
            HasCyclicDependency<D, TL, ::tmpl::list_contains_v<TL, D>>...>;
      };
      /*! \endcond */
      //************************************************************************

    }  // namespace singleton_detail

    //**************************************************************************
    /*!\brief Variable template for detecting cyclic dependencies within a list
    // \ingroup singleton
    //
    // \details
    // The has_cyclic_dependency variable template indicates the presence of
    // cyclic dependencies with a variadic set of dependencies. It metareturns
    // true if any one of the dependency types has a cyclic dependency to any
    // other dependency within the list.
    //
    // \example
    // \snippet Singleton/Test_TypeTraits.cpp has_cyclic_dependency
    //
    // \tparam D... Type of dependencies to be checked
    */
    template <typename... D>  // Dependencies (types)
    constexpr bool has_cyclic_dependency = cpp17::disjunction_v<
        singleton_detail::HasCyclicDependency<D, ::tmpl::empty_sequence>...>;
    //**************************************************************************

  }  // namespace tt

}  // namespace elastica

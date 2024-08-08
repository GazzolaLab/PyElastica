// Reused from SpECTRE : https://spectre-code.org/
// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Utilities/Demangle.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/ArraySize.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/TypeTraits/IsStdArray.hpp"
#include "Utilities/TypeTraits/Void.hpp"

//==============================================================================
//
//  DOXYGEN DOCUMENTATION
//
//==============================================================================

//******************************************************************************
/*!\defgroup PrettyTypeGroup Pretty Type
 *
 * \brief Pretty printing of types
 */
//******************************************************************************

/*!
 * \ingroup PrettyTypeGroup
 * \brief Contains all functions that are part of PrettyType, used for printing
 * types in a pretty manner.
 */
namespace pretty_type {

  namespace detail {
    //**Type definitions********************************************************
    using str_const = char const* const;  //<! Constant string type
    //**************************************************************************

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Auxiliary helper struct for pretty printing a type.
    // \ingroup PrettyTypeGroup
    //
    // \tparam T the type whose name to print
    */
    template <typename T>
    struct Type;
    /*! \endcond */
    //**************************************************************************

    template <>
    struct Type<char> {
      using type = char;
      static constexpr str_const type_name = {"char"};
    };

    template <>
    struct Type<signed char> {
      using type = signed char;
      static constexpr str_const type_name = {"signed char"};
    };

    template <>
    struct Type<unsigned char> {
      using type = unsigned char;
      static constexpr str_const type_name = {"unsigned char"};
    };

    template <>
    struct Type<wchar_t> {
      using type = wchar_t;
      static constexpr str_const type_name = {"wchar_t"};
    };

    template <>
    struct Type<char16_t> {
      using type = char16_t;
      static constexpr str_const type_name = {"char16_t"};
    };

    template <>
    struct Type<char32_t> {
      using type = char32_t;
      static constexpr str_const type_name = {"char32_t"};
    };

    template <>
    struct Type<int> {
      using type = int;
      static constexpr str_const type_name = {"int"};
    };

    template <>
    struct Type<unsigned int> {
      using type = unsigned int;
      static constexpr str_const type_name = {"unsigned int"};
    };

    template <>
    struct Type<long> {
      using type = long;
      static constexpr str_const type_name = {"long"};
    };

    template <>
    struct Type<unsigned long> {
      using type = unsigned long;
      static constexpr str_const type_name = {"unsigned long"};
    };

    template <>
    struct Type<long long> {
      using type = long long;
      static constexpr str_const type_name = {"long long"};
    };

    template <>
    struct Type<unsigned long long> {
      using type = unsigned long long;
      static constexpr str_const type_name = {"unsigned long long"};
    };

    template <>
    struct Type<short> {
      using type = short;
      static constexpr str_const type_name = {"short"};
    };

    template <>
    struct Type<unsigned short> {
      using type = unsigned short;
      static constexpr str_const type_name = {"unsigned short"};
    };

    template <>
    struct Type<float> {
      using type = float;
      static constexpr str_const type_name = {"float"};
    };

    template <>
    struct Type<double> {
      using type = double;
      static constexpr str_const type_name = {"double"};
    };

    template <>
    struct Type<long double> {
      using type = long double;
      static constexpr str_const type_name = {"long double"};
    };

    template <>
    struct Type<bool> {
      using type = bool;
      static constexpr str_const type_name = {"bool"};
    };

    template <>
    struct Type<void> {
      using type = void;
      static constexpr str_const type_name = {"void"};
    };

    template <>
    struct Type<std::string> {
      using type = std::string;
      static constexpr str_const type_name = {"std::string"};
    };

    //**Type definitions********************************************************
    //! Alias template type for a compile-time typelist map of (T, Type<T>)
    template <typename... T>
    using TemplateMap_t = tmpl::map<tmpl::pair<T, Type<T>>...>;
    //**************************************************************************

    //**Free functions**********************************************************
    /*!\brief Adds qualifiers to the string representation of the type
    // \ingroup PrettyTypeGroup
    //
    // \tparam T The type to be pretty printed
    // \return std::string with qualified name of type
    */
    template <typename T>
    std::string add_qualifiers() {
      std::stringstream ss;
      if (std::is_pointer<T>::value) {
        if (std::is_const<std::remove_pointer_t<T>>::value) {
          ss << " const";
        }
        if (std::is_volatile<std::remove_pointer_t<T>>::value) {
          ss << " volatile";
        }
        ss << "*";
      }
      if (std::is_const<std::remove_reference_t<T>>::value) {
        ss << " const";
      }
      if (std::is_volatile<std::remove_reference_t<T>>::value) {
        ss << " volatile";
      }
      if (std::is_reference<T>::value) {
        ss << "&";
      }
      return ss.str();
    }
    //**************************************************************************

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Helper auxiliary struct to construct pretty names of containers
     * \ingroup PrettyTypeGroup
     *
     * \tparam T the type whose name to print
     * \tparam M the map of the basic types to print
     * \tparam KT the struct holding the template alias template_list which is a
     * list of known specializations of construct_name for those containers
     */
    template <typename T, typename M, typename KT,
              typename = std::nullptr_t  // Restricting condition
              >
    struct construct_name;
    //**************************************************************************

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tmpl::has_key<
                     M, std::decay_t<std::remove_pointer_t<T>>>::value == 1>> {
      static std::string get() {
        constexpr str_const t =
            tmpl::at<M, std::decay_t<std::remove_pointer_t<T>>>::type_name;
        std::stringstream ss;
        ss << t << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<
            tmpl::has_key<M, std::decay_t<std::remove_reference_t<
                                 std::remove_pointer_t<T>>>>::value == 0 and
            std::is_same<
                tmpl::list<>,
                tmpl::find<
                    typename KT::template template_list<std::decay_t<
                        std::remove_reference_t<std::remove_pointer_t<T>>>>,
                    std::is_base_of<std::true_type, tmpl::_1>>>::value>> {
      static std::string get() {
        std::stringstream ss;
        ss << demangle(typeid(T).name());
        return ss.str();
      }
    };

    // STL Sequences
    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<std::vector, std::decay_t<std::remove_reference_t<
                                             std::remove_pointer_t<T>>>>>> {
      using type = std::decay_t<std::remove_pointer_t<T>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::vector<"
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::value_type>>,
                             M, KT>::get();
        ss << add_qualifiers<typename type::value_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_std_array_v<
            std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::array<"
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::value_type>>,
                             M, KT>::get();
        ss << add_qualifiers<typename type::value_type>();
        ss << ", " << tt::array_size<type>::value << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<std::deque, std::decay_t<std::remove_reference_t<
                                            std::remove_pointer_t<T>>>>>> {
      using type = std::decay_t<std::remove_pointer_t<T>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::deque<"
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::value_type>>,
                             M, KT>::get();
        ss << add_qualifiers<typename type::value_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<
            std::forward_list,
            std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>>>> {
      using type = std::decay_t<std::remove_pointer_t<T>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::forward_list<"
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::value_type>>,
                             M, KT>::get();
        ss << add_qualifiers<typename type::value_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<std::list, std::decay_t<std::remove_reference_t<
                                           std::remove_pointer_t<T>>>>>> {
      using type = std::decay_t<std::remove_pointer_t<T>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::list<"
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::value_type>>,
                             M, KT>::get();
        ss << add_qualifiers<typename type::value_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    // STL Associative containers
    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<std::map, std::decay_t<std::remove_reference_t<
                                          std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::map<"
           << construct_name<
                  std::decay_t<std::remove_pointer_t<typename type::key_type>>,
                  M, KT>::get()
           << add_qualifiers<typename type::key_type>() << ", "
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::mapped_type>>,
                             M, KT>::get()
           << add_qualifiers<typename type::mapped_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<std::multimap, std::decay_t<std::remove_reference_t<
                                               std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::multimap<"
           << construct_name<
                  std::decay_t<std::remove_pointer_t<typename type::key_type>>,
                  M, KT>::get()
           << add_qualifiers<typename type::key_type>() << ", "
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::mapped_type>>,
                             M, KT>::get()
           << add_qualifiers<typename type::mapped_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<std::multiset, std::decay_t<std::remove_reference_t<
                                               std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::multiset<"
           << construct_name<
                  std::decay_t<std::remove_pointer_t<typename type::key_type>>,
                  M, KT>::get()
           << add_qualifiers<typename type::key_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<std::set, std::decay_t<std::remove_reference_t<
                                          std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::set<"
           << construct_name<
                  std::decay_t<std::remove_pointer_t<typename type::key_type>>,
                  M, KT>::get()
           << add_qualifiers<typename type::key_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    // STL Unordered associative containers

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<
            std::unordered_map,
            std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::unordered_map<"
           << construct_name<
                  std::decay_t<std::remove_pointer_t<typename type::key_type>>,
                  M, KT>::get()
           << add_qualifiers<typename type::key_type>() << ", "
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::mapped_type>>,
                             M, KT>::get()
           << add_qualifiers<typename type::mapped_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<
            std::unordered_multimap,
            std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::unordered_multimap<"
           << construct_name<
                  std::decay_t<std::remove_pointer_t<typename type::key_type>>,
                  M, KT>::get()
           << add_qualifiers<typename type::key_type>() << ", "
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::mapped_type>>,
                             M, KT>::get()
           << add_qualifiers<typename type::mapped_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<
            std::unordered_multiset,
            std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::unordered_multiset<"
           << construct_name<
                  std::decay_t<std::remove_pointer_t<typename type::key_type>>,
                  M, KT>::get()
           << add_qualifiers<typename type::key_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<
            std::unordered_set,
            std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::unordered_set<"
           << construct_name<
                  std::decay_t<std::remove_pointer_t<typename type::key_type>>,
                  M, KT>::get()
           << add_qualifiers<typename type::key_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };
    // STL Container adaptors
    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<
            std::priority_queue,
            std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::priority_queue<"
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::value_type>>,
                             M, KT>::get()
           << add_qualifiers<typename type::value_type>() << ", "
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::container_type>>,
                             M, KT>::get()
           << add_qualifiers<typename type::container_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<std::queue, std::decay_t<std::remove_reference_t<
                                            std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::queue<"
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::value_type>>,
                             M, KT>::get()
           << add_qualifiers<typename type::value_type>() << ", "
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::container_type>>,
                             M, KT>::get()
           << add_qualifiers<typename type::container_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<std::stack, std::decay_t<std::remove_reference_t<
                                            std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::stack<"
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::value_type>>,
                             M, KT>::get()
           << add_qualifiers<typename type::value_type>() << ", "
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 typename type::container_type>>,
                             M, KT>::get()
           << add_qualifiers<typename type::container_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    // STL Smart pointers
    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<
            std::unique_ptr,
            std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::unique_ptr<"
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 decltype(*std::declval<type>())>>,
                             M, KT>::get()
           << add_qualifiers<
                  std::remove_reference_t<decltype(*std::declval<type>())>>()
           << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<
            std::shared_ptr,
            std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      static std::string get() {
        std::stringstream ss;
        ss << "std::shared_ptr<"
           << construct_name<std::decay_t<std::remove_pointer_t<
                                 decltype(*std::declval<type>())>>,
                             M, KT>::get()
           << add_qualifiers<
                  std::remove_reference_t<decltype(*std::declval<type>())>>()
           << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };

    template <typename T, typename M, typename KT>
    struct construct_name<
        T, M, KT,
        Requires<tt::is_a_v<std::weak_ptr, std::decay_t<std::remove_reference_t<
                                               std::remove_pointer_t<T>>>>>> {
      using type =
          std::decay_t<std::remove_reference_t<std::remove_pointer_t<T>>>;
      using element_type = typename type::element_type;
      static std::string get() {
        std::stringstream ss;
        ss << "std::weak_ptr<"
           << construct_name<std::decay_t<std::remove_pointer_t<element_type>>,
                             M, KT>::get()
           << add_qualifiers<element_type>() << ">";
        ss << add_qualifiers<T>();
        return ss.str();
      }
    };
  }  // namespace detail

  //**Type definitions**********************************************************
  /*!
   * \ingroup PrettyTypeGroup
   * \brief typelist of basic types that can be pretty printed
   *
   * These are specializations of tt::Type<T>
   */
  using basics_map = detail::TemplateMap_t<
      char, signed char, unsigned char, wchar_t, char16_t, char32_t, int,
      unsigned int, long, unsigned long, long long, unsigned long long, short,
      unsigned short, float, double, long double, bool, std::string>;
  //****************************************************************************

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================
  //****************************************************************************
  /*!\brief A list of type traits to check if something is an STL member
   * \ingroup PrettyTypeGroup
   *
   * Contains a template alias with the name template_list of type traits that
   * identify STL containers that can be pretty printed
   */
  struct stl_templates {
    /// List of known STL classes that can be pretty printed
    template <typename X>
    using template_list = tmpl::list<
        tt::is_a<std::vector, X>, tt::is_std_array<X>, tt::is_a<std::deque, X>,
        tt::is_a<std::forward_list, X>, tt::is_a<std::list, X>,
        tt::is_a<std::map, X>, tt::is_a<std::set, X>,
        tt::is_a<std::multiset, X>, tt::is_a<std::multimap, X>,
        tt::is_a<std::unordered_map, X>, tt::is_a<std::unordered_multimap, X>,
        tt::is_a<std::unordered_multiset, X>, tt::is_a<std::unordered_set, X>,
        tt::is_a<std::priority_queue, X>, tt::is_a<std::queue, X>,
        tt::is_a<std::stack, X>, tt::is_a<std::unique_ptr, X>,
        tt::is_a<std::shared_ptr, X>, tt::is_a<std::weak_ptr, X>>;
  };
  //****************************************************************************

  //============================================================================
  //
  //  GLOBAL FUNCTIONS
  //
  //============================================================================
  //****************************************************************************
  /*!\brief Returns a string with the prettiest typename known for the type T.
   * \ingroup PrettyTypeGroup
   *
   *  Example usage: auto name = get_name<T>();
   *
   *  \tparam T the type to print
   *  \tparam Map a tmpl::map of basic types (non-containers) and their Type<T>
   *  specializations that determine how to print the type name in a pretty form
   *  \tparam KnownTemplates struct hold template alias tmpl::list of is_...
   *  that are known how to be printed pretty \return std::string containing the
   *  typename
   */
  template <typename T, typename Map = basics_map,
            typename KnownTemplates = stl_templates>
  std::string get_name() {
    return detail::construct_name<T, Map, KnownTemplates>::get();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns a string with the prettiest typename known for the runtime
   * type of x.
   * \ingroup PrettyTypeGroup
   *
   * The result will generally not be as pretty as the result of
   * get_name, but this function will report the derived type of a class
   * when only given a base class reference, which get_type cannot do.
   */
  template <typename T>
  std::string get_runtime_type_name(const T& x) {
    return demangle(typeid(x).name());
  }
  //****************************************************************************

  namespace detail {
    std::string extract_short_name(std::string name);
  }  // namespace detail

  //****************************************************************************
  /*!\brief Return the "short name" of a class, that is, the name
   * without template parameters or scopes.
   * \ingroup PrettyTypeGroup
   *
   */
  template <typename T>
  std::string short_name() {
#define SHORT_NAME_DEMANGLE_WORKAROUND

    // We need a workaround to demangle the compiler generated names for
    // short name extraction. Name demangling is expensive as it necessarily
    // involves heap allocations.
#ifdef SHORT_NAME_DEMANGLE_WORKAROUND
    return detail::extract_short_name(demangle(typeid(T).name()));
#else
    //    return detail::extract_short_name(typeid(T).name());
#endif

#undef SHORT_NAME_DEMANGLE_WORKAROUND
  }
  //****************************************************************************

  namespace detail {
    template <typename T, typename = cpp17::void_t<>>
    struct name_helper {
      static std::string name() { return pretty_type::short_name<T>(); }
    };

    template <typename T>
    struct name_helper<T, cpp17::void_t<decltype(T::name())>> {
      static std::string name() { return T::name(); }
    };
  }  // namespace detail

  //****************************************************************************
  /*!\brief Return the result of the `name()` member of a class. If a class
   * doesn't have a `name()` member, call `pretty_type::short_name<T>()`
   * instead.
   * \ingroup PrettyTypeGroup
   *
   * \warning Do not use this inside the `name()` member of struct. This
   * can lead to recursion as `pretty_type::name<Tag>()` will call the `name()`
   * member of the struct.
   */
  template <typename T>
  std::string name() {
    return detail::name_helper<T>::name();
  }
  //****************************************************************************

}  // namespace pretty_type

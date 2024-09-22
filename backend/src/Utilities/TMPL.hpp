#pragma once

// Since this header only wraps brigand and several additions to it we mark
// it as a system header file so that clang-tidy ignores it.
#ifdef __GNUC__
#pragma GCC system_header
#endif

#define BRIGAND_NO_BOOST_SUPPORT
#include <brigand/brigand.hpp>
#include <type_traits>

#include "Requires.hpp"
//#include "TypeTraits.hpp"

namespace tmpl = brigand;

// Use spectre's TMPL

namespace brigand {
  /// Check if a typelist contains an item.
  template <typename Sequence, typename Item>
  using list_contains =
      tmpl::found<Sequence, std::is_same<tmpl::_1, tmpl::pin<Item>>>;

  template <typename Sequence, typename Item>
  constexpr const bool list_contains_v = list_contains<Sequence, Item>::value;

  /// Obtain the elements of `Sequence1` that are not in `Sequence2`.
  template <typename Sequence1, typename Sequence2>
  using list_difference =
      fold<Sequence2, Sequence1, lazy::remove<_state, _element>>;

}  // namespace brigand

namespace brigand {
  namespace detail {
    template <typename S, typename E>
    struct remove_duplicates_helper {
      using type = typename std::conditional<
          std::is_same<index_of<S, E>, no_such_type_>::value, push_back<S, E>,
          S>::type;
    };
  }  // namespace detail

  template <typename List>
  using remove_duplicates =
      fold<List, list<>, detail::remove_duplicates_helper<_state, _element>>;
}  // namespace brigand

namespace brigand {
  template <typename FirstList, typename SecondList>
  using cartesian_product = reverse_fold<
      list<FirstList, SecondList>, list<list<>>,
      lazy::join<lazy::transform<
          _2, defer<lazy::join<lazy::transform<
                  parent<_1>,
                  defer<bind<list, lazy::push_front<_1, parent<_1>>>>>>>>>>;

}

namespace brigand {
  template <bool>
  struct conditional;

  template <>
  struct conditional<true> {
    template <typename T, typename F>
    using type = T;
  };

  template <>
  struct conditional<false> {
    template <typename T, typename F>
    using type = F;
  };

  template <bool B, typename T, typename F>
  using conditional_t = typename conditional<B>::template type<T, F>;
}  // namespace brigand

/*!
 * \ingroup UtilitiesGroup
 * \brief Allows zero-cost unordered expansion of a parameter
 *
 * \details
 * Expands a parameter pack, typically useful for runtime evaluation via a
 * Callable such as a lambda, function, or function object. For example,
 * an unordered transform of a std::tuple can be implemented as:
 * \snippet Utilities/Test_TMPL.cpp expand_pack_example
 *
 * \see tuple_fold tuple_counted_fold tuple_transform std::tuple
 * EXPAND_PACK_LEFT_TO_RIGHT
 */
template <typename... Ts>
constexpr void expand_pack(Ts&&...) noexcept {}

/*!
 * \ingroup UtilitiesGroup
 * \brief Expand a parameter pack evaluating the terms from left to right.
 *
 * The parameter pack inside the argument to the macro must not be expanded
 * since the macro will do the expansion correctly for you. In the below example
 * a parameter pack of `std::integral_constant<size_t, I>` is passed to the
 * function. The closure `lambda` is used to sum up the values of all the `Ts`.
 * Note that the `Ts` passed to `EXPAND_PACK_LEFT_TO_RIGHT` is not expanded.
 *
 * \snippet Utilities/Test_TMPL.cpp expand_pack_left_to_right
 *
 * \see tuple_fold tuple_counted_fold tuple_transform std::tuple expand_pack
 */
#define EXPAND_PACK_LEFT_TO_RIGHT(...) \
  (void)std::initializer_list<char> { ((void)(__VA_ARGS__), '0')... }

/*!
 * \ingroup UtilitiesGroup
 * \brief Returns the first argument of a parameter pack
 */
template <typename T, typename... Ts>
constexpr decltype(auto) get_first_argument(T&& t, Ts&&... /*rest*/) noexcept {
  return t;
}

// namespace tmpl {
//  // Types list
//  template <class... T>
//  struct list {};
//
//  // Values list
//  template <typename T, T... Values>
//  using integral_list = list<std::integral_constant<T, Values>...>;
//
//  // Empty list
//  using empty_sequence = tmpl::list<>;
//
//  // Predefined placeholders
//  struct _1 {};
//  struct _2 {};
//
//  // Utility to expand a typelist into its variadic constituents
//  // and apply a functor/expression on the pack
//  namespace detail {
//    template <class A, template <class...> class B>
//    struct wrap;
//
//    template <template <class...> class A, class... T,
//              template <class...> class B>
//    struct wrap<A<T...>, B> {
//      using type = B<T...>;
//    };
//  }  // namespace detail
//
//  template <class A, template <class...> class B>
//  using wrap = typename detail::wrap<A, B>::type;
//
//  //============================================================================
//  //
//  // QUERYING TYPELISTS
//  //
//  //============================================================================
//
//  //****************************************************************************
//  // Whats the size of the typelist?
//  //****************************************************************************
//  template <class... T>
//  using count = std::integral_constant<std::size_t, sizeof...(T)>;
//
//  template <class TypeList>
//  using size = wrap<TypeList, count>;
//
//  //****************************************************************************
//  // Whats at the front of the current typelist?
//  //****************************************************************************
//  namespace detail {
//    template <class L>
//    struct front_impl;
//
//    template <template <class...> class L, class T, class... U>
//    struct front_impl<L<T, U...>> {
//      using type = T;
//    };
//  }  // namespace detail
//
//  template <class L>
//  using front = typename detail::front_impl<L>::type;
//
//  //============================================================================
//  //
//  // MANIPULATING TYPELISTS
//  //
//  //============================================================================
//
//  //****************************************************************************
//  // Push back a type on to a list
//  //****************************************************************************
//
//  namespace detail {
//    template <class TypeList,  // Type of the type list
//              class... T>      // Types to be appended
//    struct PushBackImpl;
//
//    template <template <class...> class TypeList,  // Type list specializaiton
//              class... U,                          // Types in the Typelist
//              class... T>  // Types to be added to the Typelist
//    struct PushBackImpl<TypeList<U...>, T...> {
//      using type = TypeList<U..., T...>;
//    };
//  }  // namespace detail
//
//  template <class TL, class... T>
//  using push_back = typename detail::PushBackImpl<TL, T...>::type;
//
//  //****************************************************************************
//  // Push front a type on to a list
//  //****************************************************************************
//
//  namespace detail {
//    template <class TypeList,  // Type of the type list
//              class... T>      // Types to be appended
//    struct PushFrontImpl;
//
//    template <template <class...> class TypeList,  // Type list specializaiton
//              class... U,                          // Types in the Typelist
//              class... T>  // Types to be added to the Typelist
//    struct PushFrontImpl<TypeList<U...>, T...> {
//      using type = TypeList<T..., U...>;
//    };
//  }  // namespace detail
//
//  template <class TL, class... T>
//  using push_front = typename detail::PushFrontImpl<TL, T...>::type;
//
//  //****************************************************************************
//  // Generating changes to a typelist
//  //****************************************************************************
//
//  // apply implementation
//  // only bare minimum : whats needed
//  namespace detail {
//
//    // Default case
//    // apply<int> for example gives a type of int
//    template <typename T, typename... Ls>
//    struct apply {
//      using type = T;
//    };
//
//    // So called lazy case
//    // Not sure why its lazy, and why these many levels of indirection is
//    needed
//    // Hard but fun to grok
//    template <template <typename...> class F, typename... Ts, typename L,
//              typename... Ls>
//    struct apply<F<Ts...>, L, Ls...>
//        : F<typename apply<Ts, L, Ls...>::type...> {};
//
//    // Provide a specializatino for call to _1 so that it unpacks it out
//    template <typename T, typename... Ts, typename... Ls>
//    struct apply<_1, list<T, Ts...>, Ls...> {
//      using type = T;
//    };
//
//    // Create an alias for comfort : no need for typenames and types
//    template <typename T, typename... Ts>
//    using bound_apply = typename apply<T, list<Ts...>>::type;
//  }  // namespace detail
//
//  //============================================================================
//  //
//  // SEARCHING TYPELISTS
//  //
//  //============================================================================
//
//  //****************************************************************************
//  // Is a type contained in a type list?
//  //****************************************************************************
//  namespace detail {
//    template <class TypeList,  // Type of the type lists
//              class T>         // Type to be searched
//    struct ContainsImpl;
//
//    template <template <class...> class TypeList,  // Type list specializaiton
//              class... U,                          // Types in the Typelist
//              class T>                             // Type to be searched
//    struct ContainsImpl<TypeList<U...>, T>
//        : public tt::detail::IsContained<T, U...> {};
//  }  // namespace detail
//
//  template <class TL, class T>
//  constexpr bool contains = detail::ContainsImpl<TL, T>::value;
//
//  //****************************************************************************
//  // Search for a type in a type list using some predicate
//  //****************************************************************************
//
//  namespace detail {
//
//    template <template <typename...> class S, template <typename...> class F,
//              typename... Ts>
//    struct finder {
//      template <typename T>
//      using P = F<Ts..., T>;
//
//      // not found case because none left
//      template <bool InNext8, bool Match, typename... Ls>
//      struct find {
//        using type = S<>;
//      };
//
//      // not found case because only one (L) left which does not satisfy the
//      // predicate
//      template <typename L>
//      struct find<true, false, L> {
//        using type = S<>;
//      };
//
//      // match base case : L matches
//      template <typename L, typename... Ls>
//      struct find<true, true, L, Ls...> {
//        using type = S<L, Ls...>;
//      };
//
//      // currently no match at L1, but potential for match in the next eight
//      // starting at L2
//      template <typename L1, typename L2, typename... Ls>
//      struct find<true, false, L1, L2, Ls...>
//          : find<true, F<Ts..., L2>::value, L2, Ls...> {};
//
//      // Entry point for non-detail cases, assumes the type list passed is
//      // <L8, LS ...>
//      template <typename L0, typename L1, typename L2, typename L3, typename
//      L4,
//                typename L5, typename L6, typename L7, typename L8,
//                typename... Ls>  // not match no longer fast track case
//      struct find<false, false, L0, L1, L2, L3, L4, L5, L6, L7, L8, Ls...>
//          : find<true, F<Ts..., L8>::value, L8, Ls...> {};
//    };
//
//    template <typename Sequence, typename Predicate>
//    struct find;
//
//    /* First order metafunctor */
//    template <template <typename...> class Sequence, typename... Ls, class
//    Pred> struct find<Sequence<Ls...>, Pred>
//        : finder<Sequence, bound_apply,
//                 Pred>::template find<false, false, void, void, void, void,
//                                      void, void, void, void, Ls...> {};
//
//    /* // Higher order metafunctor, not need ATM
//    template <template <typename...> class Sequence, typename... Ls,
//              template <typename...> class F>
//    struct find<Sequence<Ls...>, F<_1>>
//        : finder<Sequence, F>::template find<false, false, void, void, void,
//                                             void, void, void, void, void,
//                                             Ls...> {};
//                                             */
//
//  }  // namespace detail
//
//  template <typename Sequence, typename Predicate>
//  using find = typename detail::find<Sequence, Predicate>::type;
//
//  namespace detail {
//    template <typename Sequence, typename Predicate>
//    using find_size = ::tmpl::size<::tmpl::find<Sequence, Predicate>>;
//
//    template <typename Sequence, typename Predicate>
//    using empty_find =
//        cpp17::bool_constant<find_size<Sequence, Predicate>::value == 0>;
//
//    template <typename Sequence, typename Predicate>
//    using non_empty_find =
//        cpp17::bool_constant<find_size<Sequence, Predicate>::value != 0>;
//  }  // namespace detail
//
//  // Utility meta-function to check if something was found
//  template <typename Sequence, typename Predicate>
//  using found = typename detail::non_empty_find<Sequence, Predicate>;
//
//  //============================================================================
//  //
//  // GENERATING TYPELISTS
//  //
//  //============================================================================
//
//  //****************************************************************************
//  // Transform a list of types using a Functor
//  //****************************************************************************
//
//  namespace detail {
//
//    template <class Seq, class Func>
//    struct transform;
//
//    template <template <class...> class Seq, template <typename> class Func,
//              class... T>
//    struct transform<Seq<T...>, Func<_1>> {
//      using type = Seq<typename Func<T>::type...>;
//    };
//
//  }  // namespace detail
//
//  // Main transform entry point
//  template <typename Sequence1, typename Functor>
//  using transform = typename detail::transform<Sequence1, Functor>::type;
//
//  //============================================================================
//  //
//  // CONVERTING TYPELISTS TO OTHER TYPES
//  //
//  //============================================================================
//
//  template <typename... T>
//  using tuple_wrapper = typename std::tuple<T...>;
//
//  template <typename L>
//  using as_tuple = wrap<L, tuple_wrapper>;
//
//  //============================================================================
//  //
//  // RUNTIME PROCESSING OF TYPELISTS
//  //
//  //============================================================================
//  namespace detail {
//    template <class F, class... Ts>
//    F for_each_args(F f, Ts&&... a) {
//      return (void)std::initializer_list<int>{
//                 ((void)std::ref(f)(static_cast<Ts&&>(a)), 0)...},
//             f;
//    }
//
//    template <template <class...> class List, typename... Elements,
//              typename Functor>
//    Functor for_each_impl(List<Elements...>&&, Functor f) {
//      return for_each_args(f, Elements()...);
//    }
//  }  // namespace detail
//
//  template <typename List, typename Functor>
//  Functor for_each(Functor f) {
//    return detail::for_each_impl(List{}, f);
//  }
//
//}  // namespace tmpl

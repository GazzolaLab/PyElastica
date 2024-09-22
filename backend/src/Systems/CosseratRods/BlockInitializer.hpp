#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <utility>  // for declval

///
#include "Systems/CosseratRods/Types.hpp"
///
#include "Systems/Block.hpp"
#include "Systems/CosseratRods/Components/Tags.hpp"
#include "Systems/CosseratRods/CosseratRodPlugin.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/Invoke.hpp"
#include "Utilities/TypeTraits/IsCallable.hpp"

namespace elastica {

  namespace cosserat_rod {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Declaration of CosseratInitializer
     *
     * \tparam Plugin       The CosseratRod plugin class
     * \tparam Initializers Initializers, formed using blocks::initialize()
     */
    template <class /*Plugin*/, class... /*Initializers*/>
    struct CosseratInitializer;
    /*! \endcond */
    //**************************************************************************

    //**************************************************************************
    /*!\brief A customized initializer for a Block templated on a
     * CosseratRodPlugin
     * \ingroup cosserat_rod
     *
     * \tparam CRT A valid Cosserat Rod Traits class
     * \tparam ComputationalBlock A template type modeling the
     * `ComputationalBlock` concept
     * \tparam Components Variadic components for customizing behavior
     * \tparam Initializers Initializers, formed using blocks::initialize()
     *
     * \see BlockInitializer
     */
    template <typename CRT, template <typename> class ComputationalBlock,
              template <typename /*CRT*/, typename /* ComputationalBlock */>
              class... Components,
              typename... Initializers>
    struct CosseratInitializer<
        CosseratRodPlugin<CRT, ComputationalBlock, Components...>,
        Initializers...>
        : public ::blocks::BlockInitializer<
              CosseratRodPlugin<CRT, ComputationalBlock, Components...>,
              Initializers...> {
      //**Type definitions******************************************************
      //! Plugin type
      using Plugin = CosseratRodPlugin<CRT, ComputationalBlock, Components...>;
      //! Parent type
      using Parent = ::blocks::BlockInitializer<Plugin, Initializers...>;
      //! size type
      using size_type = typename CRT::size_type;
      //************************************************************************

     public:
      //! Inherited constructors from parent
      using Parent::Parent;

     private:
      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Implementation to get the number of elements for a callable type
       */
      template <typename F, Requires<::tt::is_callable<F>::value> = nullptr>
      inline constexpr auto n_elems_impl(F const& f) const noexcept
          -> size_type {
        return f();
      }
      /*! \endcond */
      //************************************************************************

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Implementation to get the number of elements for a non-callable
       * type
       */
      template <typename F, Requires<not ::tt::is_callable<F>::value> = nullptr>
      inline constexpr auto n_elems_impl(F const& f) const noexcept
          -> size_type {
        return f;
      }
      /*! \endcond */
      //************************************************************************

     public:
      //************************************************************************
      /*!\brief Gets the number of elements to initialize the current rod with
       */
      inline decltype(auto) n_elems() const noexcept {
        return n_elems_impl(blocks::get<::elastica::tags::NElement>(
            static_cast<Parent const&&>(*this)));
      }
      //************************************************************************
    };
    //**************************************************************************

    namespace detail {

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Helper to rebind a template from BlockInitializer to
       * CosseratInitializer
       */
      template <typename>
      struct Rebind;
      /*! \endcond */
      //************************************************************************

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Specialization of Rebind for CosseratPlugin
       */
      template <typename CRT,
                template <typename>
                class ComputationalBlock,  // Block with implementation
                template <typename /*CRT*/, typename /* ComputationalBlock */>
                class... Components,
                class... InitializerVariables>
      struct Rebind<blocks::BlockInitializer<
          CosseratRodPlugin<CRT, ComputationalBlock, Components...>,
          InitializerVariables...>> {
        //! rebinded type
        using type = CosseratInitializer<
            CosseratRodPlugin<CRT, ComputationalBlock, Components...>,
            InitializerVariables...>;
      };
      /*! \endcond */
      //************************************************************************

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Specialization of Rebind for TaggedCosseratPlugin
       */
      template <typename CRT,
                template <typename>
                class ComputationalBlock,  // Block with implementation
                typename Tag,              // Tag for TaggedPlugin
                template <typename /*CRT*/, typename /* ComputationalBlock */>
                class... Components,
                class... InitializerVariables>
      struct Rebind<blocks::BlockInitializer<
          TaggedCosseratRodPlugin<CRT, ComputationalBlock, Tag, Components...>,
          InitializerVariables...>>
          : public Rebind<blocks::BlockInitializer<
                CosseratRodPlugin<CRT, ComputationalBlock, Components...>,
                InitializerVariables...>> {};
      /*! \endcond */
      //************************************************************************

    }  // namespace detail

    //**************************************************************************
    /*!\brief Make a CosseratRodInitializer for some `Plugin` that corresponds
     * to a CosseratRod
     * \ingroup cosserat_rod
     *
     * \details
     * This is equivalent to blocks::initialize_block(), but specialized for
     * CosseratRodPlugin, for more details:
     *
     * \see blocks::initialize_block()
     */
    template <typename AnyCosseratPlugin, typename... NamedInitializers>
    constexpr inline decltype(auto) initialize_cosserat_rod(
        NamedInitializers&&... initializers) noexcept {
      using ReturnType = typename detail::Rebind<
          decltype(blocks::initialize_block<AnyCosseratPlugin>(
              std::declval<NamedInitializers>()...))>::type;
      return ReturnType{std::forward<NamedInitializers>(initializers).get()...};
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Make a CosseratRodInitializer for some `Plugin` that corresponds
     * to a CosseratRod
     * \ingroup cosserat_rod
     *
     * \details
     * This is equivalent to blocks::initialize_block(), but specialized for
     * CosseratRodPlugin, for more details:
     *
     * \see blocks::initialize_block()
     */
    template <typename AnyCosseratPlugin, typename... NamedInitializers>
    constexpr inline decltype(auto) initialize_cosserat_rod(
        std::tuple<NamedInitializers...> initializer_tuple) noexcept {
      return cpp17::apply(
          [](auto&& ...args){
            return initialize_cosserat_rod<AnyCosseratPlugin>(std::forward<decltype(args)>(args)...);
          },
          std::move(initializer_tuple));
    }
    //**************************************************************************

    //==========================================================================
    //
    //  FREE FUNCTIONS
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Returns the size of a (yet to be) Cosserat Rod from the
     * initializer \ingroup cosserat_rod
     *
     * \param initializer The current cosserat rod initializer
     * \return The  number of elements in the cosserat rod.
     */
    template <class Plugin, class... Initializers>
    inline constexpr auto size(
        CosseratInitializer<Plugin, Initializers...> const&
            initializer) noexcept {
      return initializer.n_elems();
    }
    //****************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

//
//  // We don't have a piecewise constructor for std::tuple
//  // and hence as an alternative, we use this class
//  template <typename CRT>  // Cosserat rod traits
//  struct CosseratBlockInitializer {
//    // Nested struct to have limited instantiations for different
//    // rod traits
//    template <typename... Initializers>
//    struct Initializer {
//      using InitializerType = std::tuple<Initializers...>;
//      std::size_t n_elems;
//      InitializerType initializers;
//
//      // logic which was in traits class now gets shifted here
//      template <typename PropertyTag>
//      constexpr decltype(auto) get() noexcept {
//        // Here initializers can be a ref too (in the case its not an
//        // immediate lambda, hence we first remove reference here
//
//        // TODO Is there no alternative than doing an O(n) search here?
//        using find_result = tmpl::find<
//            // find
//            InitializerType,
//            // condition
//            std::is_same<
//                // first
//                tmpl::pin<
//                    typename CRT::template
//                    initializer_tag_t<PropertyTag>>,
//                // second
//                tmpl::bind<CRT::template parameter_type_t,
//                           tmpl::_1>>  // condition
//            >;                         // find
//
//        static_assert(tmpl::size<find_result>::value,
//                      "Valid initializer not found!");
//
//        using PropertyTagInitializer = tmpl::front<find_result>;
//
//        // The trailing get is to return the generator from the enclosing
//        // strong type
//        return std::get<PropertyTagInitializer>(initializers).get();
//      }
//    };
//  };
//
//  namespace initializers {
//
//    template <typename PropertyTag,
//              typename CRT,  // Cosserat Rod Traits
//              typename... Initializers>
//    inline constexpr decltype(auto) get(
//        typename CosseratBlockInitializer<CRT>::template Initializer<
//            Initializers...>& initializers) noexcept {
//      return initializers.template get<PropertyTag>();
//    }
//
//    template <typename PropertyTag,
//              typename CRT,  // Cosserat Rod Traits
//              typename... Initializers>
//    inline constexpr decltype(auto) get(
//        typename CosseratBlockInitializer<CRT>::template Initializer<
//            Initializers...>&& initializers) noexcept {
//      return initializers.template get<PropertyTag>();
//    }
//
//  }  // namespace initializers
/*
template <typename PropertyTag,
          typename CRT,  // Cosserat Rod Traits
          typename... Initializers>
inline constexpr decltype(auto) get(
    typename CosseratBlockInitializer<CRT>::template Initializer<
        Initializers...> const& initializers) noexcept {
  return initializers.template get<PropertyTag>();
}
*/

//    // Ease in syntax compared to forward_as_tuple?
//    template <typename CRT =
//    ::elastica::detail_cosserat_rod::CosseratRodTraits,
//              typename... Initializers>
//    constexpr inline decltype(auto) make_cosserat_rod(
//        std::size_t n_elems, Initializers&&... initializers) noexcept {
//      // for temporary initializers, Initializers get deduced as T&&, so
//      we
//      // remove cv-rvalue reference for reference initializers,
//      Initializers get
//      // deduced as T&, so we need to remove cv-lvalue reference here this
//      is
//      // covered by decay (including array operations, which we really
//      should
//      // not be using)
//      using ReturnType = typename CosseratBlockInitializer<
//          CRT>::template Initializer<std::decay_t<Initializers>...>;
//      return ReturnType{n_elems, std::forward_as_tuple(initializers...)};
//    }

//  namespace detail {
//
//    //**Type
//    definitions********************************************************
//    /*! \cond ELASTICA_INTERNAL */
//    //! Strongly typed intiializer
//    template <typename ParameterTag>
//    struct Initializer {
//      template <typename Func>
//      using type =
//          named_type::NamedType<Func, ParameterTag, named_type::Callable>;
//    };
//    /*! \endcond */
//    //**************************************************************************
//  }  // namespace detail

//  template <typename Tag, typename F, typename... Funcs>
//  constexpr inline decltype(auto) initialize(F&& f, Funcs&&... funcs)
//  noexcept(
//      noexcept(::elastica::make_detail::make_named_functor<
//               Initializer<Tag>::template type>(
//          std::forward<F>(f), std::forward<Funcs>(funcs)...))) {
//    return ::elastica::make_detail::make_named_functor<
//        Initializer<Tag>::template type>(std::forward<F>(f),
//                                         std::forward<Funcs>(funcs)...);
//  };

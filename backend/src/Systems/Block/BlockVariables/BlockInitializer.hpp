#pragma once

//******************************************************************************
// Includes
//******************************************************************************

// blocks
#include "Systems/Block/Block/Aliases.hpp"
#include "Systems/Block/Block/Block.hpp"
//
#include "Systems/Block/BlockVariables/Aliases.hpp"
#include "Systems/Block/BlockVariables/Protocols.hpp"
#include "Systems/Block/BlockVariables/TypeTraits.hpp"
#include "Systems/Block/BlockVariables/Types.hpp"
//
#include "Utilities/IgnoreUnused.hpp"
#include "Utilities/MakeNamedFunctor.hpp"
#include "Utilities/NamedType.hpp"
#include "Utilities/NonCopyable.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/StaticWarning.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TMPLDebugging.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace blocks {

  namespace detail {

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Implementation of compile-time error feedback to the user in case
     * initializers are missing.
     * \ingroup blocks
     *
     * \tparam L list of set difference between expected and given initializers
     */
    template <typename L>
    struct ErrorFeedback {
      TypeDisplayer<L>
          these_tags_are_needed_for_initialization_but_are_not_provided_please_provide_them{};
    };
    //**************************************************************************

    template <bool B>
    struct these_tags_are_not_needed_by_the_block_but_provided {};

    template <>
    struct [[deprecated(R"error(
Please see the list of tags not needed for initialization in the instantiation of template class blocks::detail::WarningFeedback
)error")]] these_tags_are_not_needed_by_the_block_but_provided<false>{};

    //**************************************************************************
    /*!\brief Implementation of compile-time warning feedback to the user in
     * case additional, unneeded initializers are provided. \ingroup blocks
     *
     * \tparam L list of set difference between expected and given initializers
     */
    template <typename L>
    struct WarningFeedback {
      static constexpr bool value = (tmpl::size<L>::value == 0);

// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=84347
#if defined(__clang__)
      using Feedback IGNORE_UNUSED = decltype(
          these_tags_are_not_needed_by_the_block_but_provided<value>{});
#else
      using Feedback IGNORE_UNUSED = decltype(STATIC_WARNING(
          value, "These tags are not needed by the block but provided!"));
#endif
    };
    /*! \endcond */
    //**************************************************************************

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Compile-time feedback to the user in case the provided
     * initializers are additional/missing.
     * \ingroup blocks
     *
     * \tparam Expected expected type list of initializers
     * \tparam Given given type list of initializers by the user in client code
     */
    template <typename Expected, typename Given>
    struct ProvideFeedback {
      /*
      // this is not the perfect recipe as for example
      // Expected : list<A, B, C>
      // Given : list<A, B, D, E>
      // so value = false, so only a warning is generated
      static constexpr bool value =
          (tmpl::size<Expected>::value > tmpl::size<Given>::value);

      // Rather if we do expected - given we get
      // result = list<C>, which always triggers an error response
      */
      static constexpr bool value = static_cast<bool>(
          tmpl::size<tmpl::list_difference<Expected, Given>>::value);

      // decltypes are necessary to force instantiation rather than the
      // compiler smartly optimizing it away
      using Feedback IGNORE_UNUSED =
          decltype(tmpl::conditional_t<
                   value, ErrorFeedback<tmpl::list_difference<Expected, Given>>,
                   WarningFeedback<tmpl::list_difference<Given, Expected>>>{});
    };
    /*! \endcond */
    //**************************************************************************

  }  // namespace detail

  // global tags -> initializer tags
  namespace detail {
    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Marks a variable in the blocks initializer infrastructure
    // \ingroup blocks
    //
    // \details
    // BlockInitializerVariable models the concept of a Variable in the
    // initialization code of blocks module. While the needed functionality (
    // i.e. initializing the data pointed to by a tag) can be achieved in
    // different ways, this approach of marking a BlockInitializerVariable (with
    // a unique Parameter) inside a BlockInitializer provides parallels with
    // a BlockVariable (marked with the same unique Parameter) inside a Block.
    //
    //
    // \tparam ParameterTag  A unique type parameter to mark the Variable with
    // \tparam NamedFunction Type for customizing the data implementation
    */
    template <typename ParameterTag, typename NamedFunction>
    struct BlockInitializerVariable : public tt::ConformsTo<protocols::Variable>
    /*, public ParameterTag,*/  // Don't inherit to support tags that are only
                                // declared, but not defined
    {
      //**Type definitions******************************************************
      //! This type
      using This = BlockInitializerVariable<ParameterTag, NamedFunction>;
      //! UUID Parameter type
      using Parameter = ParameterTag;
      //! Data to be stored, based on requirements from TaggedTuple
      using type = NamedFunction;
      //************************************************************************
    };
    /*! \endcond */
    //**************************************************************************

  }  // namespace detail

  //****************************************************************************
  /*!\brief Data-structure for initializing a Block
   * \ingroup blocks
   *
   * \details
   * Initializing the variables in a blocks::Block is eased by using this
   * BlockInitializer. BlockInitializer is not directly used, but is indirectly
   * used by @ref blocks while using blocks::initialize() and
   * blocks::initialize_block(). Please refer to the documentation of these
   * functions instead.
   *
   * Note that if you are implementing customized `Plugin` types, it is useful
   * to extend this struct by inheritance for customizing behavior if needed,
   * while using the already existing infrastructure in @ref blocks. For one
   * such example, the elastica::cosserat_rod::CosseratInitializer.
   *
   * \tparam Plugin The plugin to initialize
   * \tparam InitializerVariables Initializers, formed using
   * blocks::initialize()
   *
   * \see blocks::initialize(), blocks::initialize_block(), blocks::Block,
   * blocks::BlockSlice
   */
  template <typename Plugin,
            typename... InitializerVariables>  // Cosserat rod traits
  struct BlockInitializer                      // <Plugin,
      : private ::elastica::NonCopyable {
    //**Type definitions********************************************************
    //! Provided BlockInitializerVariable
    using Leafs = tmpl::list<InitializerVariables...>;
    //! Conformant mapping between tags and variables
    using VariableMap = VariableMapping<Leafs>;
    //! Values of all variable initializers
    using Initializers = tuples::TaggedTuple<InitializerVariables...>;
    //**************************************************************************

    //**************************************************************************
    /*!\name Feedback type definitions */
    //@{
    //! Initialized BlockVariable(s) marked by the Block
    //\note This is guaranteed to exist in the Block by using the Facade
    using InitializedVariables = initialized_variables_t<Block<Plugin>>;
    //! Tags expected by the Initialized Variables
    using ExpectedInitializerTags =
        tmpl::transform<InitializedVariables,
                        tmpl::bind<initializer_t, tmpl::_1>>;
    //! Tags given to the current initializer
    using GivenInitializerTags =
        tmpl::transform<Leafs, tmpl::bind<parameter_t, tmpl::_1>>;
    //! Feedback to the user based on these tags above
    using FeedbackToUser IGNORE_UNUSED =
        decltype(detail::ProvideFeedback<ExpectedInitializerTags,
                                         GivenInitializerTags>{});
    //@}
    //**************************************************************************

    //**Constructors************************************************************
    /*!\name Constructors */
    //@{

    //**************************************************************************
    /*!\brief The default constructor.
     *
     * \param funcs Callables to be passed to the underlying initializer
     * data
     */
    template <typename... Funcs>
    explicit BlockInitializer(Funcs&&... funcs) noexcept(
        noexcept(Initializers(std::forward<Funcs>(funcs)...)))
        : initializers_(std::forward<Funcs>(funcs)...) {}
    //**************************************************************************

    //**************************************************************************
    /*!\brief The move constructor.
     *
     */
    BlockInitializer(BlockInitializer&& other) noexcept
        : initializers_(std::move(other.initializers_)){};
    //**************************************************************************

    //@}
    //**************************************************************************

    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~BlockInitializer() = default;
    //@}
    //**************************************************************************

    //**Data access*************************************************************
    /*!\name Data access */
    //@{

    //**************************************************************************
    /*!\brief Access to the underlying data
    //
    // \return Mutable lvalue reference to the underlying data
    */
    inline constexpr Initializers& data() & noexcept { return initializers_; }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to the underlying data
    //
    // \return Constant lvalue reference to the underlying data
    */
    inline constexpr Initializers const& data() const& noexcept {
      return initializers_;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to the underlying data
    //
    // \return Mutable rvalue reference to the underlying data
    */
    inline constexpr Initializers&& data() && noexcept {
      return static_cast<Initializers&&>(initializers_);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to the underlying data
    //
    // \return Const rvalue reference to the underlying data
    */
    inline constexpr Initializers const&& data() const&& noexcept {
      return static_cast<Initializers const&&>(initializers_);
    }
    //**************************************************************************

    //@}
    //**************************************************************************

    //**Member variables********************************************************
    /*!\name Member variables */
    //@{
    //! All block variable initializers
    Initializers initializers_;
    //@}
    //**************************************************************************
  };
  //****************************************************************************

  /*
  // get a tuple from initializer
  template <typename Tag, typename Plugin, typename TupleLike>
  inline constexpr decltype(auto) get(
      BlockInitializer<Plugin, TupleLike> const& block_initializer) noexcept {
    using Block_t = BlockInitializer<Plugin, TupleLike>;
    // here we need to make sure that all blocks give a templated type trait
    // to recover block variables from a given "outside" Tag
    using BlockVariableTag = typename Block_t::template var_from<Tag>;
    return tuples::get<BlockVariableTag>(block_initializer.data());
  }

  template <typename Tag, typename Plugin, typename TupleLike>
  inline constexpr decltype(auto) get(
      BlockInitializer<Plugin, TupleLike>& block_initializer) noexcept {
    using Block_t = BlockInitializer<Plugin, TupleLike>;
    // here we need to make sure that all blocks give a templated type trait
    // to recover block variables from a given "outside" Tag
    using BlockVariableTag = typename Block_t::template var_from<Tag>;
    return tuples::get<BlockVariableTag>(block_initializer.data());
  }
  */

  //****************************************************************************
  /* We declare own get function, rather than using the global get, because:
   * BlockInitializer doesn't model block concept (no actionable data members)
   * and hence ModelsBlockConcept<BlockInitializer> is false, which raise a
   * static_assert in the global get function
   */
  //****************************************************************************

  //****************************************************************************
  /*!\brief Extract element from a BlockInitializer
   * \ingroup blocks
   *
   * \details
   * Extracts the element of the BlockInitializer `block_initializer` whose Tag
   * type is `Tag`. Fails to compile unless the initializer has the `Tag` being
   * extracted.
   *
   * \usage
   * The usage is similar to std::get(), shown below
   * \code
     BlockInitializer<...> b;
     auto my_tag_initializer = blocks::get<tags::MyTag>(std::move(b));
   * \endcode
   *
   * \tparam Tag Tag to extract
   * \param block_initializer A BlockInitializer type
   */
  template <typename Tag, typename Plugin, typename... InitializerVariables>
  inline constexpr decltype(auto) get(
      BlockInitializer<Plugin, InitializerVariables...>&
          block_initializer) noexcept {
    using Block_t = BlockInitializer<Plugin, InitializerVariables...>;
    // here we need to make sure that all blocks give a templated type trait
    // to recover block variables from a given "outside" Tag
    using BlockVariableTag =
        typename Block_t::VariableMap::template variable_from<Tag>;
    //    using BlockVariableTag = typename Block_t::template var_from<Tag>;
    return tuples::get<BlockVariableTag>(block_initializer.data());
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Extract element from a BlockInitializer
   * \ingroup blocks
   *
   * \details
   * Extracts the element of the BlockInitializer `block_initializer` whose Tag
   * type is `Tag`. Fails to compile unless the initializer has the `Tag` being
   * extracted.
   *
   * \usage
   * The usage is similar to std::get(), shown below
   * \code
     BlockInitializer<...> b;
     auto my_tag_initializer = blocks::get<tags::MyTag>(std::move(b));
   * \endcode
   *
   * \tparam Tag Tag to extract
   * \param block_initializer A BlockInitializer type
   */
  template <typename Tag, typename Plugin, typename... InitializerVariables>
  inline constexpr decltype(auto) get(
      BlockInitializer<Plugin, InitializerVariables...> const&
          block_initializer) noexcept {
    using Block_t = BlockInitializer<Plugin, InitializerVariables...>;
    // here we need to make sure that all blocks give a templated type trait
    // to recover block variables from a given "outside" Tag
    using BlockVariableTag =
        typename Block_t::VariableMap::template variable_from<Tag>;
    //    using BlockVariableTag = typename Block_t::template var_from<Tag>;
    return tuples::get<BlockVariableTag>(block_initializer.data());
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Extract element from a BlockInitializer
   * \ingroup blocks
   *
   * \details
   * Extracts the element of the BlockInitializer `block_initializer` whose Tag
   * type is `Tag`. Fails to compile unless the initializer has the `Tag` being
   * extracted.
   *
   * \usage
   * The usage is similar to std::get(), shown below
   * \code
     BlockInitializer<...> b;
     auto my_tag_initializer = blocks::get<tags::MyTag>(std::move(b));
   * \endcode
   *
   * \tparam Tag Tag to extract
   * \param block_initializer A BlockInitializer type
   */
  template <typename Tag, typename Plugin, typename... InitializerVariables>
  inline constexpr decltype(auto) get(
      BlockInitializer<Plugin, InitializerVariables...>&&
          block_initializer) noexcept {
    using Block_t = BlockInitializer<Plugin, InitializerVariables...>;
    // here we need to make sure that all blocks give a templated type trait
    // to recover block variables from a given "outside" Tag
    using BlockVariableTag =
        typename Block_t::VariableMap::template variable_from<Tag>;
    //    using BlockVariableTag = typename Block_t::template var_from<Tag>;
    return tuples::get<BlockVariableTag>(
        static_cast<Block_t&&>(block_initializer).data());
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Extract element from a BlockInitializer
   * \ingroup blocks
   *
   * \details
   * Extracts the element of the BlockInitializer `block_initializer` whose Tag
   * type is `Tag`. Fails to compile unless the initializer has the `Tag` being
   * extracted.
   *
   * \usage
   * The usage is similar to std::get(), shown below
   * \code
     const BlockInitializer<...> b;
     auto my_tag_initializer = blocks::get<tags::MyTag>(std::move(b));
   * \endcode
   *
   * \tparam Tag Tag to extract
   *
   * \param block_initializer A BlockInitializer type
   */
  template <typename Tag, typename Plugin, typename... InitializerVariables>
  inline constexpr decltype(auto) get(
      BlockInitializer<Plugin, InitializerVariables...> const&&
          block_initializer) noexcept {
    using Block_t = BlockInitializer<Plugin, InitializerVariables...>;
    // here we need to make sure that all blocks give a templated type trait
    // to recover block variables from a given "outside" Tag
    using BlockVariableTag =
        typename Block_t::VariableMap::template variable_from<Tag>;
    // using BlockVariableTag = typename Block_t::template var_from<Tag>;
    return tuples::get<BlockVariableTag>(
        static_cast<Block_t const&&>(block_initializer).data());
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Make a BlockInitializer for some `Plugin`
   * \ingroup blocks
   *
   * \details
   * To initialize members corresponding to a particular Tag within a
   * blocks::Block (or blocks::BlockSlice), we use blocks::initialize(). To
   * coordinate all the different Tag initializers expected by the Plugin of a
   * blocks::Block, it is convenient to use this initialize_block().
   *
   * \example
   * \snippet Test_BlocksFramework.cpp initialize_block_example
   *
   * \tparam Plugin Plugin type for the BlockInitializer
   *
   * \param initializers Strongly typed initializers, formed using
   * blocks::initialize()
   *
   * \see BlockInitializer, blocks::initialize()
   */
  template <typename Plugin, typename... NamedInitializers>
  constexpr inline decltype(auto) initialize_block(
      NamedInitializers&&... initializers) noexcept {
    using ReturnType = BlockInitializer<
        Plugin,
        detail::BlockInitializerVariable<
            named_type::tag_type<std::decay_t<NamedInitializers>>,
            named_type::underlying_type<std::decay_t<NamedInitializers>>>...>;
    return ReturnType{std::forward<NamedInitializers>(initializers).get()...};
  }
  //****************************************************************************

  namespace detail {

    //**Type definitions********************************************************
    /*! \cond ELASTICA_INTERNAL */
    //! Strongly typed initializer for blocks
    template <typename ParameterTag>
    struct Initializer {
      template <typename Func>
      using type =
          named_type::NamedType<Func, ParameterTag, named_type::Gettable>;
    };
    /*! \endcond */
    //**************************************************************************

  }  // namespace detail

  //****************************************************************************
  /*!\brief Initialize a member of a particular `Tag` within a Block
   * \ingroup blocks
   *
   * \details
   * To initialize members corresponding to a particular `Tag` within a
   * blocks::Block(or blocks::BlockSlice), we can use blocks::initialize().
   * To achieve this initialization, we resort to passing in function objects
   * (such as lambdas) as arguments to this function. The passed in objects are
   * then marked with the `Tag` type using the template parameter. This is used
   * usually in conjunction with blocks::initialize_blocks() to initialize many
   * tags simultaneously
   *
   * One pattern we use is to make the passed-in lambdas typically take an
   * index parameter & returns the appropriate value for initialization. This
   * gives tremendous flexibility in initializing blocks. To offer additional
   * flexibility, one can pass in more than one functor, inside initialize().
   * The Block then considers all these functors as one big overloaded functor
   * and initializes the member with the best possible candidate. This
   * follows the typical C++ overload resolution rules and hence there is no
   * ambiguity in resolution. One can customize these overload rules in the
   * Block, if the need arises.
   *
   * \example
   * \snippet Test_BlocksFramework.cpp initialize_block_example
   *
   * As an example of passing in multiple parameters in the same example, lets
   * consider the scenario where the positions and velocities can also be
   * initialized based on a double parameter (such as the arclength of a rod).
   * We achieve this in code by
   *
   * \code
      using Plugin = AdvancedExamplePlugin;
      using MyBlock = blocks::Block<Plugin>;
      MyBlock block(blocks::initialize_block<Plugin>(
          blocks::initialize<tags::ExampleInitializedPosition>(
              [](std::size_t index) -> int {
                  return 1 + static_cast<int>(index);
              },
              [](double centerline_coordinate) -> int {
                  return 1 + static_cast<int>(5.0 * centerline_coordinate);
              }),
          blocks::initialize<tags::ExampleInitializedVelocity>(
              [](std::size_t index) -> float {
                  return static_cast<float>(index) + 5.5F;
              },
              [](double centerline_coordinate) -> float {
                  return 5.5F + static_cast<float>(5.0 * centerline_coordinate);
              })));
     \endcode
   *
   * \tparam Tag Tag type whose corresponding data member will be initialized
   *
   * \param f     A functor object to initialize the data member
   * \param funcs Other functor objects to initialize the same data member
   */
  template <typename Tag, typename F, typename... Funcs>
  constexpr inline decltype(auto)
  initialize(F&& f, Funcs&&... funcs) noexcept(noexcept(
      ::named_type::make_named_functor<detail::Initializer<Tag>::template type>(
          std::forward<F>(f), std::forward<Funcs>(funcs)...))) {
    return ::named_type::make_named_functor<
        detail::Initializer<Tag>::template type>(std::forward<F>(f),
                                                 std::forward<Funcs>(funcs)...);
  }
  //****************************************************************************

}  // namespace blocks

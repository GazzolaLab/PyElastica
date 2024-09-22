#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/Block/Block/Types.hpp"
//
#include "Systems/Block/Block/Concepts.hpp"
//
#include "Systems/Block/Block/Aliases.hpp"
#include "Systems/Block/Block/AsVariables.hpp"
#include "Systems/Block/BlockVariables/TypeTraits.hpp"
//
#include "Utilities/CRTP.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits/Cpp17.hpp"
#include "Utilities/TypeTraits/IsDetected.hpp"
//
#include <type_traits>

namespace blocks {

  namespace detail {

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Fallback for when `Parent` does not require InitializedVariables
     * in a BlockFacade
     * \ingroup blocks
     */
    struct EmptyInitializedVariables {
      using InitializedVariables = tmpl::list<>;
    };
    /*! \endcond */
    //**************************************************************************

    //************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Signature of expected a derived Block
     * \ingroup blocks
     *
     * Very similar to protocols, but we don't need to derive from the
     * protocol class. Rather, signatures rely on an implicit interface
     * guarantee.
     *
     * \tparam BlockImplementation The block that derives from BlockFacade
     */
    template <typename BlockImplementation>
    struct Signatures {
      /// [slice_signature]
      template <typename Tag>
      using slice_signature =
          decltype(std::declval<BlockImplementation&>().template slice<Tag>(
              std::declval<std::size_t>()));
      // This is equivalent to
      //   template <typename Tag>
      //   // some_return_type
      //   slice(std::size_t index) &
      //   {
      //     // return the slice at the location index
      //   }
      /// [slice_signature]

      /// [const_slice_signature]
      template <typename Tag>
      using const_slice_signature =
          decltype(std::declval<BlockImplementation const&>()
                       .template slice<Tag>(std::declval<std::size_t>()));
      // This is equivalent to
      //   template <typename Tag>
      //   // some_return_type
      //   slice(std::size_t index) const &
      //   {
      //     // return the slice at the location index
      //   }
      /// [const_slice_signature]
    };
    /*! \endcond */
    //************************************************************************

  }  // namespace detail

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief A helper facade for ease in implementing new Block types
   * \ingroup blocks
   *
   * \details
   * The BlockFacade is a helper template to assist in conforming to the
   * protocols expected by a Block class, when specializing it for
   * particular Plugin types. It provides the interface of a block in terms of a
   * few core functions and associated types, which are to be supplied by a
   * Block that derives from BlockFacade. Internally, BlockFacade uses the CRTP
   * pattern to access the core functions supplied by the Block.
   *
   * BlockFacade is modeled after boost::iterator_facade. Notably, it differs
   * from boost::iterator_facade in one key aspect : the BlockFacade is
   * internally templated on blocks::Block---hence its use for anything
   * other then implementing a specialization of blocks::Block template is
   * ill-formed by definition.
   *
   * \usage
   * While customizing a blocks::block for some type `CustomPlugin` (which needs
   * to meet the expected interface for a valid `Plugin`, see Blocks), we can
   * use BlockFacade to minimize the code we need to write, as shown below:
   * \snippet Test_BlockFacade.cpp block_facade_usage
   * For correct use with a BlockFacade, the following members in the deriving
   * Block need to be defined.
   * \snippet this slice_signature
   * \snippet this const_slice_signature
   *
   * where
   * - `Tag` is the tag to be retrieved
   *
   * \example
   * For a simple custom Plugin as shown below,
   * \snippet Test_BlocksFramework.cpp int_plugin
   * Without a BlockFacade, one needs to write out
   * \snippet Test_BlocksFramework.cpp int_block
   * With a BlockFacade however, this process becomes more simpler
   * \snippet Test_BlocksFramework.cpp fac_int_block
   *
   * Additionally for more practical, operate-intensive Plugins in \elastica,
   * such as Cosserat rods, BlockFacade takes care of setting up the necessary
   * type definitions for slicing/initialization among others, including setting
   * up subscript operators etc given the Block has the necessary functions
   *
   * \tparam Plugin The computational plugin modeling a Lagrangian entity
   *
   * \see Block
   */
  template <typename Plugin>
  class BlockFacade : public elastica::CRTPHelper<Block<Plugin>, BlockFacade>,
                      public Gettable<Block<Plugin>>,
                      public Spannable<Block<Plugin>>,
                      public Plugin {
   protected:
    //**Type definitions********************************************************
    /*! \cond ELASTICA_INTERNAL */
    //! Tag marking need for the block to be initialized
    using to_be_initialized_tag = std::true_type;
    //! Tag marking no need for block to be initialized
    using need_not_be_initialized_tag = std::false_type;
    /*! \endcond */
    //**************************************************************************

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Compile-time check for initialized variables in Plugin
     *
     * \details
     * If the `Plugin` class has no initialized_variables_t alias,
     * meaning that there are no variables to initialized, we can simply skip
     * initialization, and the `Plugin` doesn't need to initialize these members
     * and initialization can be skipped. This check for skipping can happen in
     * compile-time and is facilitated by this function.
     *
     * \metareturns cpp17::bool_constant
     */
    static inline constexpr auto parent_to_be_initialized() noexcept {
      // This one-liner doesn't work if the initialized_variables_t is protected
      // which is the intended use case
      // return ::tt::is_detected<blocks::initialized_variables_t, Plugin>{};
      return ::tt::is_detected<blocks::initialized_variables_t, Plugin>{};
    }
    /*! \endcond */
    //**************************************************************************

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Compile-time check for initialized variables in Plugin
     *
     * \metareturns cpp17::bool_constant
     *
     * \see parent_to_be_initialized()
     */
    static inline constexpr auto parent_need_not_be_initialized() noexcept {
      return cpp17::negation<decltype(parent_to_be_initialized())>{};
    }
    /*! \endcond */
    //**************************************************************************

   private:
    //**Type definitions********************************************************
    //! Parent type
    using Parent = Plugin;
    //! This type
    using This = BlockFacade<Parent>;
    //**************************************************************************

   public:
    //**Type definitions********************************************************
    //! Plugin type
    using PluginType = Parent;
    //! List of variables
    using typename Parent::Variables;
    //! List of initialized variables
    using InitializedVariables = blocks::initialized_variables_t<
        std::conditional_t<parent_to_be_initialized(), Parent,
                           detail::EmptyInitializedVariables>>;
    //! Container for all variables
    using BlockVariables = as_block_variables<Variables>;
    //! Conformant mapping between tags and variables
    using VariableMap = VariableMapping<Variables>;
    //**************************************************************************

   private:
    //**Type definitions********************************************************
    //! CRTP Type
    using CRTP = elastica::CRTPHelper<Block<Parent>, BlockFacade>;
    //! CRTP methods
    using CRTP::self;
    //! Type of gettable
    using GetAffordance = Gettable<Block<Plugin>>;
    //! Type of spannable
    using SpanAffordance = Spannable<Block<Plugin>>;
    //**************************************************************************

   protected:
    //**Constructors************************************************************
    /*!\name Constructors */
    //@{

    //**************************************************************************
    /*!\brief The default constructor.
     *
     */
    BlockFacade() noexcept(
        std::is_nothrow_default_constructible<BlockVariables>::value)
        : CRTP(),
          GetAffordance(),
          SpanAffordance(),
          Parent(),
          variables_(){};
    //**************************************************************************

    //**************************************************************************
    /*!\brief The move constructor.
     *
     * \param other block to move from
     */
    BlockFacade(BlockFacade&& other) noexcept
        : CRTP(),
          GetAffordance(),
          SpanAffordance(),
          Parent(),
          variables_(std::move(other.variables_)){};
    //**************************************************************************

   public:
    //**************************************************************************
    /*!\brief Deleted copy constructor.
     *
     */
    BlockFacade(BlockFacade const&) = delete;
    //**************************************************************************

    //@}
    //**************************************************************************

   public:
    //**Destructor**************************************************************
    /*!\name Destructor */
    //@{
    ~BlockFacade() = default;
    //@}
    //**************************************************************************

   public:
    //**Access operators********************************************************
    /*!\name Access operators */
    //@{
    //! Operator for slicing and viewing
    using SpanAffordance::operator[];
    //@}
    //**************************************************************************

   public:
    //**Data access*************************************************************
    /*!\name Data access */
    //@{

    //**************************************************************************
    /*!\brief Access to the underlying data
     *
     * \return Mutable lvalue reference to the underlying data
     */
    inline constexpr BlockVariables& data() & noexcept { return variables_; }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to the underlying data
     *
     * \return Constant lvalue reference to the underlying data
     */
    inline constexpr BlockVariables const& data() const& noexcept {
      return variables_;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to the underlying data
     *
     * \return Mutable rvalue reference to the underlying data
     */
    inline constexpr BlockVariables&& data() && noexcept {
      return static_cast<BlockVariables&&>(variables_);
    }

    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to the underlying data
     *
     * \return Const rvalue reference to the underlying data
     */
    inline constexpr BlockVariables const&& data() const&& noexcept {
      return static_cast<BlockVariables const&&>(variables_);
    }
    //**************************************************************************

    //@}
    //**************************************************************************

    //**Utility methods*********************************************************
    /*!\name Utility methods*/
    //@{

    //**************************************************************************
    /*!\brief Returns the current block
     */
    inline constexpr auto parent() & noexcept -> Block<Plugin>& {
      return self();
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Returns the current block
     */
    inline constexpr auto parent() const& noexcept -> Block<Plugin> const& {
      return self();
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Returns the current block
     */
    inline constexpr auto parent() && noexcept -> Block<Plugin>&& {
      return static_cast<Block<Plugin>&&>(*this);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Returns the current block
     */
    inline constexpr auto parent() const&& noexcept -> Block<Plugin> const&& {
      return static_cast<Block<Plugin> const&&>(*this);
    }
    //**************************************************************************

    //@}
    //**************************************************************************

   private:
    //**************************************************************************
    /*!\brief Implementation to initialize the underlying data
     */
    template <typename DownstreamBlock, typename Initializer>
    static void initialize_impl(DownstreamBlock&& downstream_block,
                                Initializer&& initializer,
                                to_be_initialized_tag /* meta */) {
      Parent::initialize(std::forward<DownstreamBlock>(downstream_block),
                         std::forward<Initializer>(initializer));
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Implementation to initialize the underlying data
     */
    template <typename DownstreamBlock, typename Initializer>
    static void initialize_impl(DownstreamBlock&&, Initializer&&,
                                need_not_be_initialized_tag /* meta */) {}
    //**************************************************************************

   protected:
    //**************************************************************************
    /*!\brief Initialize the underlying data
     */
    template <typename DownstreamBlock, typename Initializer>
    static void initialize(DownstreamBlock&& downstream_block,
                           Initializer&& initializer) {
      This::initialize_impl(std::forward<DownstreamBlock>(downstream_block),
                            std::forward<Initializer>(initializer),
                            This::parent_to_be_initialized());
    }
    //**************************************************************************

   private:
    //**Member variables********************************************************
    /*!\name Member variables */
    //@{
    //! All block variables
    BlockVariables variables_;
    //@}
    //**************************************************************************
  };
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\name Get functions for Block */
  //@{

  //****************************************************************************
  /*!\brief Extract element from a Block
   * \ingroup blocks
   *
   * \details
   * Extracts the element of the Block `block_like` whose tag type is `Tag`.
   * Fails to compile unless the block has the `Tag` being extracted.
   *
   * \usage
   * The usage is similar to std::get(), shown below
   * \code
     Block<...> b;
     auto my_tag_data = blocks::get<tags::MyTag>(b);
   * \endcode
   *
   * \tparam Tag Tag to extract
   *
   * \param block_like The block to extract the tag from
   */
  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(Block<Plugin>& block) noexcept {
    return tuples::get<BlockVariableTag>(block.data());
  }

  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(
      Block<Plugin> const& block) noexcept {
    return tuples::get<BlockVariableTag>(block.data());
  }

  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(Block<Plugin>&& block) noexcept {
    return tuples::get<BlockVariableTag>(
        static_cast<Block<Plugin>&&>(block).data());
  }

  template <typename BlockVariableTag, typename Plugin>
  inline constexpr decltype(auto) get_backend(
      Block<Plugin> const&& block) noexcept {
    return tuples::get<BlockVariableTag>(
        static_cast<Block<Plugin> const&&>(block).data());
  }

  //@}
  /*! \endcond */
  //****************************************************************************

}  // namespace blocks

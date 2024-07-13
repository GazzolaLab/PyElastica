#pragma once
//
#include "Systems/Block/Block/Types.hpp"
//
#include "Systems/Block/Block/AsVariables.hpp"
#include "Systems/Block/Block/Concepts.hpp"
#include "Systems/Block/Block/TypeTraits.hpp"
#include "Systems/Block/BlockVariables/Aliases.hpp"
//
// #include "Utilities/PrettyType.hpp"
#include "Utilities/TMPL.hpp"

namespace blocks {

  template <typename BlockLike, typename... Tags>
  struct VariableCache : public Gettable<VariableCache<BlockLike, Tags...>> {
   public:
    //! List of variables
    using VariableSlicesList = tmpl::list<
        typename BlockLike::VariableMap::template variable_from<Tags>...>;
    //! Values of all variable slices
    using BlockVariableSlices = as_block_variables<VariableSlicesList>;

    VariableCache(BlockLike block_like)
        : source_(std::move(block_like)),
          variables_(blocks::get<Tags>(source_)...) {}
    ~VariableCache() = default;

   public:
    //**Data access*************************************************************
    /*!\name Data access */
    //@{

    //**************************************************************************
    /*!\brief Access to the underlying data
     *
     * \return Mutable lvalue reference to the underlying data
     */
    inline constexpr BlockVariableSlices& data() & noexcept {
      return variables_;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to the underlying data
     *
     * \return Constant lvalue reference to the underlying data
     */
    inline constexpr BlockVariableSlices const& data() const& noexcept {
      return variables_;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to the underlying data
     *
     * \return Mutable rvalue reference to the underlying data
     */
    inline constexpr BlockVariableSlices&& data() && noexcept {
      return static_cast<BlockVariableSlices&&>(variables_);
    }

    //**************************************************************************

    //**************************************************************************
    /*!\brief Access to the underlying data
     *
     * \return Const rvalue reference to the underlying data
     */
    inline constexpr BlockVariableSlices const&& data() const&& noexcept {
      return static_cast<BlockVariableSlices const&&>(variables_);
    }
    //**************************************************************************

    //@}
    //**************************************************************************

    //**Utility methods*********************************************************
    /*!\name Utility methods*/
    //@{

    //**************************************************************************
    /*!\brief Returns the source block
     */
    inline constexpr auto source() & noexcept -> BlockLike& { return source_; }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Returns the source block
     */
    inline constexpr auto source() const& noexcept -> BlockLike const& {
      return source_;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Returns the source block
     */
    inline constexpr auto source() && noexcept -> BlockLike&& {
      return static_cast<BlockLike&&>(source_);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Returns the source block
     */
    inline constexpr auto source() const&& noexcept -> BlockLike const&& {
      return static_cast<BlockLike const&&>(source_);
    }
    //**************************************************************************

    // Parent is needed for get() to work properly

    //**************************************************************************
    /*!\brief Returns the source block
     */
    inline constexpr auto parent() & noexcept -> BlockLike& { return source_; }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Returns the source block
     */
    inline constexpr auto parent() const& noexcept -> BlockLike const& {
      return source_;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Returns the source block
     */
    inline constexpr auto parent() && noexcept -> BlockLike&& {
      return static_cast<BlockLike&&>(source_);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Returns the source block
     */
    inline constexpr auto parent() const&& noexcept -> BlockLike const&& {
      return static_cast<BlockLike const&&>(source_);
    }
    //**************************************************************************

    //@}
    //**************************************************************************

   public:
    //**************************************************************************
    /*!\brief Human-readable name of the current plugin and all derivates
     *
     * \note This is intended to work with pretty_type::name<>
     */
    // static std::string name() { return pretty_type::name<BlockLike>(); }
    static std::string name() { return BlockLike::name(); }
    //**************************************************************************

    //**Member variables********************************************************
    /*!\name Member variables */
    //@{
    //! Source block
    BlockLike source_;
    //! Cached variables
    BlockVariableSlices variables_;
    //@}
    //**************************************************************************
  };
  //****************************************************************************

  namespace detail {

    template <typename BlockLike, typename... Tags>
    constexpr decltype(auto) cache_impl(BlockLike block_like,
                                        tmpl::list<Tags...> /* meta */) {
      using RT = VariableCache<BlockLike, Tags...>;
      return RT(std::move(block_like));
    }

  }  // namespace detail

  template <typename TagsList, typename BlockLike>
  decltype(auto) cache(Gettable<BlockLike>& gettable) noexcept {
    return detail::cache_impl(gettable.self(), TagsList{});
  }

  template <typename BlockLike>
  decltype(auto) cache(Gettable<BlockLike>& gettable) noexcept {
    // Inefficiency :
    using Tags = tmpl::transform<typename BlockLike::Variables,
                                 tmpl::bind<blocks::parameter_t, tmpl::_1>>;
    return detail::cache_impl(gettable.self(), Tags{});
  }

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\name Get functions for VariableCache */
  //@{

  //****************************************************************************
  /*!\brief Extract element from a VariableCache
   * \ingroup blocks
   *
   * \details
   * Extracts the element of the VariableCache `block_like` whose tag type is
   * `Tag`. Fails to compile unless the block has the `Tag` being extracted.
   *
   * \usage
   * The usage is similar to std::get(), shown below
   * \code
     VariableCache<...> b;
     auto my_tag_data = blocks::get<tags::MyTag>(b);
   * \endcode
   *
   * \tparam Tag Tag to extract
   *
   * \param block_like The block to extract the tag from
   */
  template <typename BlockVariableTag, typename BlockLike, typename... Tags>
  inline constexpr decltype(auto) get_backend(
      VariableCache<BlockLike, Tags...>& variable_cache) noexcept {
    return tuples::get<BlockVariableTag>(variable_cache.data());
  }

  template <typename BlockVariableTag, typename BlockLike, typename... Tags>
  inline constexpr decltype(auto) get_backend(
      VariableCache<BlockLike, Tags...> const& variable_cache) noexcept {
    return tuples::get<BlockVariableTag>(variable_cache.data());
  }

  template <typename BlockVariableTag, typename BlockLike, typename... Tags>
  inline constexpr decltype(auto) get_backend(
      VariableCache<BlockLike, Tags...>&& variable_cache) noexcept {
    return tuples::get<BlockVariableTag>(
        static_cast<VariableCache<BlockLike, Tags...>&&>(variable_cache)
            .data());
  }

  template <typename BlockVariableTag, typename BlockLike, typename... Tags>
  inline constexpr decltype(auto) get_backend(
      VariableCache<BlockLike, Tags...> const&& variable_cache) noexcept {
    return tuples::get<BlockVariableTag>(
        static_cast<VariableCache<BlockLike, Tags...> const&&>(variable_cache)
            .data());
  }

  //@}
  /*! \endcond */
  //****************************************************************************

}  // namespace blocks

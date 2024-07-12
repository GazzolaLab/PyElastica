#pragma once

//******************************************************************************
// Includes
//******************************************************************************
//
#include "Systems/Block/Block/Concepts/Types.hpp"
//
#include "Systems/Block/BlockVariables/TypeTraits.hpp"  // IsVariable
//
#include "Utilities/CRTP.hpp"
#include "Utilities/TypeTraits/Cpp20.hpp"
//
#include <cstddef>  // size_t

namespace blocks {

  //**Customization*************************************************************
  /*!\name Block get customization
   * \brief Customization of get operation
   * \ingroup block_concepts
   *
   * \details
   * The get_backend() is a customization point for implementing
   * get backend of a block-like type.
   *
   * Customization is achieved by overloading this function for the target
   * block type. If not overload is provided, then a compiler/linker error is
   * raised.
   *
   * \example
   * The following shows an example of customizing the slice backend.
   * \snippet Transfer/Test_Transfer.cpp customization_eg
   *
   * \see blocks::get()
   */
  //@{
  template <typename BlockLike>
  decltype(auto) get_backend(Gettable<BlockLike>& block_like);
  template <typename BlockLike>
  decltype(auto) get_backend(Gettable<BlockLike> const& block_like);
  template <typename BlockLike>
  decltype(auto) get_backend(Gettable<BlockLike>&& block_like);
  template <typename BlockLike>
  decltype(auto) get_backend(Gettable<BlockLike> const&& block_like);
  //@}
  //****************************************************************************

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Models the Gettable concept
   * \ingroup block_concepts
   *
   * \details
   * The Gettable template class represents the concept of extracting data from
   * a block-like data-structure using the blocks::get() method.
   *
   * \see blocks::slice(), Viewable, BlockSlice
   */
  template <typename BlockLike>
  class Gettable : public elastica::CRTPHelper<BlockLike, Gettable> {
   private:
    //**Type definitions********************************************************
    //! CRTP Type
    using CRTP = elastica::CRTPHelper<BlockLike, Gettable>;
    //**************************************************************************

   public:
    //**Self methods************************************************************
    //! CRTP methods
    using CRTP::self;
    //**************************************************************************
  };
  //****************************************************************************

  namespace detail {

    struct tag_metadata {
      //**Type definitions******************************************************
      //! Tag for marking a block variable
      using is_a_block_variable = std::true_type;
      //! Tag for marking a not block variable
      using is_not_a_block_variable = std::false_type;
      //! Variable template for determining if VarTag is a block variable
      template <typename VarTag>
      using is_block_variable = IsVariable<VarTag>;
      //************************************************************************
    };

  }  // namespace detail

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  template <typename VariableTag, typename BlockLike>
  inline decltype(auto) get_backend(
      BlockLike&& block_like,
      detail::tag_metadata::is_a_block_variable /* meta*/
      ) noexcept {
    return get_backend<VariableTag>(std::forward<BlockLike>(block_like));
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Tag, typename BlockLike>
  inline decltype(auto) get_backend(
      BlockLike&& block_like,
      detail::tag_metadata::is_not_a_block_variable /* meta*/
      ) noexcept {
    // parent is always & or const&
    using ParentBlock = cpp20::remove_cvref_t<
        decltype(std::forward<BlockLike>(block_like).parent())>;
    using VariableTag =
        typename ParentBlock::VariableMap::template variable_from<Tag>;
    return get_backend<VariableTag>(std::forward<BlockLike>(block_like));
  }
  /*! \endcond */
  //****************************************************************************

  //============================================================================
  //
  //  BLOCK GET FUNCTION
  //
  //============================================================================

  //**Get functions*************************************************************
  /*!\name Get functions*/
  //@{

  //****************************************************************************
  /*!\brief Extract element from a Gettable datastructure
   * \ingroup block_concepts
   *
   * \details
   * Extracts the element on data-structures modeling the Gettable concept.
   * Extraction is performed for the Gettable type `gettable` whose element to
   * be extracted has a tag type `Tag`. Fails to compile unless the block has
   * the `Tag` being extracted.
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

  //****************************************************************************
  /*!\brief Slice operator
   * \ingroup block_concepts
   *
   * \details
   * Implements slices on entities modeling Sliceable concept.
   *
   * \param sliceable Data-structure modeling the Sliceable concept
   * \param index The index of the slice (int or from_end)
   *
   * \example
   * Given a block `b`, we can slice it using
   * \code
   * auto b_slice = block::slice(b, 5UL); // Gets b[5], at index 5 from start
   * // Gets b[-2]
   * auto b_slice_from_end = block::slice(b, elastica::end - 2UL); // 2 from end
   * \endcode
   *
   * \note
   * not marked noexcept because we can have out of range indices
   *
   * \see blocks::Sliceable , blocks::BlockSlice
   */

  //****************************************************************************
  template <typename Tag, typename BlockLike>
  inline decltype(auto) get(Gettable<BlockLike>& gettable) noexcept {
    return get_backend<Tag>(gettable.self(),
                            detail::tag_metadata::is_block_variable<Tag>{});
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Tag, typename BlockLike>
  inline decltype(auto) get(Gettable<BlockLike> const& gettable) noexcept {
    return get_backend<Tag>(gettable.self(),
                            detail::tag_metadata::is_block_variable<Tag>{});
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Tag, typename BlockLike>
  inline decltype(auto) get(Gettable<BlockLike>&& gettable) noexcept {
    return get_backend<Tag>(static_cast<BlockLike&&>(gettable),
                            detail::tag_metadata::is_block_variable<Tag>{});
  }
  //****************************************************************************

  //****************************************************************************
  template <typename Tag, typename BlockLike>
  inline decltype(auto) get(Gettable<BlockLike> const&& gettable) noexcept {
    return get_backend<Tag>(static_cast<BlockLike const&&>(gettable),
                            detail::tag_metadata::is_block_variable<Tag>{});
  }
  //****************************************************************************

}  // namespace blocks

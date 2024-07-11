#pragma once

//******************************************************************************
// Includes
//******************************************************************************

// #include "Systems/Block/Block/Block.hpp"  // only declaration, so ok to include.
#include "Systems/Block/Block/Concepts/Types.hpp"

namespace blocks {

  //////////////////////////////////////////////////////////////////////////////
  //
  // Forward declarations of block types
  //
  //////////////////////////////////////////////////////////////////////////////

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  template <typename /* Plugin */>
  class BlockSlice;

  template <typename /* Plugin */>
  class ConstBlockSlice;

  template <typename /*Plugin*/>
  class BlockFacade;

  template <typename /* Plugin */>
  class BlockView;

  template <typename /* Plugin */>
  class ConstBlockView;

  template <typename /*Plugin*/>
  class BlockIterator;

  template <typename /*Plugin*/>
  class ConstBlockIterator;

  template <typename /*Plugin*/>
  struct Metadata;

  template <typename /*Derived*/>
  class Sliceable;

  template <typename BlockLike, typename... Tags>
  struct VariableCache;

  /*! \endcond */
  //****************************************************************************

}  // namespace blocks

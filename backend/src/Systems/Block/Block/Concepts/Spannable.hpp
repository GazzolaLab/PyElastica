#pragma once

//******************************************************************************
// Includes
//******************************************************************************
//
#include "Systems/Block/Block/Concepts/Types.hpp"
//
#include "Systems/Block/Block/Concepts/Spannable.hpp"
#include "Systems/Block/Block/Concepts/Subdividable.hpp"
#include "Systems/Block/Block/Concepts/Viewable.hpp"
//
#include "Utilities/CRTP.hpp"
#include "Utilities/End.hpp"
//
#include <cstddef>  // size_t

namespace blocks {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Models the Spannable concept
   * \ingroup block_concepts
   *
   * \details
   * The Spannable template class represents the (composite) concept of a
   * spannable block-like data-structure. A Spannable must satisfy the
   * requirements of blocks::Subdividable, blocks::Sliceable, blocks::Viewable
   */
  template <typename BlockLike>
  class Spannable : public elastica::CRTPHelper<BlockLike, Spannable>,
                    public Subdividable<BlockLike>,
                    public Sliceable<BlockLike>,
                    public Viewable<BlockLike> {
   private:
    //**Type definitions********************************************************
    //! CRTP Type
    using CRTP = elastica::CRTPHelper<BlockLike, Spannable>;
    //**************************************************************************

   protected:
    //**Type definitions********************************************************
    //! Type of subdividable
    using SubdividableAffordance = Subdividable<BlockLike>;
    //! Type of sliceable
    using SliceAffordance = Sliceable<BlockLike>;
    //! Type of viewable
    using ViewAffordance = Viewable<BlockLike>;
    //**************************************************************************

   public:
    //**Access operators********************************************************
    /*!\name Access operators */
    //@{
    //! Operator for slicing
    using SliceAffordance::operator[];
    //! Operator for viewing
    using ViewAffordance::operator[];
    //@}
    //**************************************************************************
  };
  //****************************************************************************

}  // namespace blocks

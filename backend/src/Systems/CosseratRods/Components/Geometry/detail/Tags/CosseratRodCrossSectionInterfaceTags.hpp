#pragma once

namespace elastica {

  //============================================================================
  //
  //  TAG DEFINITIONS
  //
  //============================================================================

  namespace tags {

    //**************************************************************************
    /*!\brief Tag indicating the dimension (i.e the characteristic length scale)
     * of an element of a cosserat rod. Usually this is the radius for rods with
     * circular cross sections and side lengths for rods with square cross
     * sections
     * \ingroup system_tags
     */
    struct ElementDimension {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating volumes of an element of a cosserat rod
     * \ingroup system_tags
     */
    struct ElementVolume {};
    //**************************************************************************

  }  // namespace tags

}  // namespace elastica

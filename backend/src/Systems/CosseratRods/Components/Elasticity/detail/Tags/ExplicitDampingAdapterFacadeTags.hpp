#pragma once

namespace elastica {

  //============================================================================
  //
  //  TAG DEFINITIONS
  //
  //============================================================================

  namespace tags {

    //**************************************************************************
    /*!\brief Tag indicating rate of damping for the translational component of
     * of cosserat rod dynamics. Used in adding a damping force with an usual
     * form of
     * \[F_damp \propto mass \cdot rate \cdot v\]
     * \ingroup system_tags
     */
    struct ForceDampingRate {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating rate of damping for the rotational component of
     * of cosserat rod dynamics. Used in adding a damping torque with an usual
     * form of
     * \[T_damp \propto mass \cdot rate \cdot \omega \]
     * \ingroup system_tags
     */
    struct TorqueDampingRate {};
    //**************************************************************************

  }  // namespace tags

}  // namespace elastica

#pragma once

namespace elastica {

  //============================================================================
  //
  //  TAG DEFINITIONS
  //
  //============================================================================
  namespace tags {

    //**************************************************************************
    /*!\brief Tag indicating rigidities for the bending and twisting mode of
     * of a cosserat rod
     * \ingroup system_tags
     */
    struct BendingTwistRigidityMatrix {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating rigidities for the shearing and stretching mode of
     * of a cosserat rod
     * \ingroup system_tags
     */
    struct ShearStretchRigidityMatrix {};
    //**************************************************************************

  }  // namespace tags

}  // namespace elastica

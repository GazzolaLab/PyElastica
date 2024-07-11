#pragma once

namespace elastica {

  namespace tags {

    //==========================================================================
    //
    //  TAG DEFINITIONS
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Tag indicating positions in the Eulerian lab frame of reference.
     * \ingroup system_tags
     */
    struct Position {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating velocity in the Eulerian lab frame of reference.
     * \ingroup system_tags
     */
    struct Velocity {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating translational acceleration in the Eulerian lab
     * frame of reference.
     * \ingroup system_tags
     */
    struct Acceleration {};
    //**************************************************************************

    // so3 group

    //**************************************************************************
    /*!\brief Tag indicating orientations as a set of three axes, called the
     * directors, in the Lagrangian material frame of reference
     * \ingroup system_tags
     */
    struct Director {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating angular velocity in the Lagrangian material frame
     * of reference
     * \ingroup system_tags
     */
    struct AngularVelocity {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating angular accelerations in the Lagrangian material
     * frame
     * \ingroup system_tags
     */
    struct AngularAcceleration {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating material description of a physical system, see
     * elastica::Material
     * \ingroup system_tags
     */
    struct Material {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating mass of a physical system
     * \ingroup system_tags
     */
    struct Mass {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating mass second moment of inertia of a physical system
     * \ingroup system_tags
     */
    struct MassSecondMomentOfInertia {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating inverse mass of a physical system
     * \ingroup system_tags
     */
    struct InvMass {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating inverse mass second moment of inertia of a physical
     * system
     * \ingroup system_tags
     */
    struct InvMassSecondMomentOfInertia {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating external loads in the Eulerian lab frame of
     * reference
     * \ingroup system_tags
     */
    struct ExternalLoads {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating external torques in the Lagrangian material frame
     * of reference
     * \ingroup system_tags
     */
    struct ExternalTorques {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating total number of elements within a system of
     * cardinality != 1
     * \ingroup system_tags
     */
    struct NElement {};
    //**************************************************************************

  }  // namespace tags

}  // namespace elastica

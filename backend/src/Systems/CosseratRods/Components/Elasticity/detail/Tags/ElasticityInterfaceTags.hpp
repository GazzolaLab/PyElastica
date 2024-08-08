#pragma once

namespace elastica {

  //============================================================================
  //
  //  TAG DEFINITIONS
  //
  //============================================================================
  namespace tags {

    //**************************************************************************
    /*!\brief Tag indicating internal loads of a cosserat rod
     * \ingroup system_tags
     */
    struct InternalLoads {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating internal stress of a cosserat rod
     * \ingroup system_tags
     */
    struct InternalStress {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating internal torques of a cosserat rod
     * \ingroup system_tags
     */
    struct InternalTorques {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating internal couples of a cosserat rod
     * \ingroup system_tags
     */
    struct InternalCouple {};
    //**************************************************************************

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    struct _DummyElementVector {};
    struct _DummyElementVector2 {};
    struct _DummyVoronoiVector {};
    /*! \endcond */
    //**************************************************************************

  }  // namespace tags

}  // namespace elastica

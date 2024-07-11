#pragma once

namespace elastica {

  //============================================================================
  //
  //  TAG DEFINITIONS
  //
  //============================================================================

  namespace tags {

    //**************************************************************************
    /*!\brief Tag indicating curvature of a cosserat rod
     * \ingroup system_tags
     */
    struct Curvature {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating dilatation (stretch) of elements in a cosserat rod
     * \ingroup system_tags
     */
    struct ElementDilatation {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating length of elements in a cosserat rod
     * \ingroup system_tags
     */
    struct ElementLength {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating reference (stress-free state) curvature of a
     * cosserat rod
     * \ingroup system_tags
     */
    struct ReferenceCurvature {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating reference (stress-free state) element lengths of a
     * cosserat rod
     * \ingroup system_tags
     */
    struct ReferenceElementLength {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating reference (stress-free state) strains associated
     * with shear and stretch modes of a cosserat rod
     * \ingroup system_tags
     */
    struct ReferenceShearStretchStrain {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating reference (stress-free state) voronoi lengths of a
     * cosserat rod
     * \ingroup system_tags
     */
    struct ReferenceVoronoiLength {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating  strains associated with shear and stretch modes of
     * a cosserat rod
     * \ingroup system_tags
     */
    struct ShearStretchStrain {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating tangents associated with a cosserat rod
     * \ingroup system_tags
     */
    struct Tangent {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating dilatation (stretch) of voronoi regions in a
     * cosserat rod
     * \ingroup system_tags
     */
    struct VoronoiDilatation {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Tag indicating length of voronoi regions in a cosserat rod
     * \ingroup system_tags
     */
    struct VoronoiLength {};
    //**************************************************************************

  }  // namespace tags

}  // namespace elastica

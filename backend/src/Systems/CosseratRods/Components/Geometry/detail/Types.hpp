#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/CosseratRods/Components/Geometry/detail/Tags.hpp"
#include "Systems/CosseratRods/Components/Noexcept.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      namespace detail {

        ////////////////////////////////////////////////////////////////////////
        //
        // Forward declarations of helper geometry components
        //
        ////////////////////////////////////////////////////////////////////////

        //**********************************************************************
        /*! \cond ELASTICA_INTERNAL */
        template <typename /* Rod Traits */, typename /*Block*/>
        class RodSpanwiseGeometry;

        template <typename /* Rod Traits */, typename /*Block*/>
        class CosseratRodSpanwiseGeometry;

        template <typename /* Rod Traits */, typename /* Block */,
                  // template <typename /* Rod Traits */, typename /* Block */>
                  class /*Cross Section Implementation*/>
        class CosseratRodCrossSectionInterface;
        /*! \endcond */
        //**********************************************************************

      }  // namespace detail

    }  // namespace component

    ////////////////////////////////////////////////////////////////////////////
    //
    // Forward declarations of functions
    //
    ////////////////////////////////////////////////////////////////////////////

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    template <typename CRT, typename ComputationalBlock>
    void compute_reference_geometry(
        component::detail::CosseratRodSpanwiseGeometry<CRT, ComputationalBlock>&
            block_like) COSSERATROD_LIB_NOEXCEPT;

    template <typename CRT, typename ComputationalBlock>
    void update_spanwise_variables(
        component::detail::CosseratRodSpanwiseGeometry<CRT, ComputationalBlock>&
            block_like) COSSERATROD_LIB_NOEXCEPT;

    template <typename Traits, typename ComputationalBlock>
    void update_shear_stretch_strain(
        component::detail::CosseratRodSpanwiseGeometry<
            Traits, ComputationalBlock>& block_like) COSSERATROD_LIB_NOEXCEPT;

    template <typename Traits, typename ComputationalBlock>
    void update_curvature(component::detail::CosseratRodSpanwiseGeometry<
                          Traits, ComputationalBlock>& block_like)
        COSSERATROD_LIB_NOEXCEPT;

    template <typename Traits, typename ComputationalBlock,
              typename CrossSection>
    void compute_cross_sectional_variables(
        component::detail::CosseratRodCrossSectionInterface<
            Traits, ComputationalBlock, CrossSection>& block_like)
        COSSERATROD_LIB_NOEXCEPT;

    template <typename Traits, typename ComputationalBlock,
              typename CrossSection>
    void compute_shear_strains(
        component::detail::CosseratRodCrossSectionInterface<
            Traits, ComputationalBlock, CrossSection>& block_like)
        COSSERATROD_LIB_NOEXCEPT;

    template <typename Traits, typename ComputationalBlock,
              typename CrossSection>
    void compute_curvature(component::detail::CosseratRodCrossSectionInterface<
                           Traits, ComputationalBlock, CrossSection>&
                               block_like) COSSERATROD_LIB_NOEXCEPT;
    /*! \endcond */
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/CosseratRods/Components/Elasticity/detail/Tags.hpp"
//
#include "Systems/CosseratRods/Components/Noexcept.hpp"
#include "Utilities/Math/Types.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      namespace detail {

        ////////////////////////////////////////////////////////////////////////
        //
        // Forward declarations of helper elasticity components
        //
        ////////////////////////////////////////////////////////////////////////

        //**********************************************************************
        /*! \cond ELASTICA_INTERNAL */
        template <typename /* Rod Traits */, typename /*Block*/,
                  typename /* ElasticityModel */>
        class ElasticityInterface;

        template <typename /* Rod Traits */, typename /*Block*/>
        class LinearHyperElasticityFacade;

        template <typename /* Rod Traits */, typename /* Block */,
                  class /*ElasticityModelWithBlock*/,
                  template <typename /* Rod Traits */> class /*VariableInfo*/
                  >
        class ExplicitDampingAdapterFacade;

        template <typename /* Rod Traits */, typename /*Block*/,
                  class /*ElasticityModel*/>
        class ExplicitDampingAdapter;

        template <typename /* Rod Traits */, typename /*Block*/,
                  class /*ElasticityModel*/>
        class ExplicitDampingAdapterPerRod;

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
    template <typename Traits, typename ComputationalBlock>
    auto compute_effective_area_along_principal_directions(
        component::detail::LinearHyperElasticityFacade<
            Traits, ComputationalBlock>& block_like,
        std::size_t i) COSSERATROD_LIB_NOEXCEPT -> Vec3;

    template <typename Traits, typename ComputationalBlock,
              class ElasticityModel>
    void compute_internal_forces(
        component::detail::ElasticityInterface<Traits, ComputationalBlock,
                                               ElasticityModel>& block_like)
        COSSERATROD_LIB_NOEXCEPT;

    template <typename Traits, typename ComputationalBlock,
              class ElasticityModel>
    void compute_internal_torques(
        component::detail::ElasticityInterface<Traits, ComputationalBlock,
                                               ElasticityModel>& block_like)
        COSSERATROD_LIB_NOEXCEPT;

    template <typename CRT, typename ComputationalBlock,
              class ElasticityModelWithBlock,
              template <typename /* CRT */> class VariableInfo>
    void compute_internal_modeled_loads_impl(
        component::detail::ExplicitDampingAdapterFacade<
            CRT, ComputationalBlock, ElasticityModelWithBlock, VariableInfo>&
            block_like) COSSERATROD_LIB_NOEXCEPT;

    template <typename CRT, typename ComputationalBlock,
              class ElasticityModelWithBlock,
              template <typename /* CRT */> class VariableInfo>
    void compute_internal_modeled_torques_impl(
        component::detail::ExplicitDampingAdapterFacade<
            CRT, ComputationalBlock, ElasticityModelWithBlock, VariableInfo>&
            block_like) COSSERATROD_LIB_NOEXCEPT;
    /*! \endcond */
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

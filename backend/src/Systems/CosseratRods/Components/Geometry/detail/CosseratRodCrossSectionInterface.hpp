#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cmath>
#include <utility>

#include "Simulator/Frames/LagrangianFrame.hpp"
#include "Utilities/CRTP.hpp"
#include "Utilities/Math/Vec3.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

///// Types always first
#include "Systems/CosseratRods/Components/Geometry/detail/Types.hpp"
/////
#include "Systems/CosseratRods/Components/Geometry/detail/CosseratRodSpanwiseGeometry.hpp"
#include "Systems/CosseratRods/Components/Geometry/detail/Tags/CosseratRodCrossSectionInterfaceTags.hpp"
#include "Systems/CosseratRods/Components/Noexcept.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      namespace detail {

        //======================================================================
        //
        //  CLASS DEFINITION
        //
        //======================================================================

        //**********************************************************************
        /*!\brief Variables corresponding to the geometry interface component
         * \ingroup cosserat_rod_component
         *
         * \details
         * CosseratRodCrossSectionInterfaceVariables contains the definitions of
         * variables used within the Blocks framework for a rod data-structure
         * with an interface expected by other components.
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         *
         * \see CosseratRodCrossSectionInterface
         */
        template <typename CRT>
        class CosseratRodCrossSectionInterfaceVariables {
         private:
          //**Type definitions**************************************************
          //! Traits type
          using Traits = CRT;
          //********************************************************************

         protected:
          //**Variable definitions**********************************************
          /*!\name Variable definitions*/
          //@{

          //********************************************************************
          /*!\brief Variable marking ElementDimension within the Cosserat rod
           * hierarchy
           */
          struct ElementDimension
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::ElementDimension,  //
                    typename Traits::DataType::Scalar,   //
                    typename Traits::Place::OnElement> {
            //**Type definitions************************************************
            //! The element type of a ghost
            using ghost_type = typename Traits::DataType::Scalar::ghost_type;
            //******************************************************************

            //******************************************************************
            /*!\brief Obtain the ghost value
             *
             * \details
             * Overrides the default value for putting in ghost elements.
             */
            static inline constexpr auto ghost_value() noexcept -> ghost_type {
              return ghost_type(1.0);
            }
            //******************************************************************
          };
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking ElementVolume within the Cosserat rod
           * hierarchy
           */
          struct ElementVolume : public Traits::template CosseratRodVariable<
                                     ::elastica::tags::ElementVolume,    //
                                     typename Traits::DataType::Scalar,  //
                                     typename Traits::Place::OnElement> {
            //**Type definitions************************************************
            //! The element type of a ghost
            using ghost_type = typename Traits::DataType::Scalar::ghost_type;
            //******************************************************************

            //******************************************************************
            /*!\brief Obtain the ghost value
             *
             * \details
             * Overrides the default value for putting in ghost elements.
             */
            static inline constexpr auto ghost_value() noexcept -> ghost_type {
              return ghost_type(1.0);
            }
            //******************************************************************
          };
          //********************************************************************

          //@}
          //********************************************************************

          //**Type definitions**************************************************
          //! List of computed variables
          using ComputedVariables = tmpl::list<ElementVolume>;
          //! List of initialized variables
          using InitializedVariables = tmpl::list<ElementDimension>;
          //! List of all variables
          using Variables =
              tmpl::append<InitializedVariables, ComputedVariables>;
          //********************************************************************
        };
        //**********************************************************************

        // TODO : Protocol for cross section trait
        //======================================================================
        //
        //  CLASS DEFINITION
        //
        //======================================================================

        //**********************************************************************
        /*!\brief Interface component corresponding to a spanwise rod with a
         * particular shape of cross section
         * \ingroup cosserat_rod_component
         *
         * \details
         * CosseratRodCrossSectionInterface implements the penultimate Geometry
         * component use within Cosserat rods implemented in the Blocks
         * framework. It denotes a Cosserat rod data-structure with a particular
         * shape of cross section (such as a square or a circle) spanning a
         * single spatial dimension (i.e. along centerline coordinates)
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         * \tparam ComputationalBlock The final block which is derived from the
         * current component
         * \tparam CrossSectionOperations Operations for particular cross
         * section shapes (such as a square or a circle)
         *
         * \see CosseratRodCrossSectionInterfaceVariables
         */
        template <typename CRT, typename ComputationalBlock,
                  typename CrossSectionOperations>
        class CosseratRodCrossSectionInterface
            : public CosseratRodSpanwiseGeometry<CRT, ComputationalBlock>,
              public CosseratRodCrossSectionInterfaceVariables<CRT>,
              public CrossSectionOperations,
              public CRTPHelper<ComputationalBlock,
                                CosseratRodCrossSectionInterface> {
         protected:
          //**Type definitions**************************************************
          //! Frame type
          using Frame = ::elastica::detail::LagrangianFrame;
          //! Frame direction type
          using DirectionType = typename Frame::DirectionType;
          //! Cross section type
          // this is protected for access in CrossSectionOperations
          using CrossSection = CrossSectionOperations;
          //********************************************************************

         private:
          //**Type definitions**************************************************
          //! Traits type
          using Traits = CRT;
          //! This type
          using This =
              CosseratRodCrossSectionInterface<Traits, ComputationalBlock,
                                               CrossSection>;
          //! Parent type
          using Parent =
              CosseratRodSpanwiseGeometry<Traits, ComputationalBlock>;
          //! Variable definitions
          using VariableDefinitions =
              CosseratRodCrossSectionInterfaceVariables<Traits>;
          //! CRTP Type
          using CRTP =
              CRTPHelper<ComputationalBlock, CosseratRodCrossSectionInterface>;
          //! Real number type
          using real_type = typename Traits::real_type;
          //! Index type
          using index_type = typename Traits::index_type;
          //********************************************************************

         protected:
          //**Type definitions**************************************************
          //! List of initialized variables
          using InitializedVariables =
              tmpl::append<typename Parent::InitializedVariables,
                           typename VariableDefinitions::InitializedVariables>;
          //! List of computed variables
          using ComputedVariables =
              tmpl::append<typename Parent::ComputedVariables,
                           typename VariableDefinitions::ComputedVariables>;
          //! List of all variables
          using Variables =
              tmpl::append<typename Parent::Variables,
                           typename VariableDefinitions::Variables>;
          //! Current Variable definitions
          using typename Parent::Curvature;
          using typename Parent::ElementDilatation;
          using typename Parent::ElementLength;
          using typename Parent::ReferenceCurvature;
          using typename Parent::ReferenceElementLength;
          using typename Parent::ReferenceVoronoiLength;
          using typename Parent::ShearStretchStrain;
          using typename Parent::Tangent;
          using typename Parent::VoronoiDilatation;
          using typename Parent::VoronoiLength;

          using typename VariableDefinitions::ElementDimension;
          using typename VariableDefinitions::ElementVolume;
          //********************************************************************

         private:
          //**Parent methods****************************************************
          //! Methods inherited from parent class
          using CrossSection::compute_cross_section_area;
          using CrossSection::compute_second_moment_of_area;
          using CrossSection::update_dimension;

         public:
          using CrossSection::shape_factor;
          //********************************************************************

          //**Static members****************************************************
          //! unique ID for D1 direction
          static constexpr DirectionType D1 = Frame::D1;
          //! unique ID for D2 direction
          static constexpr DirectionType D2 = Frame::D2;
          //! unique ID for D3 direction
          static constexpr DirectionType D3 = Frame::D3;
          //********************************************************************

         protected:
          //********************************************************************
          /*!\copydoc ComponentInitializationDocStub
           */
          template <typename BlockLike, typename CosseratInitializer>
          static void initialize(
              CosseratRodCrossSectionInterface<Traits, BlockLike, CrossSection>&
                  this_component,
              CosseratInitializer&& initializer) {
            // 1. Initialize parent
            Parent::initialize(this_component,
                               std::forward<CosseratInitializer>(initializer));

            // 2. Initialize required variables
            initialize_component<
                typename VariableDefinitions::InitializedVariables>(
                this_component.self(),
                std::forward<CosseratInitializer>(initializer));

            // 3. Computed tags (<conserved> volumes)
            {
              auto&& volumes(blocks::get<::elastica::tags::ElementVolume>(
                  this_component.self()));
              auto&& dimensions(blocks::get<::elastica::tags::ElementDimension>(
                  this_component.self()));
              // FIXME : use lengths instead of ref_lengths
              // https://github.com/tp5uiuc/elasticapp/issues/462
#define FIX_462_IMPLEMENTED 0
#if FIX_462_IMPLEMENTED
              using LengthVariableTag = ::elastica::tags::ElementLength;
#else
              using LengthVariableTag =
                  ::elastica::tags::ReferenceElementLength;
#endif
#undef FIX_462_IMPLEMENTED

              auto&& lengths_variable(
                  blocks::get<LengthVariableTag>(this_component.self()));

              volumes =
                  compute_cross_section_area(dimensions) * lengths_variable;
            }

            // 4. Update cross-sectional variables (dimension)
            { compute_cross_sectional_variables(this_component.self()); }
          }
          //********************************************************************

         public:
          //**CRTP method*******************************************************
          /*!\name CRTP method*/
          //@{
          using CRTP::self;
          //@}
          //********************************************************************

          //**Get methods*******************************************************
          /*!\name Get methods*/
          //@{

          //********************************************************************
          /*!\name dimension methods*/
          //@{
          /*!\brief Gets dimension of the current rod
           */
          inline constexpr decltype(auto) get_element_dimension() & noexcept {
            return blocks::get<tags::ElementDimension>(self());
          }
          inline constexpr decltype(auto) get_element_dimension()
              const& noexcept {
            return blocks::get<tags::ElementDimension>(self());
          }
          inline constexpr decltype(auto) get_element_dimension(
              index_type idx) & noexcept {
            return ElementDimension::slice(get_element_dimension(), idx);
          }
          inline constexpr decltype(auto) get_element_dimension(
              index_type idx) const& noexcept {
            return ElementDimension::slice(get_element_dimension(), idx);
          }
          //@}
          //********************************************************************

          //********************************************************************
          /*!\name volume methods*/
          //@{
          /*!\brief Gets volume of the current rod
           */
          inline constexpr decltype(auto) get_element_volume() & /*noexcept*/ {
            return blocks::get<::elastica::tags::ElementVolume>(self());
          }
          inline constexpr decltype(auto) get_element_volume() const& noexcept {
            return blocks::get<::elastica::tags::ElementVolume>(self());
          }
          inline constexpr decltype(auto) get_element_volume(
              index_type idx) & noexcept {
            return ElementVolume::slice(get_element_volume(), idx);
          }
          inline constexpr decltype(auto) get_element_volume(
              index_type idx) const& noexcept {
            return ElementVolume::slice(get_element_volume(), idx);
          }
          //@}
          //********************************************************************

          //********************************************************************
          /*!\name area methods*/
          //@{
          /*!\brief Gets area of the current rod
           */
          inline constexpr auto get_area() const& COSSERATROD_LIB_NOEXCEPT {
            // Should give out a VecVecExpr rather than Dimensions object
            // for expression template mechanisms
            return compute_cross_section_area(This::get_element_dimension());
          }
          inline constexpr auto get_area(
              index_type idx) const& COSSERATROD_LIB_NOEXCEPT {
            // Should give out a VecVecExpr rather than Dimensions object
            // for expression template mechanisms
            return compute_cross_section_area(This::get_element_dimension(idx));
          }
          //@}
          //********************************************************************

          //********************************************************************
          /*!\name second moment of area methods*/
          //@{
          /*!\brief Gets second moment of area of the current rod
           */
          template <DirectionType Dim, Requires<(Dim < Frame::D3)> = nullptr>
          inline constexpr decltype(auto) get_second_moment_of_area(
              index_type idx) const COSSERATROD_LIB_NOEXCEPT {
            // Should give out a VecVecExpr rather than Dimensions object
            // for expression template mechanisms
            return compute_second_moment_of_area(
                This::get_element_dimension(idx));
          }
          template <DirectionType Dim, Requires<(Dim == Frame::D3)> = nullptr>
          inline constexpr decltype(auto) get_second_moment_of_area(
              index_type idx) const COSSERATROD_LIB_NOEXCEPT {
            // Should give out a VecVecExpr rather than Dimensions object
            // for expression template mechanisms
            return real_t(2.0) * compute_second_moment_of_area(
                                     This::get_element_dimension(idx));
          }
          // do not have APIs return vec3, seems to reduce customizability and
          // promotes bad code
          inline constexpr auto get_second_moment_of_area(index_type idx) const
              COSSERATROD_LIB_NOEXCEPT {
            return Vec3{real_t(This::get_second_moment_of_area<D1>(idx)),
                        real_t(This::get_second_moment_of_area<D2>(idx)),
                        real_t(This::get_second_moment_of_area<D3>(idx))};
          }

          //@}
          //********************************************************************

          //@}
          //********************************************************************
        };
        //**********************************************************************

      }  // namespace detail

    }  // namespace component

    //**************************************************************************
    /*!\brief Update rod cross-sectional geometric properties given state.
     * \ingroup cosserat_rod_custom_entries
     *
     * \details
     * Ref: eq (3.3) from \cite gazzola2018forward. \n
     * radius{i} = (volume{i} / M_PI / length{i})^0.5
     *
     * The following members are modified:
     *  - radius(n_elems) (tag: ElementDimension)
     *
     * with the following inputs:
     *  - volume_batch(n_elems)
     *  - element_length (3, n_nodes)
     *
     * \return void/None
     */
    template <typename Traits, typename ComputationalBlock,
              typename CrossSection>
    void compute_cross_sectional_variables(
        component::detail::CosseratRodCrossSectionInterface<
            Traits, ComputationalBlock, CrossSection>& block_like)
        COSSERATROD_LIB_NOEXCEPT {
      auto&& volume(blocks::get<tags::ElementVolume>(block_like.self()));
      auto&& element_length(
          blocks::get<tags::ElementLength>(block_like.self()));
      auto&& dimensions(blocks::get<tags::ElementDimension>(block_like.self()));
      dimensions = CrossSection::update_dimension(volume, element_length);
    }
    //**********************************************************************

    //**************************************************************************
    /*!\brief Compute elemental shear and stretch mode strains
     * \ingroup cosserat_rod_custom_entries
     *
     * \details
     * Computes elemental shear and stretch modes strains given the rod
     * positions and directors (state), calls compute_all_dilatations()
     * inside. Ref: eq (2.9) from \cite gazzola2018forward.
     *
     * strain_batch{i} =
     * dilatation_batch{i} * director_collection_batch{i} *
     * tangent_batch{i} - {0.0, 0.0, 1.0}
     *
     * The following members are modified:
     * - strain_batch(3, n_elems)
     *
     * with the following inputs:
     * - director_collection_batch(3, 3, n_elems)
     * - temp container of (3, n_elems) for intermediate storage
     *
     * given inputs of compute_all_dilatations() are set.
     *
     * \example
     * The following shows a typical use of the
     * compute_shear_stretch_strains()
     * function with the expected (correct) result also shown.
     * \snippet test_gov_eqns.cpp compute_shear_stretch_strains_example
     *
     * \return void/None
     *
     * \see fill later?
     */
    template <typename Traits, typename ComputationalBlock,
              typename CrossSection>
    void compute_shear_strains(
        component::detail::CosseratRodCrossSectionInterface<
            Traits, ComputationalBlock, CrossSection>& block_like)
        COSSERATROD_LIB_NOEXCEPT {
      // Q (et - d_3)
      // Q is always latest, only e and t need to be computed
      // Dilatation updated here!
      update_spanwise_variables(block_like.self());
      compute_cross_sectional_variables(block_like.self());
      update_shear_stretch_strain(block_like.self());
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Compute curvature
     * \ingroup cosserat_rod_custom_entries
     * \note
     * kappa is the only rotational diagnostic, so we direcetly compute
     * that here
     */
    template <typename Traits, typename ComputationalBlock,
              typename CrossSection>
    void compute_curvature(component::detail::CosseratRodCrossSectionInterface<
                           Traits, ComputationalBlock, CrossSection>&
                               block_like) COSSERATROD_LIB_NOEXCEPT {
      update_curvature(block_like.self());
    }
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

//  public RodKinematics<RodTraits>
// Sparse_strategy
// Dense strategy
/* Lessons learnt:
 * We wanted to CRTP multiple times to achieve multiple-inheritance
 * during compile-time. However putting the tag structs inside the
 * CRTP base class is problematic as the tags are also stamped with
 * the CRTP-ed derived class (aka the CrossSection template) below.
 * For some reason (still unknown), this causes the tuples to not find
 * the tags initialized from the block ||-> plugin || -> circle geometry
 * which is then specially CRTped to the CrossSectionBase class.
 *
 * The error comes because
 * We search for
    elastica::detail_cosserat_rod::CosseratRodTraits::VectorSlice<elastica::detail_cosserat_rod::CrossSectionBase<elastica::detail_cosserat_rod::CosseratRodTraits,
    elastica::detail_cosserat_rod::CircleCrossSection<elastica::detail_cosserat_rod::CosseratRodTraits,
    elastica::Block<elastica::CosseratRodPlugin<elastica::detail_cosserat_rod::CosseratRodTraits,
    elastica::Block, elastica::detail_cosserat_rod::CircleCrossSection> > >
    >::VolumeCollectionTag, elastica::detail_cosserat_rod::ElementTagImpl>;

    But in the tuple we only have
    elastica::detail_cosserat_rod::CosseratRodTraits::VectorSlice<elastica::detail_cosserat_rod::CrossSectionBase<elastica::detail_cosserat_rod::CosseratRodTraits,
    elastica::detail_cosserat_rod::CircleCrossSection<elastica::detail_cosserat_rod::CosseratRodTraits,
    elastica::CosseratRodPlugin<elastica::detail_cosserat_rod::CosseratRodTraits,
    elastica::Block, elastica::detail_cosserat_rod::CircleCrossSection> >
    >::VolumeCollectionTag, elastica::detail_cosserat_rod::ElementTagImpl>,

    Notice how an extra "Block" template gets added to the search parameter
   around the CosseratRodPlugin. This seems to be the right type to be
   searched for, at least while looking at the hierarchy that we have.
 However in the tuple types there's a deficiency of the block types for some
 reason.

   To avoid these hassle, we again split the CrossSectionBase into
 CrossSectionOperations templated on the CosseratRodTraits. The sole purpose
 of this parent is just to provide meta-data on what the cross-section base
 should contain.
 *
 */

/*
 * In this multiple CRTP inheritance, another trick is used. We know
 * that the CrossSection class will only have static members. Then
 * instead of templating on the ComputaionalBlock (the impl) class
 * we template on the CrossSection instead and give it a friend access.
 * Thie leads to a more "sensible" code since only CrossSection can
 * be used as the final class (aka CrossSectionBase cannot be used
 * to instantiate a Cosserat Rod), thus preventing wrong misuse. Also
 * the CrossSection hierarchy is self contained and can be reused
 * without worrying about blocks getting interchanged in the
 * CrossSection Base and CrossSection (since only one block is
 * templated)
 *
 *
 */

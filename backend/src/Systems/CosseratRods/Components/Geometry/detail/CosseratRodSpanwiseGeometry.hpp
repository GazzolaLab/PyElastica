#pragma once

//****************************************************************************
// Includes
//****************************************************************************

#include "ErrorHandling/Assert.hpp"
// module
#include "Systems/Block.hpp"
/// Types always first
#include "Systems/CosseratRods/Components/Geometry/detail/Types.hpp"
#include "Systems/CosseratRods/_Types.hpp"  // for ghosts lookup
///
#include "Systems/CosseratRods/Components/Geometry/detail/RodSpanwiseGeometry.hpp"
#include "Systems/CosseratRods/Components/Geometry/detail/Tags/CosseratRodSpanwiseGeometryTags.hpp"
#include "Systems/CosseratRods/Components/Initialization.hpp"
#include "Systems/CosseratRods/Components/Noexcept.hpp"
//
#include "Utilities/CRTP.hpp"
#include "Utilities/TMPL.hpp"
//
#include <cstddef>
#include <utility>

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      namespace detail {

        //**********************************************************************
        /*!\brief Variables corresponding to a spanwise Cosserat rod component
         * \ingroup cosserat_rod_component
         *
         * \details
         * CosseratRodSpanwiseGeometryVariables contains the definitions of
         * variables used within the Blocks framework for a Cosserat rod
         * data-structure spanning a single spatial dimension (i.e. along
         * centerline coordinates)
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         *
         * \see CosseratRodSpanwiseGeometry
         */
        template <typename CRT>
        class CosseratRodSpanwiseGeometryVariables {
         private:
          //**Type definitions**************************************************
          //! Traits type
          using Traits = CRT;
          //********************************************************************

         protected:
          //**Variable definitions**********************************************
          /*!\name Variable definitions*/
          //@{

          //**Initialized Variable definitions**********************************
          /*!\name Initialized Variable definitions*/
          //@{

          //********************************************************************
          /*!\brief Variable marking ReferenceElementLengths within the Cosserat
           * rod hierarchy
           */
          struct ReferenceElementLength
              : public Traits::template CosseratRodVariable<
                    ::elastica::tags::ReferenceElementLength,  //
                    typename Traits::DataType::Scalar,         //
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

          //**Computed Variable definitions*************************************
          /*!\name Computed Variable definitions*/
          //@{

          //********************************************************************
          /*!\brief Variable marking ReferenceCurvatures within the Cosserat
           * rod hierarchy (rest kappa)
           */
          struct ReferenceCurvature
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::ReferenceCurvature,  //
                    typename Traits::DataType::Vector,     //
                    typename Traits::Place::OnVoronoi> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking ReferenceShearStretchStrain within the
           * Cosserat rod hierarchy (sigma)
           */
          struct ReferenceShearStretchStrain
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::ReferenceShearStretchStrain,  //
                    typename Traits::DataType::Vector,              //
                    typename Traits::Place::OnElement> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking ReferenceVoronoiLengths within the Cosserat
           * rod hierarchy
           */
          struct ReferenceVoronoiLength
              : public Traits::template CosseratRodVariable<
                    ::elastica::tags::ReferenceVoronoiLength,  //
                    typename Traits::DataType::Scalar,         //
                    typename Traits::Place::OnVoronoi> {
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
          /*!\brief Variable marking ElementLength within the Cosserat rod
           * hierarchy
           */
          struct ElementLength : public Traits::template CosseratRodVariable<
                                     ::elastica::tags::ElementLength,    //
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

          //********************************************************************
          /*!\brief Variable marking ElementDilatation within the Cosserat
           * rod hierarchy
           */
          struct ElementDilatation
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::ElementDilatation,  //
                    typename Traits::DataType::Scalar,    //
                    typename Traits::Place::OnElement> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking VoronoiLength within the Cosserat rod
           * hierarchy
           */
          struct VoronoiLength : public Traits::template CosseratRodVariable<
                                     ::elastica::tags::VoronoiLength,    //
                                     typename Traits::DataType::Scalar,  //
                                     typename Traits::Place::OnVoronoi> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking VoronoiDilatation within the Cosserat
           * rod hierarchy
           */
          struct VoronoiDilatation
              : public Traits::template CosseratRodVariable<
                    ::elastica::tags::VoronoiDilatation,  //
                    typename Traits::DataType::Scalar,    //
                    typename Traits::Place::OnVoronoi> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking Curvature within the Cosserat rod hierarchy
           * (kapa)
           */
          struct Curvature : public Traits::template CosseratRodVariable<
                                 ::elastica::tags::Curvature,        //
                                 typename Traits::DataType::Vector,  //
                                 typename Traits::Place::OnVoronoi> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking Tangent within the Cosserat rod hierarchy
           */
          struct Tangent : public Traits::template CosseratRodVariable<
                               ::elastica::tags::Tangent,          //
                               typename Traits::DataType::Vector,  //
                               typename Traits::Place::OnElement> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking ShearStretchStrain within the Cosserat
           * rod hierarchy (sigma)
           */
          struct ShearStretchStrain
              : public Traits::template CosseratRodVariable<
                    ::elastica::tags::ShearStretchStrain,  //
                    typename Traits::DataType::Vector,     //
                    typename Traits::Place::OnElement> {};
          //********************************************************************

          //@}
          //********************************************************************

          //@}
          //********************************************************************

          //**Type definitions**************************************************
          //! List of computed variables
          using ComputedVariables =
              tmpl::list<ReferenceElementLength, ReferenceVoronoiLength,
                         ElementLength, VoronoiLength, VoronoiDilatation,
                         Curvature, Tangent, ShearStretchStrain>;
          //! List of initialized variables
          using InitializedVariables =
              tmpl::list<ElementDilatation, ReferenceCurvature,
                         ReferenceShearStretchStrain>;
          //! List of all variables
          using Variables =
              tmpl::append<InitializedVariables, ComputedVariables>;
          //********************************************************************
        };
        //**********************************************************************

        // TODO Check for EBO
        //======================================================================
        //
        //  CLASS DEFINITION
        //
        //======================================================================

        //**********************************************************************
        /*!\brief Component corresponding to a spanwise Cosserat rod with a
         * \ingroup cosserat_rod_component
         *
         * \details
         * CosseratRodSpanwiseGeometry implements an intermediate Geometry
         * component for use within Cosserat rods implemented in the Blocks
         * framework. It denotes a Cosserat rod data-structure without any
         * details on the shape of cross section spanning a
         * single spatial dimension (i.e. along centerline coordinates)
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         * \tparam ComputationalBlock The final block which is derived from the
         * current component
         *
         * \see CosseratRodSpanwiseGeometryVariables
         */
        template <typename CRT, typename ComputationalBlock>
        class CosseratRodSpanwiseGeometry
            : public RodSpanwiseGeometry<CRT, ComputationalBlock>,
              public CosseratRodSpanwiseGeometryVariables<CRT>,
              public CRTPHelper<ComputationalBlock,
                                CosseratRodSpanwiseGeometry> {
         private:
          //**Type definitions**************************************************
          //! Traits type
          using Traits = CRT;
          //! This type
          using This = CosseratRodSpanwiseGeometry<Traits, ComputationalBlock>;
          //! Parent type
          using Parent = RodSpanwiseGeometry<Traits, ComputationalBlock>;
          //! Type of Variable definitions
          using VariableDefinitions =
              CosseratRodSpanwiseGeometryVariables<Traits>;
          //! CRTP Type
          using CRTP =
              CRTPHelper<ComputationalBlock, CosseratRodSpanwiseGeometry>;
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
          //! Initialized variables
          using typename VariableDefinitions::ReferenceCurvature;
          using typename VariableDefinitions::ReferenceElementLength;
          using typename VariableDefinitions::ReferenceShearStretchStrain;
          //! Computed variables
          using typename VariableDefinitions::Curvature;
          using typename VariableDefinitions::ElementDilatation;
          using typename VariableDefinitions::ElementLength;
          using typename VariableDefinitions::ReferenceVoronoiLength;
          using typename VariableDefinitions::ShearStretchStrain;
          using typename VariableDefinitions::Tangent;
          using typename VariableDefinitions::VoronoiDilatation;
          using typename VariableDefinitions::VoronoiLength;
          //********************************************************************

          //********************************************************************
          /*!\copydoc ComponentInitializationDocStub
           */
          template <typename BlockLike, typename CosseratInitializer>
          static void initialize(
              CosseratRodSpanwiseGeometry<Traits, BlockLike>& this_component,
              CosseratInitializer&& initializer) {
            // 1. Initialize parent
            Parent::initialize(this_component,
                               std::forward<CosseratInitializer>(initializer));

            // 2. Initialize required variables
            initialize_component<
                typename VariableDefinitions::InitializedVariables>(
                this_component.self(),
                std::forward<CosseratInitializer>(initializer));

            // 3. Computed reference variables
            //  - ReferenceElementLength
            //  - ReferenceVoronoiLength
            { compute_reference_geometry(this_component.self()); }

            {
              update_spanwise_variables(this_component.self());
              update_shear_stretch_strain(this_component.self());
              update_curvature(this_component.self());
            }
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
          /*!\name element length methods*/
          //@{
          /*!\brief Gets element lengths of the current rod
           */
          inline constexpr decltype(auto) get_element_length() & noexcept {
            return blocks::get<::elastica::tags::ElementLength>(self());
          }
          inline constexpr decltype(auto) get_element_length() const& noexcept {
            return blocks::get<::elastica::tags::ElementLength>(self());
          }
          inline constexpr decltype(auto) get_element_length(
              index_type idx) & noexcept {
            return ElementLength::slice(get_element_length(), idx);
          }
          inline constexpr decltype(auto) get_element_length(
              index_type idx) const& noexcept {
            return ElementLength::slice(get_element_length(), idx);
          }
          //@}
          //********************************************************************

          //********************************************************************
          /*!\name reference element length methods*/
          //@{
          /*!\brief Gets reference element lengths of the current rod
           */
          inline constexpr decltype(auto)
          get_reference_element_length() & noexcept {
            return blocks::get<::elastica::tags::ReferenceElementLength>(
                self());
          }
          inline constexpr decltype(auto) get_reference_element_length()
              const& noexcept {
            return blocks::get<::elastica::tags::ReferenceElementLength>(
                self());
          }
          inline constexpr decltype(auto) get_reference_element_length(
              index_type idx) & noexcept {
            return ReferenceElementLength::slice(get_reference_element_length(),
                                                 idx);
          }
          inline constexpr decltype(auto) get_reference_element_length(
              index_type idx) const& noexcept {
            return ReferenceElementLength::slice(get_reference_element_length(),
                                                 idx);
          }
          //@}
          //********************************************************************

          //********************************************************************
          /*!\name reference curvature methods*/
          //@{
          /*!\brief Gets reference curvature of the current rod
           */
          inline constexpr decltype(auto) get_reference_curvature() & noexcept {
            return blocks::get<::elastica::tags::ReferenceCurvature>(self());
          }
          inline constexpr decltype(auto) get_reference_curvature()
              const& noexcept {
            return blocks::get<::elastica::tags::ReferenceCurvature>(self());
          }
          inline constexpr decltype(auto) get_reference_curvature(
              index_type idx) & noexcept {
            return ReferenceCurvature::slice(get_reference_curvature(), idx);
          }
          inline constexpr decltype(auto) get_reference_curvature(
              index_type idx) const& noexcept {
            return ReferenceCurvature::slice(get_reference_curvature(), idx);
          }
          //@}
          //********************************************************************

          //********************************************************************
          /*!\name reference shear_stretch_strain methods*/
          //@{
          /*!\brief Gets reference shear_strain of the current rod
           */
          inline constexpr decltype(auto)
          get_reference_shear_stretch_strain() & noexcept {
            return blocks::get<::elastica::tags::ReferenceShearStretchStrain>(
                self());
          }
          inline constexpr decltype(auto) get_reference_shear_stretch_strain()
              const& noexcept {
            return blocks::get<::elastica::tags::ReferenceShearStretchStrain>(
                self());
          }
          inline constexpr decltype(auto) get_reference_shear_stretch_strain(
              index_type idx) & noexcept {
            return ReferenceShearStretchStrain::slice(
                get_reference_shear_stretch_strain(), idx);
          }
          inline constexpr decltype(auto) get_reference_shear_stretch_strain(
              index_type idx) const& noexcept {
            return ReferenceShearStretchStrain::slice(
                get_reference_shear_stretch_strain(), idx);
          }
          //@}
          //********************************************************************

          //********************************************************************
          /*!\name tangent methods*/
          //@{
          /*!\brief Gets tangents of the current rod
           */
          inline constexpr decltype(auto) get_tangent() & noexcept {
            return blocks::get<::elastica::tags::Tangent>(self());
          }
          inline constexpr decltype(auto) get_tangent() const& noexcept {
            return blocks::get<::elastica::tags::Tangent>(self());
          }
          inline constexpr decltype(auto) get_tangent(
              index_type idx) & noexcept {
            return Tangent::slice(get_tangent(), idx);
          }
          inline constexpr decltype(auto) get_tangent(
              index_type idx) const& noexcept {
            return Tangent::slice(get_tangent(), idx);
          }
          //@}
          //********************************************************************

          //@}
          //********************************************************************
        };
        //**********************************************************************

      }  // namespace detail

    }  // namespace component

    //**********************************************************************
    /*!\brief Compute reference geometry
     * \ingroup cosserat_rod_custom_entries
     */
    template <typename Traits, typename ComputationalBlock>
    void compute_reference_geometry(
        component::detail::CosseratRodSpanwiseGeometry<
            Traits, ComputationalBlock>& block_like) COSSERATROD_LIB_NOEXCEPT {
      using ::elastica::tags::ElementDilatation;
      using ::elastica::tags::Position;
      using ::elastica::tags::ReferenceElementLength;
      using ::elastica::tags::ReferenceVoronoiLength;
      auto&& position(blocks::get<Position>(block_like.self()));
      auto&& dilatation(blocks::get<ElementDilatation>(block_like.self()));
      auto&& reference_length(
          blocks::get<ReferenceElementLength>(block_like.self()));
      auto&& reference_voronoi_length(
          blocks::get<ReferenceVoronoiLength>(block_like.self()));

      auto&& position_diff(Traits::Operations::difference_kernel(position));
      reference_length =
          Traits::Operations::batch_norm(position_diff) / dilatation;
      reference_voronoi_length =
          Traits::Operations::average_kernel(reference_length);
    }
    //**********************************************************************

    //**********************************************************************
    /*!\brief Updates rod spanwise geometric properties given state.
     * \ingroup cosserat_rod_custom_entries
     *
     * \details
     * Translational functions
     * Updates rod spanwise geometric properties (elemental lengths,
     * tangents and radii) from given nodal positions (state).
     * Ref: eq (3.3) from \cite gazzola2018forward. \n
     * length{i} = || position{i+1} - position{i} || \n
     * tangent{i} = (position{i+1} - position{i}) / length{i} \n
     *
     * The following members are modified:
     *  - length_batch(n_elems)
     *  - tangent_batch(3, n_elems)
     *  - radius_batch(n_elems)
     *
     * with the following inputs:
     *  - position_batch(3, n_nodes)
     *  - volume_batch(n_elems)
     *
     * \example
     * The following shows a typical use of the
     * compute_geometry_from_state()
     *
     * function with the expected (correct) result also shown.
     *
     * \return void/None
     *
     */
    template <typename Traits, typename ComputationalBlock>
    void update_spanwise_variables(
        component::detail::CosseratRodSpanwiseGeometry<
            Traits, ComputationalBlock>& block_like) COSSERATROD_LIB_NOEXCEPT {
      // Compute eq (3.3) from 2018 RSOS paper
      // 0. Create reference variables
      auto&& position(
          blocks::get<::elastica::tags::Position>(block_like.self()));
      auto&& element_length(
          blocks::get<tags::ElementLength>(block_like.self()));
      auto&& dilatation(
          blocks::get<tags::ElementDilatation>(block_like.self()));
      auto&& reference_length(
          blocks::get<tags::ReferenceElementLength>(block_like.self()));
      auto&& tangent(blocks::get<tags::Tangent>(block_like.self()));
      auto&& voronoi_length(
          blocks::get<::elastica::tags::VoronoiLength>(block_like.self()));
      auto&& voronoi_dilatation(
          blocks::get<tags::VoronoiDilatation>(block_like.self()));
      auto&& reference_voronoi_length(
          blocks::get<tags::ReferenceVoronoiLength>(block_like.self()));

      // 1. Compute new lengths
      auto&& position_diff(Traits::Operations::difference_kernel(position));
      element_length = Traits::Operations::batch_norm(position_diff);
      // +1e-14; if want to match pyelastica

      using ::blocks::fill_ghosts_for;
      fill_ghosts_for<tags::ElementLength>(block_like.self());

      // 2. Update tangents
      // before division by element_length, I fill ghosts so that
      // enable_floating_point exceptions do not throw a raise by 0 error
      Traits::Operations::batch_division_matvec(tangent, position_diff,
                                                element_length);

      // 3. Compute dilatation of elements
      dilatation = element_length / reference_length;

      // 4. Compute voronoi length
      voronoi_length = Traits::Operations::average_kernel(element_length);

      // 5. Compute dilatation of voronoi regions
      voronoi_dilatation = voronoi_length / reference_voronoi_length;
    }
    //**********************************************************************

    //**********************************************************************
    /*!\brief Update strains
     * \ingroup cosserat_rod_custom_entries
     */
    template <typename Traits, typename ComputationalBlock>
    void update_shear_stretch_strain(
        component::detail::CosseratRodSpanwiseGeometry<
            Traits, ComputationalBlock>& block_like) COSSERATROD_LIB_NOEXCEPT {
      auto&& shear_stretch_strains(
          blocks::get<::elastica::tags::ShearStretchStrain>(block_like.self()));
      auto&& element_dilatation(
          blocks::get<::elastica::tags::ElementDilatation>(block_like.self()));
      auto&& director(
          blocks::get<::elastica::tags::Director>(block_like.self()));
      auto&& tangent(blocks::get<::elastica::tags::Tangent>(block_like.self()));

      Traits::Operations::batch_matvec_scale(shear_stretch_strains, director,
                                             tangent, element_dilatation);
      Traits::Operations::batch_subtract_z_unit(shear_stretch_strains);
    }
    //**********************************************************************

    //**********************************************************************
    /*!\brief Update curvature
     * \ingroup cosserat_rod_custom_entries
     */
    template <typename Traits, typename ComputationalBlock>
    void update_curvature(component::detail::CosseratRodSpanwiseGeometry<
                          Traits, ComputationalBlock>& block_like)
        COSSERATROD_LIB_NOEXCEPT {
      auto&& curvature(
          blocks::get<::elastica::tags::Curvature>(block_like.self()));
      auto&& director(
          blocks::get<::elastica::tags::Director>(block_like.self()));
      auto&& reference_voronoi_length(
          blocks::get<::elastica::tags::ReferenceVoronoiLength>(
              block_like.self()));
      Traits::Operations::spanwise_inv_rotate_and_divide(
          curvature, director, reference_voronoi_length);
    }
    //**********************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

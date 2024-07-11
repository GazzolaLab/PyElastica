#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cstdint>  // for size_t
#include <utility>  // for move

#include "Simulator/Materials.hpp"
//
#include "Systems/Block.hpp"
//
#include "Utilities/CRTP.hpp"
#include "Utilities/Math/Vec3.hpp"
#include "Utilities/TMPL.hpp"

// module
/// Types always first
#include "Systems/CosseratRods/Components/Kinematics/Types.hpp"
///
#include "Systems/CosseratRods/Components/Initialization.hpp"
#include "Systems/CosseratRods/Components/Kinematics/Protocols.hpp"
#include "Systems/CosseratRods/Components/Kinematics/Tags.hpp"  // there is only one tags here
#include "Systems/CosseratRods/Components/Noexcept.hpp"
#include "Systems/CosseratRods/Components/helpers.hpp"

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
        /*!\brief Variables corresponding to kinematics of a span-wise rod
         * component within the Cosserat rod hierarchy
         * \ingroup cosserat_rod_component
         *
         * \details
         * RodKinematicsVariables contains the definitions of variables
         * used within the Blocks framework for the kinematics of a rod
         * data-structure spanning a single spatial dimension (i.e. along
         * center-line coordinates)
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         *
         * \see WithRodKinematics
         */
        template <typename CRT>
        class RodKinematicsVariables {
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
          /*!\brief Variable scalar mass within the Cosserat rod hierarchy
           *  Lumped mass on node. (should be in element)
           */
          struct Mass : public Traits::template CosseratRodInitializedVariable<
                            ::elastica::tags::Mass,             //
                            typename Traits::DataType::Scalar,  //
                            typename Traits::Place::OnNode> {
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
          /*!\brief Variable inverse scalar mass within the Cosserat rod
           * hierarchy Lumped mass on node. (should be in element)
           *
           * \note
           * Ghost is defaulted to 0.0 since it is always in the numerator
           */
          struct InvMass : public Traits::template CosseratRodVariable<
                               ::elastica::tags::InvMass,          //
                               typename Traits::DataType::Scalar,  //
                               typename Traits::Place::OnNode> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking material
           */
          struct Material
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::Material,  //
                    typename Traits::DataType::Index,
                    typename Traits::Place::OnElement> {
            // Doesnt need ghosting logic, 0 == iron is fine
          };
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking Velocity within the Cosserat rod hierarchy
           */
          struct Velocity
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::Velocity,         //
                    typename Traits::DataType::Vector,  //
                    typename Traits::Place::OnNode> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking AngularVelocity (\f$ \omega \f$) within the
           * Cosserat rod hierarchy
           */
          struct AngularVelocity
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::AngularVelocity,  //
                    typename Traits::DataType::Vector,
                    typename Traits::Place::OnElement> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking Acceleration within the Cosserat rod
           * hierarchy
           */
          struct Acceleration
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::Acceleration,     //
                    typename Traits::DataType::Vector,  //
                    typename Traits::Place::OnNode> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking AngularAcceleration within the
           * Cosserat rod hierarchy
           */
          struct AngularAcceleration
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::AngularAcceleration,  //
                    typename Traits::DataType::Vector,
                    typename Traits::Place::OnElement> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking external load within the Cosserat rod
           * hierarchy
           */
          struct ExternalLoads : public Traits::template CosseratRodVariable<
                                     ::elastica::tags::ExternalLoads,    //
                                     typename Traits::DataType::Vector,  //
                                     typename Traits::Place::OnNode> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking external torque within the Cosserat rod
           * hierarchy
           */
          struct ExternalTorques : public Traits::template CosseratRodVariable<
                                       ::elastica::tags::ExternalTorques,  //
                                       typename Traits::DataType::Vector,  //
                                       typename Traits::Place::OnElement> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking MassSecondMomentOfInertia within the
           * Cosserat rod hierarchy
           */
          struct MassSecondMomentOfInertia
              : public Traits::template CosseratRodVariable<
                    ::elastica::tags::MassSecondMomentOfInertia,  //
                    typename Traits::DataType::Matrix,
                    typename Traits::Place::OnElement> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking InvMassSecondMomentOfInertia within the
           * Cosserat rod hierarchy
           */
          struct InvMassSecondMomentOfInertia
              : public Traits::template CosseratRodVariable<
                    ::elastica::tags::InvMassSecondMomentOfInertia,  //
                    typename Traits::DataType::Vector,
                    typename Traits::Place::OnElement> {};
          //********************************************************************

          //@}
          //********************************************************************

          //**Type definitions**************************************************
          //! List of computed variables
          using ComputedVariables =
              tmpl::list<Mass, InvMass, ExternalLoads, ExternalTorques,
                         MassSecondMomentOfInertia,
                         InvMassSecondMomentOfInertia>;
          //! List of initialized variables
          using InitializedVariables =
              tmpl::list<Material, Velocity, AngularVelocity, Acceleration,
                         AngularAcceleration>;
          //! List of all variables
          using Variables =
              tmpl::append<InitializedVariables, ComputedVariables>;
          //********************************************************************
        };
        //**********************************************************************

      }  // namespace detail

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Kinematics component corresponding to a spanwise rod
       * \ingroup cosserat_rod_component
       *
       * \details
       * WithRodKinematics implements the final Kinematics component for use
       * with Cosserat rods implemented in the Blocks framework. It denotes the
       * kinematic variables and methods (anything temporally varying) needed
       * for a Cosserat rod data-structure, as defined by the Geometry
       * component.
       *
       * \note
       * It requires a valid Geometry component declared in the Blocks
       * Hierarchy to ensure it is properly used. Else, a compilation error is
       * thrown.
       *
       * \usage
       * Since WithRodKinematics is a valid and complete Kinematics component
       * adhering to protocols::Kinematics, one can use it to declare a
       * CosseratRodPlugin within the @ref blocks framework
       * \code
       * // pre-declare RodTraits, Blocks
       * using CircularCosseratRod = CosseratRodPlugin<RodTraits, Block,
       * // Conforms to protocols::Geometry1D!
       * components::WithCircularCosseratRod,
       * components::WithRodKinematics>;
       * \endcode
       *
       * \tparam CRT A valid Cosserat Rod Traits class
       * \tparam ComputationalBlock The final block which is derived from the
       * current component
       *
       * \see CosseratRodPlugin, RodKinematicsVariables
       */
      // Forward declare all Layers and policies
      template <typename CRT, typename ComputationalBlock>
      class WithRodKinematics
          : public CRTPHelper<ComputationalBlock, WithRodKinematics>,
            public detail::RodKinematicsVariables<CRT>,
            public ::tt::ConformsTo<protocols::Kinematics1D>,
            public KinematicsComponent<
                WithRodKinematics<CRT, ComputationalBlock>> {
       private:
        //**Type definitions****************************************************
        //! Traits type
        using Traits = CRT;
        //! This type
        using This = WithRodKinematics<Traits, ComputationalBlock>;
        //! Type of Variable definitions
        using VariableDefinitions = detail::RodKinematicsVariables<Traits>;
        //! CRTP Type
        using CRTP = CRTPHelper<ComputationalBlock, WithRodKinematics>;
        //! Real number type
        using real_type = typename Traits::real_type;
        //! Index type
        using index_type = typename Traits::index_type;
        //**********************************************************************

       protected:
        //**Type definitions****************************************************
        //! List of computed variables
        using typename VariableDefinitions::ComputedVariables;
        //! List of initialized variables
        using typename VariableDefinitions::InitializedVariables;
        //! List of all variables
        using typename VariableDefinitions::Variables;

        //! Current Variable definitions
        using typename VariableDefinitions::Acceleration;
        using typename VariableDefinitions::AngularAcceleration;
        using typename VariableDefinitions::AngularVelocity;
        using typename VariableDefinitions::ExternalLoads;
        using typename VariableDefinitions::ExternalTorques;
        using typename VariableDefinitions::InvMass;
        using typename VariableDefinitions::InvMassSecondMomentOfInertia;
        using typename VariableDefinitions::Mass;
        using typename VariableDefinitions::MassSecondMomentOfInertia;
        using typename VariableDefinitions::Material;
        using typename VariableDefinitions::Velocity;
        //**********************************************************************

        //********************************************************************
        /*!\copydoc ComponentInitializationDocStub
         */
        template <typename BlockLike, typename CosseratInitializer>
        static void initialize(
            WithRodKinematics<Traits, BlockLike>& this_component,
            CosseratInitializer&& initializer) {
          // 1. Initialize required variables
          initialize_component<
              typename VariableDefinitions::InitializedVariables>(
              this_component.self(),
              std::forward<CosseratInitializer>(initializer));

          auto const n_elem = blocks::get<elastica::tags::NElement>(
              cpp17::as_const(initializer))();

          /* mass second moment of inertia */
          {
            using Variable = MassSecondMomentOfInertia;
            using Tag = blocks::parameter_t<Variable>;

            auto&& variable(blocks::get<Tag>(this_component.self()));

            // already initialized
            auto&& material(blocks::get<tags::Material>(this_component.self()));
            auto&& reference_lengths(blocks::get<tags::ReferenceElementLength>(
                this_component.self()));

            const auto dofs = Variable::get_dofs(n_elem);
            for (std::size_t idx = 0U; idx < dofs; ++idx) {
              // (I_1, I_2, I_3)
              Variable::slice(variable, idx) = real_type(0.0);
              const Vec3 I_idx =
                  this_component.self().get_second_moment_of_area(idx) *
                  ::elastica::Material::get_density(
                      Traits::DataType::Index::slice(material, idx)) *
                  Traits::DataType::Scalar::slice(reference_lengths, idx);
              Variable::diagonal_assign(variable, idx, I_idx);
            }
          }

          /* inverse mass second moment of inertia */
          {
            using Variable = InvMassSecondMomentOfInertia;
            using Tag = blocks::parameter_t<Variable>;

            auto&& variable(blocks::get<Tag>(this_component.self()));
            auto&& material(blocks::get<tags::Material>(this_component.self()));
            auto&& reference_lengths(blocks::get<tags::ReferenceElementLength>(
                this_component.self()));

            const auto dofs = Variable::get_dofs(n_elem);
            for (std::size_t idx = 0U; idx < dofs; ++idx) {
              // inv(I_1, I_2, I_3)
              const Vec3 I_idx =
                  real_type(1.0) /
                  (this_component.self().get_second_moment_of_area(idx) *
                   ::elastica::Material::get_density(
                       Traits::DataType::Index::slice(material, idx)) *
                   Traits::DataType::Scalar::slice(reference_lengths, idx));
              Variable::slice(variable, idx) = I_idx;
              // Variable::diagonal_assign(variable, idx, I_idx);
            }
          }

          // Variables Initialization
          {
            // using MassVariable = typename Traits::DataType::Scalar;

            auto&& mass(
                blocks::get<::elastica::tags::Mass>(this_component.self()));

            mass = 0.0;  // clear
            for (std::size_t idx = 0UL; idx < n_elem; ++idx) {
              typename Traits::real_type mass_on_element =
                  ::elastica::Material::get_density(
                      this_component.self().get_material(idx)) *
                  this_component.self().get_element_volume(idx);
              Mass::slice(mass, idx) += 0.5 * mass_on_element;
              Mass::slice(mass, idx + 1UL) += 0.5 * mass_on_element;
            }
          }

          {
            using Variable = InvMass;
            using Tag = blocks::parameter_t<Variable>;

            auto&& inv_mass(blocks::get<Tag>(this_component.self()));
            auto&& mass(
                blocks::get<::elastica::tags::Mass>(this_component.self()));
            const auto dofs = Variable::get_dofs(n_elem);

            inv_mass = real_type(0.0);  // clear
            for (std::size_t idx = 0UL; idx < dofs; ++idx) {
              Variable::slice(inv_mass, idx) =
                  real_type(1.0) / Mass::slice(mass, idx);
            }
          }

          /* Reset forces and torques */
          {
            auto&& external_loads(blocks::get<::elastica::tags::ExternalLoads>(
                this_component.self()));
            auto&& external_torques(
                blocks::get<::elastica::tags::ExternalTorques>(
                    this_component.self()));
            external_loads = 0.0;
            external_torques = 0.0;
          }
        }
        //**********************************************************************

       public:
        //**CRTP method*********************************************************
        /*!\name CRTP method*/
        //@{
        using CRTP::self;
        //@}
        //**********************************************************************

        //**Get methods*********************************************************
        /*!\name Get methods*/
        //@{

#define STAMP_GETTERS(Var, func)                                         \
  inline constexpr decltype(auto) func()& noexcept {                     \
    return blocks::get<::elastica::tags::Var>(self());                   \
  }                                                                      \
  inline constexpr decltype(auto) func() const& noexcept {               \
    return blocks::get<::elastica::tags::Var>(self());                   \
  }                                                                      \
  inline constexpr decltype(auto) func(index_type idx)& noexcept {       \
    return Var::slice(func(), idx);                                      \
  }                                                                      \
  inline constexpr decltype(auto) func(index_type idx) const& noexcept { \
    return Var::slice(func(), idx);                                      \
  }

        //**********************************************************************
        /*!\name Velocity methods*/
        //@{
        /*!\brief Gets velocity of the current rod
         */
        STAMP_GETTERS(Velocity, get_velocity)
        //@}
        //**********************************************************************

        //**********************************************************************
        /*!\name Angular Velocity methods*/
        //@{
        /*!\brief Gets angular velocity of the current rod
         */
        STAMP_GETTERS(AngularVelocity, get_angular_velocity)
        //@}
        //**********************************************************************

        //**********************************************************************
        /*!\name Aceeleration methods*/
        //@{
        /*!\brief Gets acceleration of the current rod
         */
        STAMP_GETTERS(Acceleration, get_acceleration)
        //@}
        //**********************************************************************

        //**********************************************************************
        /*!\name Angular acceleration methods*/
        //@{
        /*!\brief Gets angular acceleration of the current rod
         */
        STAMP_GETTERS(AngularAcceleration, get_angular_acceleration)
        //@}
        //**********************************************************************

        //**********************************************************************
        /*!\name Material methods*/
        //@{
        /*!\brief Gets material of the current rod
         */
        STAMP_GETTERS(Material, get_material)
        //@}
        //**********************************************************************

        //**********************************************************************
        /*!\name Mass methods*/
        //@{
        /*!\brief Gets mass of the current rod
         */
        STAMP_GETTERS(Mass, get_mass)
        //@}
        //**********************************************************************

        //**********************************************************************
        /*!\name External Loads methods*/
        //@{
        /*!\brief Gets external loads of the current rod
         */
        STAMP_GETTERS(ExternalLoads, get_external_loads)
        //@}
        //**********************************************************************

        //**********************************************************************
        /*!\name External Torques methods*/
        //@{
        /*!\brief Gets external torques of the current rod
         */
        STAMP_GETTERS(ExternalTorques, get_external_torques)
        //@}
        //**********************************************************************

        //**********************************************************************
        /*!\name Mass SMOI methods*/
        //@{
        /*!\brief Gets mass second moment of inertia of the current rod
         */
        STAMP_GETTERS(MassSecondMomentOfInertia,
                      get_mass_second_moment_of_inertia)
        //@}
        //**********************************************************************

        //**********************************************************************
        /*!\name Inverse Mass SMOI methods*/
        //@{
        /*!\brief Gets inverse fo mass second moment of inertia of the current
         * rod
         */
        STAMP_GETTERS(InvMassSecondMomentOfInertia,
                      get_inverse_mass_second_moment_of_inertia)
        //@}
        //**********************************************************************

#undef STAMP_GETTERS

        //@}
        //**********************************************************************
      };
      //************************************************************************

      // clang-format off
//******************************************************************************
/*!\brief Documentation stub with tags of WithRodKinematics
 * \ingroup cosserat_rod_component
 *
| Kinematics Variables           ||
|--------------------------------|------------------------------------------------------------------------------------------------------------------|
| On Nodes    (`n_elements+1`)   | elastica::tags::Acceleration, elastica::tags::ExternalLoads, elastica::tags::Mass, elastica::tags::Velocity      |
| On Elements (`n_elements`)     | elastica::tags::AngularVelocity, elastica::tags::AngularAcceleration,                                            |
|^                               | elastica::tags::ExternalTorques, elastica::tags::MassSecondMomentOfInertia, elastica::tags::Material             |
|^                               | elastica::tags::InvMassSecondMomentOfInertia                                                                     |
*/
      template <typename CRT, typename ComputationalBlock>
      using WithRodKinematicsTagsDocsStub = WithRodKinematics<CRT, ComputationalBlock>;
//******************************************************************************
      // clang-format on

    }  // namespace component

    //**************************************************************************
    /*!\brief Gets temporal rate of dilatation based on the current state
     * of the current rod
     * \ingroup cosserat_rod_custom_entries
     *
     * \details
     * Computes rate of change of elemental dilatations (de / dt) given the
     * rod positions and velocities (state). \n
     * dilatation_rate{i} =
     * (position{i+1} - position{i}) dot
     * (velocity{i+1} - velocity{i}) /
     * (length{i} * rest_length{i}) \n
     * Assumes compute_all_dilatations() has been called.
     *
     * \return dilatation rate
     */
    template <typename Traits, typename ComputationalBlock>
    decltype(auto) get_dilatation_rate(
        component::WithRodKinematics<Traits, ComputationalBlock> const&
            block_like) COSSERATROD_LIB_NOEXCEPT {
      auto&& position(blocks::get<tags::Position>(block_like.self()));
      auto&& length(blocks::get<tags::ElementLength>(block_like.self()));
      auto&& reference_length(
          blocks::get<tags::ReferenceElementLength>(block_like.self()));
      auto&& velocity(blocks::get<tags::Velocity>(block_like.self()));

      return Traits::Operations::batch_dot(
                 Traits::Operations::difference_kernel(position),
                 Traits::Operations::difference_kernel(velocity)) /
             (length * reference_length);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Gets temporal rate of curvature based on the current state
     * of the current rod
     *
     * TODO get curvature rate
     */
    template <typename Traits, typename ComputationalBlock>
    decltype(auto) get_curvature_rate(
        component::WithRodKinematics<Traits, ComputationalBlock> const&
            block_like) COSSERATROD_LIB_NOEXCEPT {
      auto&& curvature(blocks::get<tags::Curvature>(block_like.self()));
      auto&& omega(blocks::get<tags::AngularAcceleration>(block_like.self()));
      auto&& averaged_omega = Traits::Operations::average_kernel(omega);

      return Traits::Operations::difference_kernel(omega) +
             Traits::Operations::batch_cross(curvature, averaged_omega);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Gets temporal rate of curvature based on the current state
     * of the current rod
     * TODO get shear stretch strain rate
     */
    template <typename Traits, typename ComputationalBlock>
    decltype(auto) get_shear_stretch_strain_rate(
        component::WithRodKinematics<Traits, ComputationalBlock> const&
            block_like) COSSERATROD_LIB_NOEXCEPT {
      auto&& director(blocks::get<tags::Director>(block_like.self()));
      auto&& velocity(blocks::get<tags::Velocity>(block_like.self()));
      auto&& reference_length(
          blocks::get<tags::ReferenceElementLength>(block_like.self()));
      auto&& sigma(blocks::get<tags::ShearStretchStrain>(block_like.self()));
      auto&& omega(blocks::get<tags::AngularAcceleration>(block_like.self()));

      auto& sigma_rate(sigma);
      auto&& diff_velocity = Traits::Operations::difference_kernel(velocity);
      Traits::Operations::batch_add_z_unit(sigma_rate);
      // z_vector = np.array([ 0.0, 0.0, 1.0 ]).reshape(3, -1);
      return Traits::Operations::batch_cross(sigma_rate, omega) -
             Traits::Operations::batch_matvec(director, diff_velocity) /
                 reference_length;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Update acceleration based on the current state
     *
     * \details
     * Updates accelerations of the system (translational and angular) given
     * the state of the system. \n
     * acceleration_batch{i} =
     * (internal_force_batch{i} + external_force_batch{i})
     * / mass_batch{i} \n
     * alpha_batch{i} = inv_mass_second_moment_of_inertia_batch{i} *
     * (internal_torque_batch{i} + external_torque_batch{i}) *
     * dilatation_batch{i} \n
     * Also resets external_force_batch and external_torque_batch = 0.0
     *
     * \example
     * The following shows a typical use of the update_acceleration()
     * function with the expected (correct) result also shown.
     * \snippet test_gov_eqns.cpp update_acceleration_example
     *
     * \return void/None
     *
     * \see fill later?
     */
    template <typename Traits, typename ComputationalBlock>
    void update_acceleration(
        component::WithRodKinematics<Traits, ComputationalBlock>& block_like)
        COSSERATROD_LIB_NOEXCEPT {
      // Immutable
      auto&& inv_mass(
          blocks::get<tags::InvMass>(cpp17::as_const(block_like).self()));
      auto&& dilatation(blocks::get<tags::ElementDilatation>(
          cpp17::as_const(block_like).self()));
      auto&& inv_mass_SMOI(blocks::get<tags::InvMassSecondMomentOfInertia>(
          cpp17::as_const(block_like).self()));
      auto&& internal_loads(
          blocks::get<tags::InternalLoads>(cpp17::as_const(block_like).self()));
      auto&& internal_torques(blocks::get<tags::InternalTorques>(
          cpp17::as_const(block_like).self()));
      auto&& external_loads(
          blocks::get<tags::ExternalLoads>(cpp17::as_const(block_like).self()));
      auto&& external_torques(blocks::get<tags::ExternalTorques>(
          cpp17::as_const(block_like).self()));

      // Mutable
      auto&& acceleration(blocks::get<tags::Acceleration>(block_like.self()));
      auto&& angular_acceleration(
          blocks::get<tags::AngularAcceleration>(block_like.self()));

      Traits::Operations::batch_multiplication_matvec(
          acceleration, (internal_loads + external_loads), inv_mass);

      Traits::Operations::batch_multiplication_matvec(
          angular_acceleration,
          inv_mass_SMOI % (internal_torques + external_torques), dilatation);
    }
    //*********************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

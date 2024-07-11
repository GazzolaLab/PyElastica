#pragma once

//******************************************************************************
// Includes
//******************************************************************************
///
#include "Systems/CosseratRods/Components/Elasticity/detail/Types.hpp"
///
#include "Systems/Block.hpp"
//
#include "Systems/common/Tags.hpp"
//
#include "Systems/CosseratRods/Components/Elasticity/detail/Tags/LinearHyperElasticityFacadeTags.hpp"
#include "Systems/CosseratRods/Components/Geometry/detail/Tags/CosseratRodSpanwiseGeometryTags.hpp"
#include "Systems/CosseratRods/Components/Noexcept.hpp"
//
#include "Utilities/Math/Vec3.hpp"
#include "Utilities/TMPL.hpp"

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
        /*!\brief Variables corresponding to facade for linear hyper-elastic
         * models
         * \ingroup cosserat_rod_component
         *
         * \details
         * LinearHyperElasticityFacadeVariables contains the definitions of
         * variables used within the Blocks framework for convenient
         * implementation of new linear hyperelastic models.
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         *
         * \see LinearHyperElasticityFacade
         */
        template <typename CRT>
        class LinearHyperElasticityFacadeVariables {
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
          /*!\brief Variable marking the BendingTwistRigidityMatrix
           * \f$ \hat{B} \f$ within the Cosserat rod hierarchy
           */
          struct BendingTwistRigidityMatrix
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::BendingTwistRigidityMatrix,  //
                    typename Traits::DataType::Vector,             //
                    typename Traits::Place::OnVoronoi> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking the ShearStretchRigidityMatrix
           * \f$ \hat{S} \f$ within the Cosserat rod hierarchy
           */
          struct ShearStretchRigidityMatrix
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::ShearStretchRigidityMatrix,  //
                    typename Traits::DataType::Vector,             //
                    typename Traits::Place::OnElement> {};
          //********************************************************************

          //@}
          //********************************************************************

          //**Type definitions**************************************************
          //! List of computed variables
          using ComputedVariables = tmpl::list<>;
          //! List of initialized variables
          using InitializedVariables = tmpl::list<BendingTwistRigidityMatrix,
                                                  ShearStretchRigidityMatrix>;
          //! List of all variables
          using Variables =
              tmpl::append<InitializedVariables, ComputedVariables>;
          //********************************************************************
        };
        //**********************************************************************

        //======================================================================
        //
        //  CLASS DEFINITION
        //
        //======================================================================

        //**********************************************************************
        /*!\brief  Facade for convenient implementation of linear hyper-elastic
         * models
         * \ingroup cosserat_rod_component
         *
         * \details
         * LinearHyperElasticityFacade is a helper for conveniently implementing
         * new linear hyper-elastic models for use within Cosserat rods
         * implemented in the Blocks framework. It sets up common variables and
         * methods (that interface with the Geometry component) for a linear
         * hyper-elastic model that operates on a 1-D parametrized entity
         * spanning a single spatial dimension (i.e. along centerline
         * coordinates, such as a Cosserat rod data-structure) and gives the
         * loads and torques.
         *
         * \usage
         * Since LinearHyperElasticityFacade is useful for creating new
         * linear hyper-elastic modes with an interface, one should use it using
         * by deriving from this class, as shown below
         *
         * \code
         * template <typename Traits, typename Block>
         * class MyCustomLinearHyperElasticityModel
         * : public LinearHyperElasticityFacade<Traits, Block> {
         *   // ...rest of the implementation
         * };
         * \endcode
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         * \tparam ComputationalBlock The final block which is derived from the
         * current component
         *
         * \see LinearHyperElasticityFacadeVariables
         */
        template <typename CRT, typename ComputationalBlock>
        class LinearHyperElasticityFacade
            : public LinearHyperElasticityFacadeVariables<CRT>,
              public CRTPHelper<ComputationalBlock,
                                LinearHyperElasticityFacade> {
         private:
          //**Type definitions**************************************************
          //! Traits type
          using Traits = CRT;
          //! This type
          using This = LinearHyperElasticityFacade<Traits, ComputationalBlock>;
          //! Type defining variables
          using VariableDefinitions = LinearHyperElasticityFacadeVariables<CRT>;
          //! CRTP Type
          using CRTP =
              CRTPHelper<ComputationalBlock, LinearHyperElasticityFacade>;
          //! Index type
          using index_type = typename Traits::index_type;
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
          /*!\name bending-twist methods*/
          //@{
          /*!\brief Gets get_bending_twist_rigidity_matrix of the current rod
           */
          inline constexpr decltype(auto)
          get_bending_twist_rigidity_matrix() & noexcept {
            return blocks::get<::elastica::tags::BendingTwistRigidityMatrix>(
                self());
          }
          inline constexpr decltype(auto) get_bending_twist_rigidity_matrix()
              const& noexcept {
            return blocks::get<::elastica::tags::BendingTwistRigidityMatrix>(
                self());
          }
          inline constexpr decltype(auto) get_bending_twist_rigidity_matrix(
              index_type idx) & noexcept {
            return BendingTwistRigidityMatrix::slice(
                get_bending_twist_rigidity_matrix(), idx);
          }
          inline constexpr decltype(auto) get_bending_twist_rigidity_matrix(
              index_type idx) const& noexcept {
            return BendingTwistRigidityMatrix::slice(
                get_bending_twist_rigidity_matrix(), idx);
          }
          //@}
          //********************************************************************

          //********************************************************************
          /*!\name shear-stretch methods*/
          //@{
          /*!\brief Gets get_shear_stretch_rigidity_matrix of the current rod
           */
          inline constexpr decltype(auto)
          get_shear_stretch_rigidity_matrix() & noexcept {
            return blocks::get<::elastica::tags::ShearStretchRigidityMatrix>(
                self());
          }
          inline constexpr decltype(auto) get_shear_stretch_rigidity_matrix()
              const& noexcept {
            return blocks::get<::elastica::tags::ShearStretchRigidityMatrix>(
                self());
          }
          inline constexpr decltype(auto) get_shear_stretch_rigidity_matrix(
              index_type idx) & noexcept {
            return ShearStretchRigidityMatrix::slice(
                get_shear_stretch_rigidity_matrix(), idx);
          }
          inline constexpr decltype(auto) get_shear_stretch_rigidity_matrix(
              index_type idx) const& noexcept {
            return ShearStretchRigidityMatrix::slice(
                get_shear_stretch_rigidity_matrix(), idx);
          }
          //@}
          //********************************************************************

          //@}
          //********************************************************************

         protected:
          //**Type definitions**************************************************
          //! List of computed variables
          using typename VariableDefinitions::ComputedVariables;
          //! List of initialized variables
          using typename VariableDefinitions::InitializedVariables;
          //! List of all variables
          using typename VariableDefinitions::Variables;
          //! Variables from definitions
          using typename VariableDefinitions::BendingTwistRigidityMatrix;
          using typename VariableDefinitions::ShearStretchRigidityMatrix;
          //********************************************************************

         protected:
          //********************************************************************
          /*!\copydoc ComponentInitializationDocStub
           */
          template <typename BlockLike, typename CosseratInitializer>
          static void initialize(
              LinearHyperElasticityFacade<Traits, BlockLike>& this_component,
              CosseratInitializer&& initializer) {
            // 1. Initialize generators first

            // In this case, the generator expressions are customized to only
            // provide E or B, the modulus of elasticity. We are responsible
            // for putting in the correct are etc. from the expression.
            auto const n_elem = blocks::get<elastica::tags::NElement>(
                cpp17::as_const(initializer))();

            // bending and twisting only
            {
              using Variable = BendingTwistRigidityMatrix;
              using ScalarVariable = typename Traits::DataType::Scalar;
              using Tag = blocks::parameter_t<Variable>;

              auto&& variable(blocks::get<Tag>(this_component.self()));
              auto&& stiffness_along_principal_directions_initializer =
                  blocks::get<Tag>(std::move(initializer));

              auto&& rest_length(
                  blocks::get<elastica::tags::ReferenceElementLength>(
                      this_component.self()));

              auto const dofs = Variable::get_dofs(n_elem);
              auto stiffness =
                  stiffness_along_principal_directions_initializer(0);
              auto MOI = this_component.self().get_second_moment_of_area(0);
              auto len = ScalarVariable::slice(rest_length, 0);

              for (std::size_t idx = 0U; idx < dofs; ++idx) {
                // (E, E, G) * (I_1, I_2, I_3)
                auto const prev_stiffness = stiffness;
                auto const prev_MOI = MOI;
                auto const prev_len = len;

                stiffness =
                    stiffness_along_principal_directions_initializer(idx + 1);
                MOI = this_component.self().get_second_moment_of_area(idx + 1);
                len = ScalarVariable::slice(rest_length, idx + 1);

                Variable::slice(variable, idx) =
                    (prev_len * prev_stiffness * prev_MOI +
                     len * stiffness * MOI) /
                    (prev_len + len);
              }
            }

            // shear and stretch only
            {
              using Variable = ShearStretchRigidityMatrix;
              using Tag = blocks::parameter_t<Variable>;

              auto&& variable(blocks::get<Tag>(this_component.self()));
              auto&& stiffness_along_principal_directions_initializer =
                  blocks::get<Tag>(std::move(initializer));

              auto const dofs = Variable::get_dofs(n_elem);
              for (std::size_t idx = 0U; idx < dofs; ++idx) {
                // (G, G, E) * (\alpha A, \alpha A, A)
                Variable::slice(variable, idx) =
                    stiffness_along_principal_directions_initializer(idx) *
                    compute_effective_area_along_principal_directions(
                        this_component.self(), idx);
              }
            }
          }
          //********************************************************************
        };
        //**********************************************************************

      }  // namespace detail

    }  // namespace component

    //**************************************************************************
    /*!\name Area along principal direction methods
     * \ingroup cosserat_rod_custom_entries
     *
     *!\brief Gets effective area along principal directions of the
     * current rod
     */
    template <typename Traits, typename ComputationalBlock>
    auto compute_effective_area_along_principal_directions(
        component::detail::LinearHyperElasticityFacade<
            Traits, ComputationalBlock>& block_like,
        std::size_t i) COSSERATROD_LIB_NOEXCEPT->Vec3 {
      return Vec3{real_t(block_like.self().get_area(i) *
                         block_like.self().shape_factor()),
                  real_t(block_like.self().get_area(i) *
                         block_like.self().shape_factor()),
                  real_t(block_like.self().get_area(i))};
    }
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

/*
 *
* Elasticity layer
- Prerequisites : GeometryLayer
[ ] rest_B
[ ] rest_S
[ ] internal_model_loads_ (n_l)
[ ] internal_model_couples_ (tau_l)
 */

/*
 * The following CRTP relies on a trick. Normally one uses
 * CRTP like the following:
 *
 * template <typename D>
 * class Base{
 * //...impl details
 * };
 *
 * class Derived : Base<Derived>{
 *  // impl details
 * };
 *
 * In case of multiple inheritance then two patterns exist:
 * 1. Cascade
 * template <typename D>
 * class Base{
 * //...impl details
 * };
 *
 * template <typename T>
 * class MiddleDerived : Base<MiddleDerived<T>>{
 *  // impl details
 * };
 *
 * template <typename Block>
 * class Derived : MiddleDerived<Derived<Block>>{
 *  // impl details
 * }
 *
 * class Block : Derived<Block>{
 *
 * };
 *
 * This way base has access to middle derived's functionality which
 * in turn has access to Derived's functionality.
 *
 * Advantages:
 * - this pattern marks Derived like a "final" class : no additional
 * derivation is possible and hence the hierarchy is "complete". There
 * is only one way to use it correctly.
 *
 *
 * Disadvantages:
 * - One downside is that middle derived cannot be used separately
 * if its coded for taking in Derived. (this is the patten used in
 * CrossSectionBase, for example, since CrossSectionBase is
 * like an ABC and shouldnt be used in the final block). To overcome
 * this one can use T as a dummy parameter (which
 * can be defaulted).
 * - To resolve the final
 * functions too additional code is needed (to forward to the correct
 * class).
 *
 *
 * 2. Concrete
 * template <typename D>
 * class Base{
 * //...impl details
 * };
 *
 * template <typename B>
 * class MiddleDerived : Base<B>{
 *  // impl details
 * };
 *
 * template <typename Block> // This is the block
 * class Derived : MiddleDerived<Block>{
 *  // impl details
 * }
 *
 * class Block : Derived<Block>{
 * };
 *
 * This way base has access to middle derived' and derived's overriden
 * functionality (if any), throgh the final impl class Block,
 * thus making it better in terms of code reduce.
 *
 * Advantages:
 * 1. Final implementation avaialbe in the top level class : so it
 * looks like a conventional CRTP hierarchy.
 *
 * Disadvantages:
 * 1. More tightly coupled code to the Block : i.e. we need a Block
 * class to do anything. This is not that disadvantageous as Block
 * is needed for a concrete implementation of CRTP anyway.
 * 2. Easy to extend and misuse.
 * i.e. one can easily make a
 *
 * template <tpyename B>
 * NewDerived : Derived<B>{
 *  // crazy functionality
 * };
 *
 * While this seems beneficial from OCP, this gives too much freedom.
 * Quoting Sutter,
 * However, in this case one has to distinguish between "protecting
 * against Murphy, versus protecting against Machiavelli". However,
 * whereas the intentional misuse cannot be prevented (Machiavelli), the
 * accidental misuse via inheritance is rather improbable (Murphy). aka
 * people need to know about CRTP first. Techniques are possible to
 * prevent deriving
 * https://stackoverflow.com/questions/18174441/crtp-and-multilevel-inheritance
 * but is overkill at this point.
 *
 * NICE: There's a simple fix for this, just by making the
 * implementation private and not giving access to the final classes, we
 * can still make it closed for extension.
 *
 */
// For all purposes, this behaves as an ABC derived from another ABC
// instead of enforcing protocols

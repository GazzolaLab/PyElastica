#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cstdint>  // for size_t
#include <utility>  // for move

#include "Systems/Block.hpp"
#include "Utilities/CRTP.hpp"
#include "Utilities/TMPL.hpp"

// module
/// Types always first
#include "Systems/CosseratRods/Components/Geometry/detail/Types.hpp"
///
#include "Systems/CosseratRods/Components/Geometry/detail/Tags/RodSpanwiseGeometryTags.hpp"
#include "Systems/CosseratRods/Components/Initialization.hpp"

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
        /*!\brief Variables corresponding to a spanwise rod component
         * \ingroup cosserat_rod_component
         *
         * \details
         * RodSpanwiseGeometryVariables contains the definitions of variables
         * used within the Blocks framework for a rod data-structure spanning a
         * single spatial dimension (i.e. along centerline coordinates)
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         *
         * \see RodSpanwiseGeometry
         */
        template <typename CRT>
        class RodSpanwiseGeometryVariables {
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
          /*!\brief Variable marking Positions within the Cosserat rod hierarchy
           */
          struct Position
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::Position,         //
                    typename Traits::DataType::Vector,  //
                    typename Traits::Place::OnNode> {};
          //********************************************************************

          //********************************************************************
          /*!\brief Variable marking Director(Q) within the Cosserat rod
           * hierarchy
           */
          struct Director
              : public Traits::template CosseratRodInitializedVariable<
                    ::elastica::tags::Director,  //
                    typename Traits::DataType::Matrix,
                    typename Traits::Place::OnElement> {
            //**Type definitions************************************************
            //! The element type of a ghost
            using ghost_type = typename Traits::DataType::Matrix::ghost_type;
            //******************************************************************

            static inline auto ghost_value() noexcept -> ghost_type {
              using Real = typename Traits::DataType::Scalar::ghost_type;
              // Identity is too perfect, lets have something more approximate
              return ghost_type{{Real{0.8}, Real{+0.6}, Real{0.0}},
                                {Real{0.0}, Real{+0.0}, Real{1.0}},
                                {Real{0.6}, Real{-0.8}, Real{0.0}}};
            }
          };
          //********************************************************************

          //@}
          //********************************************************************

          //**Type definitions**************************************************
          //! List of computed variables
          using ComputedVariables = tmpl::list<>;
          //! List of initialized variables
          using InitializedVariables = tmpl::list<Position, Director>;
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
        /*!\brief Component corresponding to a spanwise rod
         * \ingroup cosserat_rod_component
         *
         * \details
         * RodSpanwiseGeometry implements a part of Geometry component
         * for use within Cosserat rods implemented in the Blocks framework. It
         * denotes a rod data-structure spanning a single spatial dimension
         * (i.e. along centerline coordinates)
         *
         * \tparam CRT A valid Cosserat Rod Traits class
         * \tparam ComputationalBlock The final block which is derived from the
         * current component
         *
         * \see RodSpanwiseGeometryVariables
         */
        template <typename CRT, typename ComputationalBlock>
        class RodSpanwiseGeometry
            : public CRTPHelper<ComputationalBlock, RodSpanwiseGeometry>,
              public RodSpanwiseGeometryVariables<CRT> {
         private:
          //**Type definitions**************************************************
          //! Traits type
          using Traits = CRT;
          //! This type
          using This = RodSpanwiseGeometry<Traits, ComputationalBlock>;
          //! Type of Variable definitions
          using VariableDefinitions = RodSpanwiseGeometryVariables<Traits>;
          //! CRTP Type
          using CRTP = CRTPHelper<ComputationalBlock, RodSpanwiseGeometry>;
          //! Index type
          using index_type = typename Traits::index_type;
          //********************************************************************

         protected:
          //**Type definitions**************************************************
          //! List of computed variables
          using typename VariableDefinitions::ComputedVariables;
          //! List of initialized variables
          using typename VariableDefinitions::InitializedVariables;
          //! List of all variables
          using typename VariableDefinitions::Variables;

          //! Current Variable definitions
          using typename VariableDefinitions::Director;
          using typename VariableDefinitions::Position;
          //********************************************************************

          //********************************************************************
          /*!\copydoc ComponentInitializationDocStub
           */
          template <typename BlockLike, typename CosseratInitializer>
          static void initialize(
              RodSpanwiseGeometry<Traits, BlockLike>& this_component,
              CosseratInitializer&& initializer) {
            initialize_component<This::InitializedVariables>(
                this_component.self(),
                std::forward<CosseratInitializer>(initializer));
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
          /*!\name Position methods*/
          //@{
          /*!\brief Gets position of the current rod
           */
          inline constexpr decltype(auto) get_position() & noexcept {
            return blocks::get<tags::Position>(self());
          }
          inline constexpr decltype(auto) get_position() const& noexcept {
            return blocks::get<tags::Position>(self());
          }
          inline constexpr decltype(auto) get_position(
              index_type idx) & noexcept {
            return Position::slice(get_position(), idx);
          }
          inline constexpr decltype(auto) get_position(
              index_type idx) const& noexcept {
            return Position::slice(get_position(), idx);
          }
          //@}
          //********************************************************************

          //********************************************************************
          /*!\name Director methods*/
          //@{
          /*!\brief Gets director of the current rod
           */
          inline constexpr decltype(auto) get_director() & noexcept {
            return blocks::get<tags::Director>(self());
          }
          inline constexpr decltype(auto) get_director() const& noexcept {
            return blocks::get<tags::Director>(self());
          }
          inline constexpr decltype(auto) get_director(
              index_type idx) & noexcept {
            return Director::slice(get_director(), idx);
          }
          inline constexpr decltype(auto) get_director(
              index_type idx) const& noexcept {
            return Director::slice(get_director(), idx);
          }
          //@}
          //********************************************************************

          //@}
          //********************************************************************
        };
        //**********************************************************************

      }  // namespace detail

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica

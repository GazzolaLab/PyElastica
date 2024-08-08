#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <type_traits>

/// Types
#include "Systems/States/Expressions/States/Types.hpp"
///
#include "ErrorHandling/Assert.hpp"
#include "Systems/States/Expressions/Expr/StateTimeDeltaMultExprBase.hpp"
#include "Systems/States/Expressions/OrderTags/TypeTraits.hpp"
#include "Systems/States/Expressions/States/StatesBase.hpp"
#include "Systems/States/TypeTraits/IsExpression.hpp"
#include "Time/SimulationTime.hpp"
#include "Time/Types.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Expression template object for multiplication between States and
     * TimeDelta
     * \ingroup states
     *
     * StatesTimeDeltaMultExpr class represents the compile time expression for
     * multiplication between a state and a time delta type
     *
     * \tparam ST Type of the state
     * \tparam TDT Type of the time-delta
     */
    template <typename ST, typename TDT>
    class StatesTimeDeltaMultExpr
        : public StateTimeDeltaMultExprBase<
              StatesBase<StatesTimeDeltaMultExpr<ST, TDT>>> {
     public:
      //**Type definitions******************************************************
      //! Type of this StateTimeDeltaMultExpr instance.
      using This = StatesTimeDeltaMultExpr<ST, TDT>;
      //! Base type of this StateTimeDeltaMultExpr instance.
      using BaseType = StateTimeDeltaMultExprBase<StatesBase<This>>;
      //! Operand type of the left-hand side dense vector expression.
      using LeftOperand =
          std::conditional_t<tt::is_expression_v<ST>, const ST, const ST&>;
      //! Operand type of the right-hand side dense vector expression.
      using RightOperand = TDT;
      //! Order type
      using Order = tt::lower_order_t<ST>;
      //************************************************************************

      //**Static members********************************************************
      //!< Number of groups / dimensions associated with this expression
      static constexpr unsigned int dimensions = ST::dimensions;
      //************************************************************************

      //! TimeType of this, to prevent dt * dt * (A) but allow
      // 5 * (dt * A) or (A * dt * 5)
      // (A * 5 * dt + B * )
      // dt * (5 * A) disallow
      // dt * (TimeType{5} * A) disallow
      // using TimeUnitType = time_unit_trait_t<TUTST, TDT>;
      // using TimeDeltaType = RightOperand;

      //**Constructor***********************************************************
      /*!\brief Constructor for the StateTimeDeltaMultExpr class.
       *
       * \param state The state in the multiplication expression.
       * \param time_delta The time-delta in the multiplication expression.
       */
      inline StatesTimeDeltaMultExpr(const ST& state, TDT time_delta) noexcept
          : state_(state), time_delta_(time_delta) {}
      //************************************************************************

      //**Size function*********************************************************
      /*!\brief Returns the current size/dimension of the state.
       *
       * \return The size of the state.
      */
      inline auto size() const noexcept { return state_.size(); }
      //************************************************************************

      //**Get functions*********************************************************
      /*!\name Get functions */
      //@{

      //**Get function**********************************************************
      /*!\brief Get function for the direct access to the state elements.
       *
       * \tparam Idx Access index. The index has to be in the range
       * \f$[0..dimensions]\f$, else a compile-time error is thrown.
       * \return The evaluated value of expression
      */
      template <unsigned int Idx>
      inline constexpr auto get() const noexcept {
        return state_.template get<Idx>() * time_delta_;
      }
      //************************************************************************

      //**Get function**********************************************************
      /*!\brief Get function for the direct access to the state elements.
       *
       * \tparam Group Access group. The group should belong to the underlying
       * concrete state, else a compile-time error is thrown.
       * \return The evaluated value of expression
      */
      template <typename Group>
      inline constexpr auto get() const noexcept {
        return state_.template get<Group>() * time_delta_;
      }
      //************************************************************************

      //@}
      //************************************************************************

      //**Left operand access***************************************************
      /*!\brief Returns the left-hand side state operand.
       *
       * \return The left-hand side state operand.
      */
      inline auto leftOperand() const noexcept -> LeftOperand { return state_; }
      //************************************************************************

      //**Right operand access**************************************************
      /*!\brief Returns the right-hand side time delta operand.
       *
       * \return The right-hand side time delta operand.
      */
      inline auto rightOperand() const noexcept -> RightOperand {
        return time_delta_;
      }
      //************************************************************************

     private:
      //**Member variables******************************************************
      //! State object of the multiplication expression.
      LeftOperand state_;
      //! Time-delta of the multiplication expression.
      RightOperand time_delta_;
      //************************************************************************
    };
    //**************************************************************************

    //==========================================================================
    //
    //  GLOBAL BINARY ARITHMETIC OPERATORS
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Operator for the multiplication of a States and TimeDelta
     * \ingroup states
     *
     * \details
     * This operator represents the multiplication between a States and a
     * TimeDelta value:
     * (\f$ \textrm{b} = \textrm{a} * dt \f$).
     * used as shown below:
     *
     * \usage
     * \code
     * using namespace elastica::states;
     * States<SO3<std::vector<double>, tags::DerivativeTag>,
     * SE3<std::vector<float>, tags::DerivativeTag>> a;
     * TimeDelta dt{2.0};
     *
     * //... Resizing and initialization
     * auto b = a * dt;
     * \endcode
     *
     * The function returns an expression `b` representing a multiplication
     * between the argument state and time-delta.
     *
     * \param state The left-hand side state for the multiplication.
     * \param time_delta The right-hand side TimeDelta for the multiplication
     * \return The result state scaled (dimensionally) by the time-delta
     */
    template <typename ST>  // Type of the left-hand side state
    inline decltype(auto) operator*(const StatesBase<ST>& state,
                                    TimeDelta const time_delta) noexcept {
      using ReturnType = const StatesTimeDeltaMultExpr<ST, TimeDelta>;
      return ReturnType(*state, time_delta);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Operator for the multiplication of a States and TimeDelta
     * \ingroup states
     *
     * \details
     * This operator represents the multiplication between a States and a
     * TimeDelta value:
     * (\f$ \textrm{b} = dt * \textrm{a} \f$).
     * used as shown below:
     *
     * \usage
     * \code
     * using namespace elastica::states;
     * States<SO3<std::vector<double>, tags::DerivativeTag>,
     * SE3<std::vector<float>, tags::DerivativeTag>> a;
     * TimeDelta dt{2.0};
     *
     * //... Resizing and initialization
     * auto b = dt * a;
     * \endcode
     *
     * The function returns an expression `b` representing a multiplication
     * between the argument state and time-delta.
     *
     * \param time_delta The left-hand side TimeDelta for the multiplication
     * \param state The right-hand side state for the multiplication.
     * \return The result state scaled (dimensionally) by the time-delta
     */
    template <typename ST>  // Type of the right-hand side state
    inline decltype(auto) operator*(TimeDelta const time_delta,
                                    const StatesBase<ST>& state) noexcept {
      using ReturnType = const StatesTimeDeltaMultExpr<ST, TimeDelta>;
      return ReturnType(*state, time_delta);
    }
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

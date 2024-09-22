#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <type_traits>

/// Forward declaration first
#include "Systems/States/Expressions/SO3/Types.hpp"
///
#include "ErrorHandling/Assert.hpp"
#include "Systems/States/Expressions/Expr/StateTimeDeltaMultExprBase.hpp"
#include "Systems/States/Expressions/OrderTags/TypeTraits.hpp"
#include "Systems/States/Expressions/SO3/SO3Base.hpp"
#include "Systems/States/TypeTraits/Aliases.hpp"
#include "Systems/States/TypeTraits/IsExpression.hpp"
#include "Systems/States/TypeTraits/IsVectorized.hpp"
#include "Time/SimulationTime.hpp"
#include "Time/Types.hpp"
#include "Utilities/Requires.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Expression template object for multiplication between SO3 states
     * and TimeDelta
     * \ingroup states
     *
     * SO3TimeDeltaMultExpr class represents the compile time expression for
     * multiplication between a SO3 state and a time delta type
     *
     * \tparam ST Type of the state
     * \tparam TDT Type of the time-delta
     */
    template <typename ST, typename TDT>
    class SO3TimeDeltaMultExpr : public StateTimeDeltaMultExprBase<
                                     SO3Base<SO3TimeDeltaMultExpr<ST, TDT>>> {
     public:
      //**Type definitions******************************************************
      //! Type of this SO3TimeDeltaMultExpr instance.
      using This = SO3TimeDeltaMultExpr<ST, TDT>;
      //! Base type of this SO3TimeDeltaMultExpr instance.
      using BaseType = StateTimeDeltaMultExprBase<SO3Base<This>>;
      //! Operand type of the left-hand side dense vector expression.
      using LeftOperand =
          std::conditional_t<tt::is_expression_v<ST>, const ST, const ST&>;
      //! Operand type of the right-hand side dense vector expression.
      using RightOperand = TDT;
      //! Order Tag
      using Order = tt::lower_order_t<ST>;
      //! Vectorized dispatch type
      using is_vectorized = tt::IsVectorized<ST>;
      //************************************************************************

      //**Constructor***********************************************************
      /*!\brief Constructor for the SO3TimeDeltaMultExpr class.
       *
       * \param state The SO3 state of the multiplication expression.
       * \param time_delta The time-delta of the multiplication expression.
       */
      inline SO3TimeDeltaMultExpr(const ST& state,
                                  TDT const time_delta) noexcept
          : state_(state), time_delta_(time_delta) {}
      //************************************************************************

      //**Size function*********************************************************
      /*!\brief Returns the current size/dimension of the state.
      //
      // \return The size of the state.
      */
      inline auto size() const noexcept -> std::size_t { return state_.size(); }
      //************************************************************************

      //**Get functions*********************************************************
      /*!\name Get functions */
      //@{

      //**Get function**********************************************************
      /*!\brief Get function for the direct access to the vector elements.
       *
       * \return The evaluated value of expression
       */
      template <bool B = is_vectorized{}, Requires<B> = nullptr>
      inline constexpr auto get() const noexcept {
        return state_.get() * static_cast<double>(time_delta_);
      }
      //************************************************************************

      //**Get function**********************************************************
      /*!\brief Get function for the direct access to the vector elements.
       *
       * \param index Index of data
       * \return The evaluated value of expression
       */
      template <bool B = not is_vectorized{}, Requires<B> = nullptr>
      inline constexpr auto get(std::size_t index) const noexcept {
        return state_.get(index) * static_cast<double>(time_delta_);
      }
      //************************************************************************

      //@}
      //************************************************************************

      //**Left operand access***************************************************
      /*!\brief Returns the left-hand side SO3 state operand.
      //
      // \return The left-hand side SO3 state operand.
      */
      inline auto leftOperand() const noexcept -> LeftOperand { return state_; }
      //************************************************************************

      //**Right operand access**************************************************
      /*!\brief Returns the right-hand side time delta operand.
      //
      // \return The right-hand side time delta operand.
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
    /*!\brief Operator for the multiplication of a SO3 and TimeDelta
     * \ingroup states
     *
     * \details
     * This operator represents the multiplication between a SO3 and a
     * TimeDelta value:
     * (\f$ \textrm{b} = \textrm{a} * dt \f$).
     * used as shown below:
     *
     * \usage
     * \code
     * using namespace elastica::states;
     * SO3<std::vector<double>, tags::DerivativeTag> a;
     * TimeDelta dt{2.0};
     *
     * //... Resizing and initialization
     * auto b = a * dt;
     * \endcode
     *
     * The function returns an expression `b` representing a multiplication
     * between the argument SO3 and time-delta.
     *
     * \param state The left-hand side state for the multiplication.
     * \param time_delta The right-hand side TimeDelta for the multiplication
     * \return The result state scaled (dimensionally) by the time-delta
     */
    template <typename ST>  // Type of the left-hand side state
    inline decltype(auto) operator*(SO3Base<ST> const& state,
                                    TimeDelta const time_delta) noexcept {
      using ReturnType = const SO3TimeDeltaMultExpr<ST, TimeDelta>;
      return ReturnType(*state, time_delta);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Operator for the multiplication of a SO3 and TimeDelta
     * \ingroup states
     *
     * \details
     * This operator represents the multiplication between a SO3 and a
     * TimeDelta value:
     * (\f$ \textrm{b} = dt * \textrm{a} \f$).
     * used as shown below:
     *
     * \usage
     * \code
     * using namespace elastica::states;
     * SO3<std::vector<double>, tags::DerivativeTag> a;
     * TimeDelta dt{2.0};
     *
     * //... Resizing and initialization
     * auto b = dt * a;
     * \endcode
     *
     * The function returns an expression `b` representing a multiplication
     * between the argument SO3 and time-delta.
     *
     * \param time_delta The right-hand side TimeDelta for the multiplication
     * \param state The left-hand side state for the multiplication.
     * \return The result state scaled (dimensionally) by the time-delta
     */
    template <typename ST>  // Type of the left-hand side state
    inline decltype(auto) operator*(TimeDelta const time_delta,
                                    SO3Base<ST> const& state) noexcept {
      using ReturnType = const SO3TimeDeltaMultExpr<ST, TimeDelta>;
      return ReturnType(*state, time_delta);
    }
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

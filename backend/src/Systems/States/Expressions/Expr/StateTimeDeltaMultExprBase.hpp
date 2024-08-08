#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "ErrorHandling/Assert.hpp"
#include "Systems/States/Expressions/Expr/StateTimeMultExpr.hpp"
#include "Systems/States/TypeTraits/IsScalar.hpp"
#include "Utilities/Requires.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Base class for all state--time-delta multiplication expression
     * templates.
     * \ingroup states
     *
     * The StateTimeDeltaMultExprBase serves to makr all expression templates
     * that implement a state--time-delta multiplication. All classes, that
     * represent a state--time-delta multiplication and that are used within the
     * expression template environment of \elastica have to derive
     * publicly from this class in order to qualify as state/time-delta
     * multiplication expression template.
     *
     * \tparam MultExpr type of the multiplication expression
     */
    template <typename MultExpr>  // State base type of the expression
    struct StateTimeDeltaMultExprBase : public StateTimeMultExpr<MultExpr> {};
    //**************************************************************************

    //==========================================================================
    //
    //  GLOBAL RESTRUCTURING UNARY ARITHMETIC OPERATORS
    //
    //==========================================================================

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Unary minus operator for the negation of a state-time-delta
    multiplication
    //        (\f$ \vec{a} = -(\vec{b} * s) \f$).
    // \ingroup state
    //
    // \param vec The state-time-delta multiplication to be negated.
    // \return The negation of the state-time-delta multiplication.
    //
    // This operator implements a performance optimized treatment of the
    negation of a state-time-delta
    // multiplication expression.
    */
    /*
    template <typename VT>  // Vector base type of the expression
    inline decltype(auto) operator-(const StateTimeDeltaMultExprBase<VT>& vec) {
      //      //ELASTICA_FUNCTION_TRACE;

      return (*vec).leftOperand() * (-(*vec).rightOperand());
    }
     */
    /*! \endcond */
    //**************************************************************************

    //==========================================================================
    //
    //  GLOBAL RESTRUCTURING BINARY ARITHMETIC OPERATORS
    //
    //==========================================================================

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Multiplication operator for the multiplication of a
     * state--time-delta multiplication expression and a scalar value.
     * \ingroup states
     *
     * \details
     * This operator implements a performance optimized treatment of the
     * multiplication of a state-time-delta multiplication expression and a
     * scalar value:
     * (\f$ \textrm{a} = (\textrm{b} *s1 ) * s2 \f$).
     *
     * \param state The left-hand side expression
     * \param scalar The right-hand side scalar value for the multiplication.
     * \return The scaled result state.
     */
    template <typename ST,      // State base type of the expression
              typename Scalar,  // Type of the right-hand side scalar
              Requires<tt::is_scalar_v<Scalar> > = nullptr>
    inline decltype(auto) operator*(const StateTimeDeltaMultExprBase<ST>& state,
                                    Scalar scalar) noexcept {
      return (*state).leftOperand() * ((*state).rightOperand() * scalar);
    }
    /*! \endcond */
    //**************************************************************************

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Multiplication operator for the multiplication of a
     * state--time-delta multiplication expression and a scalar value.
     * \ingroup states
     *
     * \details
     * This operator implements a performance optimized treatment of the
     * multiplication of a state-time-delta multiplication expression and a
     * scalar value:
     * (\f$ \textrm{a} = s2 * (\textrm{b}*s1) \f$).
     *
     * \param scalar The left-hand side scalar value for the multiplication.
     * \param state The right-hand side expression
     * \return The scaled result state.
     */
    template <typename ST,      // State base type of the expression
              typename Scalar,  // Type of the right-hand side scalar
              Requires<tt::is_scalar_v<Scalar> > = nullptr>
    inline decltype(auto) operator*(
        Scalar scalar, const StateTimeDeltaMultExprBase<ST>& state) noexcept {
      return (*state).leftOperand() * (scalar * (*state).rightOperand());
    }
    /*! \endcond */
    //**************************************************************************

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Division operator for the division of a state--time-delta
     * multiplication expression and a scalar value.
     * \ingroup states
     *
     * \details
     * This operator implements a performance optimized treatment of the
     * division of a state-time-delta multiplication expression by a
     * scalar value.
     * (\f$ \textrm{a} = (\textrm{b} *s1 ) / s2 \f$).
     *
     * \param vec The left-hand side state-time-delta multiplication.
     * \param scalar The right-hand side time-delta value for the division.
     * \return The scaled result state.
     */
    template <typename ST,      // State base type of the expression
              typename Scalar,  // Type of the right-hand side scalar
              Requires<tt::is_scalar_v<Scalar> > = nullptr>
    inline decltype(auto) operator/(const StateTimeDeltaMultExprBase<ST>& vec,
                                    Scalar scalar) noexcept {
      ELASTICA_ASSERT(scalar != Scalar(0), "Division by zero detected");
      return (*vec).leftOperand() * ((*vec).rightOperand() / scalar);
    }
    /*! \endcond */
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

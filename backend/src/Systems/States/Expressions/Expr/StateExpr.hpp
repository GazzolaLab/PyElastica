#pragma once

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Base class for all state expression templates.
     * \ingroup states
     *
     * StateExpr is the base class for all state expression templates. All
     * classes that represent an expression and that are used within the states
     * expression template environment of \elastica have to derive
     * publicly from this class in order to qualify as a valid state expression
     * template. Only in case a class is derived publicly from the StateExpr
     * base class, the elastica::states::tt::IsExpression type trait recognizes
     * the class as valid expression template:
     *
     * \code
     * template <typename T>
     * struct MyCustomState : public StateExpr<T> {};
     * \endcode
     *
     * \tparam Expr type of the expression
     */
    template <typename Expr>
    struct StateExpr : public Expr {};
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

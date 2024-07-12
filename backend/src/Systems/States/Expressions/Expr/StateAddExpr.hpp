#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/States/Expressions/Expr/StateExpr.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Base class for all state addition expression templates.
     * \ingroup states
     *
     * StateAddExpr serves to mark for all expression templates that
     * implement additions between two temporally evolving states. All classes
     * that represent a state addition (additions between groups, such as SO3,
     * SE3 and between collections of groups, such as States) to be used
     * within the expression template environment of \elastica have
     * to derive publicly from this class in order to qualify as addition
     * expression template.
     *
     * \tparam AddExpr type of the addition expression
     */
    template <typename AddExpr>
    struct StateAddExpr : public StateExpr<AddExpr> {};
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

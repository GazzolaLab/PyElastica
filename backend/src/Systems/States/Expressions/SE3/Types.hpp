#pragma once

namespace elastica {

  namespace states {

    ////////////////////////////////////////////////////////////////////////////
    //
    // Forward declarations of state types
    //
    ////////////////////////////////////////////////////////////////////////////

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    template <typename ST>  // Type of the state
    class SE3Base;

    template <typename Type,      // storage type
              typename OrderTag>  // order of derivative
    class SE3;

    template <typename ST1,  // Type of the left-hand side state
              typename ST2>  // Type of the right-hand side state
    class SE3SE3AddExpr;

    template <typename ST1,  // Type of the left-hand side state
              typename TDT>  // Type of the right-hand time-delta value
    class SE3TimeDeltaMultExpr;
    /*! \endcond */
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

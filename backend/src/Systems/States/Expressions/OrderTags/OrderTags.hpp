#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/States/Expressions/OrderTags/Types.hpp"

namespace elastica {

  namespace states {

    namespace tags {

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Tag for marking primitive states.
       * \ingroup states
       *
       * The PrimitiveTag marks a temporally evolving variable as a Primitive
       * for state classes within the Elastica++ library. For example, in
       * integrating the ODE system below
       *
       * \f{eqnarray*}{
       *  \frac{dx}{dt} &=& v \\
       *  \frac{dv}{dt} &=& a \\
       * \f}
       *
       * \f$ x \f$ is marked with a PrimitiveTag to indicate its a primitive
       * integral quantity, i.e. it cannot be simplified further as a derivative
       * of another temporally evolving quantity. Its typically used with
       * SO3 and SE3, as shown below
       *
       * \example
       * \code
       * using ::elastica::states;
       * using Position = SE3<vector<float>, tags::PrimitiveTag>;
       * using Director = SO3<matrix<float>, tags::PrimitiveTag>;
       * \endcode
       *
       * \see elastica::states::SO3, elastica::states::SE3
       */
      struct PrimitiveTag {};
      //************************************************************************

      //************************************************************************
      /*!\brief Tag for marking derivative states.
       * \ingroup states
       *
       * The DerivativeTag marks a temporally evolving variable as a Derivative
       * for state classes within the Elastica++ library. For example, in
       * integrating the ODE system below
       *
       * \f{eqnarray*}{
       *  \frac{dx}{dt} &=& v \\
       *  \frac{dv}{dt} &=& a \\
       * \f}
       *
       * \f$ v \f$ is marked with a DerivativeTag to indicate its a derivative
       * integral quantity, i.e. it can be simplified further as a derivative
       * of another temporally evolving quantity, in this case
       * \f$ v = \frac{dx}{dt}\f$. Its typically used with SO3 and SE3, as
       * shown below
       *
       * \example
       * \code
       * using ::elastica::states;
       * using Velocity = SE3<vector<float>, tags::DerivativeTag>;
       * using AngularVelocity = SO3<vector<float>, tags::DerivativeTag>;
       * \endcode
       *
       * \see elastica::states::SO3, elastica::states::SE3
       */
      struct DerivativeTag {};
      //************************************************************************

      //************************************************************************
      /*!\brief Tag for marking double derivative states.
       * \ingroup states
       *
       * The DoubleDerivativeTag marks a temporally evolving variable as a
       * DoubleDerivative for state classes within the Elastica++ library. For
       * example, in integrating the ODE system below
       *
       * \f{eqnarray*}{
       *  \frac{dx}{dt} &=& v \\
       *  \frac{dv}{dt} &=& a \\
       * \f}
       *
       * \f$ v \f$ is marked with a DoubleDerivativeTag to indicate its a double
       * derivative integral quantity, i.e. it can be simplified further as a
       * double derivative of another temporally evolving quantity, in this case
       * \f$ v = \frac{dx}{dt}\f$. Its typically used with SO3 and SE3, as
       * shown below
       *
       * \example
       * \code
       * using ::elastica::states;
       * using Acceleration = SE3<vector<float>, tags::DoubleDerivativeTag>;
       * using AngularAcceleration = SO3<vector<float>,
       * tags::DoubleDerivativeTag>;
       * \endcode
       *
       * \see elastica::states::SO3, elastica::states::SE3
       */
      struct DoubleDerivativeTag {};
      //************************************************************************

      namespace internal {
        struct DerivativeMultipliedByTimeTag {};
        struct DoubleDerivativeMultipliedByTimeTag {};
      }  // namespace internal

    }  // namespace tags

  }  // namespace states

}  // namespace elastica

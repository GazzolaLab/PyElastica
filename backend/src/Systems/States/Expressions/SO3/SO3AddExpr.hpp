#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <stdexcept>    // for invalid argument
#include <type_traits>  // for conditional_t

///
#include "Systems/States/Expressions/SO3/Types.hpp"
///
#include "Systems/States/Expressions/OrderTags/OrderTags.hpp"
#include "Systems/States/Expressions/SO3/SO3.hpp"
#include "Systems/States/Expressions/SO3/SO3Base.hpp"
#include "Systems/States/Expressions/SO3/SO3RotRotAddExpr.hpp"
#include "Systems/States/Expressions/SO3/SO3SO3AddExpr.hpp"
#include "Systems/States/TypeTraits/Aliases.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  GLOBAL BINARY ARITHMETIC OPERATORS
    //
    //==========================================================================

    // Developer note : we push these definitions here so that SO3 and
    // SO3SO3RotRotExpr are complete

    //**************************************************************************
    /*!\brief Operator for the addition of a primitive SO3 states with a SO3
     * expression
     * \ingroup states
     *
     * \details
     * This operator represents the addition of a primitive SO3 state with a
     * time-delta scaled SO3 expression
     * (\f$ \textrm{c} = \textrm{a} + dt * \textrm{b} \f$).
     * used as shown below:
     *
     * \usage
     * \code
     * using namespace elastica::states;
     * using matrix_type = // your favorite matrix type;
     * using vector_type = // your favorite vector type;
     * SO3<matrix_type, tags::PrimitiveTag> a;
     * SO3<vector_type, tags::DerivativeTag> b;
     * elastica::TimeDelta td{2.0};
     *
     * //... Resizing and initialization
     * auto c = a + dt * b;
     * \endcode
     *
     * The function returns an expression `c` representing an addition between
     * the two argument states. In case the current sizes of the two given
     * states don't match, a `std::invalid_argument` is thrown.
     *
     * \param lhs The left-hand side state for the state addition.
     * \param rhs The right-hand side state for the state addition.
     * \return The sum of the two states.
     * \exception std::invalid_argument State sizes do not match.
     */
    template <typename Type,  // Type of the left-hand state expression
              typename ST2>   // Type of the right-hand state expression
    inline decltype(auto) operator+(SO3<Type, tags::PrimitiveTag> const& lhs,
                                    SO3Base<ST2> const& rhs) {
      // ST1 => Q
      // this is only for Q + (w * dt)
      // also for Q + Q1 as ST matches Q, but this is not necessarily possible
      // can have a static assert here for this case ST \neq SO3<Type,
      // Primitive>
      static_assert(
          cpp17::is_same_v<tt::order_t<ST2>,
                           tags::internal::DerivativeMultipliedByTimeTag>,
          R"error(
We can only add a primitive to a derivative scaled by time in the time-stepping
algorithm! Specifically, adding a primitive to a primitive is forbidden.
)error");

      if ((*lhs).size() != (*rhs).size()) {
        throw std::invalid_argument("State sizes do not match");
      }

      // Put Q on LHS here, so that one declaration goes down
      using LHSType = SO3<Type, tags::PrimitiveTag>;
      using ReturnType = const SO3RotRotAddExpr<LHSType, ST2>;
      return ReturnType(*lhs, *rhs);
    }
    //**************************************************************************

    //    template <typename ST1,   // Type of the left-hand state expression
    //              typename Type>  // Type of the right-hand state expression
    //    inline decltype(auto) operator+(SO3Base<ST1> const& lhs,
    //                                    SO3<Type, tags::PrimitiveTag> const&
    //                                    rhs) {
    //      // for (w * dt) + Q
    //      // we need the opposite defined
    //      // ST1 can be deduced
    //      // This is a logical mistake in all sense of the world as matrix
    //      // multiplication (additions in the SO3 space) is not commutative.
    //      Yet
    //      // to "correct" for this mistake we provide this overload.
    //      return rhs + lhs;
    //    }

    //**************************************************************************
    /*!\brief Operator for the addition of a SO3RotRotAddExpr with a SO3
     * expression
     * \ingroup states
     *
     * \details
     * This operator represents the addition of a SO3RotRotAddExpr state with a
     * time-delta scaled SO3 expression
     * (\f$ \textrm{d} = (\textrm{a} + dt * \textrm{b}) + dt * \textrm{c} \f$).
     * used as shown below:
     *
     * \usage
     * \code
     * using namespace elastica::states;
     * using matrix_type = // your favorite matrix type;
     * using vector_type = // your favorite vector type;
     * SO3<matrix_type, tags::PrimitiveTag> a;
     * SO3<vector_type, tags::DerivativeTag> b, c;
     * elastica::TimeDelta td{2.0};
     *
     * //... Resizing and initialization
     * auto temp = a + dt * b;
     * auto d = a + dt * c;
     * \endcode
     *
     * The function returns an expression `d` representing an addition between
     * the two argument states. In case the current sizes of the two given
     * states don't match, a `std::invalid_argument` is thrown.
     *
     * \param lhs The left-hand side expression for addition.
     * \param rhs The right-hand side expression for addition.
     * \return The sum of the two states.
     * \exception std::invalid_argument State sizes do not match.
     */
    template <typename ST1,  // Type of the left-hand state expression
              typename ST2,  // Type of the right-hand state expression
              typename ST3>  // Type of the right-hand state expression
    inline decltype(auto) operator+(SO3RotRotAddExpr<ST1, ST2> const& lhs,
                                    SO3Base<ST3> const& rhs) {
      static_assert(
          cpp17::is_same_v<tt::order_t<ST3>,
                           tags::internal::DerivativeMultipliedByTimeTag>,
          R"error(
We can only add a primitive to a derivative scaled by time in the time-stepping
algorithm! Specifically, adding a primitive to a primitive is forbidden.
)error");

      if ((*lhs).size() != (*rhs).size()) {
        throw std::invalid_argument("State sizes do not match");
      }

      // Q + (w1 * dt + w2 * dt)
      // Q + SO3SO3AddExpr
      return (*lhs).leftOperand() + ((*lhs).rightOperand() + *rhs);
    }
    //**************************************************************************

    //    template <typename ST1,  // Type of the left-hand state expression
    //              typename ST2,  // Type of the right-hand state expression
    //              typename ST3>  // Type of the right-hand state expression
    //    inline decltype(auto) operator+(SO3Base<ST1> const& lhs,
    //                                    SO3RotRotAddExpr<ST2, ST3> const& rhs)
    //                                    {
    //      // for (w * dt) + (Q + w2 * dt)
    //      // This is a logical mistake in all sense of the world as matrix
    //      // multiplication (additions in the SO3 space) is not commutative.
    //      Yet
    //      // to "correct" for this mistake we provide this overload.
    //      return rhs + lhs;
    //    }

  }  // namespace states

}  // namespace elastica

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <stdexcept>
#include <type_traits>

/// Types always first
#include "Systems/States/Expressions/SO3/Types.hpp"
///
#include "ErrorHandling/Assert.hpp"
#include "Systems/States/Expressions/Expr/StateAddExpr.hpp"
#include "Systems/States/Expressions/OrderTags.hpp"
#include "Systems/States/Expressions/SO3/SO3Base.hpp"
#include "Systems/States/TypeTraits/Aliases.hpp"
#include "Systems/States/TypeTraits/IsExpression.hpp"
#include "Systems/States/TypeTraits/IsVectorized.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Expression template object for additions between two SO3 primitive
     * states.
     * \ingroup states
     *
     * SO3RotRotAddExpr class represents the compile time expression for
     * additions between two SO3 primitive objects
     *
     * \tparam STL Type of the left-hand side state
     * \tparam STR Type of the right-hand side state
     */
    template <typename STL, typename STR>
    class SO3RotRotAddExpr
        : public StateAddExpr<SO3Base<SO3RotRotAddExpr<STL, STR>>> {
     public:
      //**Type definitions******************************************************
      //! This type
      using This = SO3RotRotAddExpr<STL, STR>;
      //! Base type of this instance.
      using BaseType = StateAddExpr<SO3Base<This>>;
      //! Operand type of the left-hand side state expression.
      using LeftOperand =
          std::conditional_t<tt::is_expression_v<STL>, const STL, const STL&>;
      //! Operand type of the right-hand side state expression.
      using RightOperand =
          std::conditional_t<tt::is_expression_v<STR>, const STR, const STR&>;
      //! Order Tag
      using Order = tags::PrimitiveTag;
      //! Vectorized dispatch type
      using is_vectorized =
          cpp17::conjunction<tt::IsVectorized<STL>, tt::IsVectorized<STR>>;
      //************************************************************************

      //**Constructor***********************************************************
      /*!\brief Constructor for the SO3RotRotAddExpr class.
       *
       * \param lhs The left-hand side operand of the addition expression.
       * \param rhs The right-hand side operand of the addition expression.
      */
      inline SO3RotRotAddExpr(const STL& lhs, const STR& rhs) noexcept
          :  // Left-hand side dense vector of the addition expression
            lhs_(lhs),
            // Right-hand side dense vector of the addition expression
            rhs_(rhs) {}
      //************************************************************************

      // one can define get operator here and assign to the appropriate
      // backend rather than doing it in SO3. However for catching coding errors
      // here I leave it undefined.
      //**Subscript operator****************************************************
      /*!\brief Subscript operator for the direct access to the vector elements.
      //
      // \return The resulting value.
      */
      //      template <bool B = is_vectorized{}, Requires<B> = nullptr>
      //      inline constexpr auto get() const noexcept {
      //        return state_.get() * static_cast<double>(time_delta_);
      //      }
      //************************************************************************

      //**Subscript operator****************************************************
      /*!\brief Subscript operator for the direct access to the vector elements.
      //
      // \return The resulting value.
      */
      //      template <bool B = not is_vectorized{}, Requires<B> = nullptr>
      //      inline constexpr auto get(std::size_t idx) const noexcept {
      //        return state_.get(idx) * static_cast<double>(time_delta_);
      //      }
      //************************************************************************

      //**Size function*********************************************************
      /*!\brief Returns the current size/dimension of the vector.
       *
       * \return The size of the vector.
      */
      inline auto size() const noexcept -> std::size_t { return lhs_.size(); }
      //************************************************************************

      //**Left operand access***************************************************
      /*!\brief Returns the left-hand side SO3 state operand.
       *
       * \return The left-hand side SO3 state operand.
      */
      inline auto leftOperand() const noexcept -> LeftOperand { return lhs_; }
      //************************************************************************

      //**Right operand access**************************************************
      /*!\brief Returns the right-hand side SO3 state operand.
       *
       * \return The right-hand side state operand.
      */
      inline auto rightOperand() const noexcept -> RightOperand { return rhs_; }
      //************************************************************************

     private:
      //**Member variables******************************************************
      //! Left-hand side dense vector of the addition expression.
      LeftOperand lhs_;
      //! Right-hand side dense vector of the addition expression.
      RightOperand rhs_;
      //************************************************************************
    };
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

//=================================================================================================
//
//  GLOBAL BINARY ARITHMETIC OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Addition operator for the addition of two dense vectors (\f$
\vec{a}=\vec{b}+\vec{c} \f$).
// \ingroup dense_vector
//
// \param lhs The left-hand side dense vector for the vector addition.
// \param rhs The right-hand side dense vector for the vector addition.
// \return The sum of the two vectors.
// \exception std::invalid_argument Vector sizes do not match.
//
// This operator represents the addition of two dense vectors:

   \code
   blaze::DynamicVector<double> a, b, c;
   // ... Resizing and initialization
   c = a + b;
   \endcode

// The operator returns an expression representing a dense vector of the
higher-order element
// type of the two involved vector element types \a VT1::ElementType and \a
VT2::ElementType.
// Both vector types \a VT1 and \a VT2 as well as the two element types \a
VT1::ElementType
// and \a VT2::ElementType have to be supported by the AddTrait class
template.\n
// In case the current sizes of the two given vectors don't match, a \a
std::invalid_argument
// is thrown.
*/
//    template <
//        typename STL,  // Type of the left-hand state expression
//        typename STR,  // Type of the right-hand state expression
//        typename TDT,  // Type of the right-hand time-delta value
//        Requires<cpp17::conjunction_v<IsPrimitive<order_t<STL>>,
//                                      IsDerivative<order_t<STR>>>> =
//                                      nullptr>
//    inline decltype(auto) operator+(SO3Base<STL>& lhs,
//                                    SO3TimeDeltaMultExpr<STR, TDT>& rhs) {
//      // STL => Q
//      // this is for Q + (w * dt)
//
//      // within brackets
//      // STL => SO3TimeDeltaMultExpr<STR, TDT>
//      // STR => SO3TimeDeltaMultExpr<STR, TDT>
//      // for Q + ((w * dt) + (w *dt * 2))
//      // also for w + (\alpha * dt)
//      // also for (Q + w * dt) + (w* dt)
//      static_assert(
//          has_type_v<AddTrait<result_type_t<STL>,
//                              result_type_t<SO3TimeDeltaMultExpr<STR,
//                              TDT>>>>,
//          "+ operation not allowed");
//
//      if ((*lhs).size() != (*rhs).size()) {
//        throw std::invalid_argument("State sizes do not match");
//      }
//
//      // Put Q on LHS here, so that one declaration goes down
//
//      using ReturnType =
//          const SO3RotRotAddExpr<STL, SO3TimeDeltaMultExpr<STR, TDT>>;
//      return ReturnType(*lhs, *rhs);
//    }
//    //*************************************************************************************************

//    template <
//        typename STL,  // Type of the left-hand state expression
//        typename TDT,  // Type of the left-hand time-delta value
//        typename STR,  // Type of the right-hand state expression
//        Requires<cpp17::conjunction_v<IsPrimitive<tag_type_t<STR>>,
//                                      IsDerivative<tag_type_t<STL>>>> =
//                                      nullptr>
//    inline decltype(auto) operator+(SO3TimeDeltaMultExpr<STL, TDT>& lhs,
//                                    SO3Base<STR>& rhs) {
//      // BLAZE_FUNCTION_TRACE;
//      static_assert(
//          has_type_v<AddTrait<result_type_t<SO3TimeDeltaMultExpr<STL,
//          TDT>>,
//                              result_type_t<STR>>>,
//          "+ operation not allowed");
//
//      if ((*lhs).size() != (*rhs).size()) {
//        throw std::invalid_argument("State sizes do not match");
//      }
//
//      // We put Q on LHS here, so that we are assured Q is always the left
//      // operand
//      using ReturnType =
//          const SO3RotRotAddExpr<STR, SO3TimeDeltaMultExpr<STL, TDT>>;
//      return ReturnType(*rhs, *lhs);
//    }
//    //*************************************************************************************************

//    // Associativity always from left to right
//    // Q + omega1 * dt + omega2 * dt
//    // ---------------   -----------
//    // STL      STR          ST3
//    template <typename STL,  // Type of the left-hand state expression
//              typename STR,  // Type of the right-hand state expression
//              typename ST3,  // bl
//              typename TDT,  // bl
//              Requires<is_derivative_v<tag_type_t<ST3>>> = nullptr>
//    inline decltype(auto) operator+(SO3RotRotAddExpr<STL, STR>& lhs,
//                                    SO3TimeDeltaMultExpr<ST3, TDT>& rhs) {
//      static_assert(
//          has_type_v<AddTrait<result_type_t<SO3TimeDeltaMultExpr<ST3,
//          TDT>>,
//                              result_type_t<SO3RotRotAddExpr<STL, STR>>>>,
//          "+ operation not allowed");
//
//      // Split such that STR and ST3 are together, i.e
//      // S1 + (STR + ST3) with brackets
//      return (*lhs).leftOperand() + ((*lhs).rightOperand() + *rhs);
//    }

//    template <typename STL,  // Type of the left-hand state expression
//              typename STR,  // Type of the right-hand state expression
//              typename ST3,  // as
//              typename TDT,  // as
//              Requires<is_derivative_v<tag_type_t<ST3>>> = nullptr>
//    inline decltype(auto) operator+(SO3TimeDeltaMultExpr<ST3, TDT>& lhs,
//                                    SO3RotRotAddExpr<STL, STR>& rhs) {
//      static_assert(
//          has_type_v<AddTrait<result_type_t<SO3TimeDeltaMultExpr<ST3,
//          TDT>>,
//                              result_type_t<SO3RotRotAddExpr<STL, STR>>>>,
//          "+ operation not allowed");
//
//      // Split such that STR and ST3 are together, i.e
//      // S1 + (STR + ST3) with brackets
//      return (*rhs).leftOperand() + ((*rhs).rightOperand() + *lhs);
//    }

//    // (w1 * dt + w2 * dt) + w3 * dt
//    // (a1 * dt + s2 * dt)
//    // (w1 * dt + w2 * dt)
//    template <typename STL,  // Type of the left-hand state expression
//              typename STR,  // Type of the right-hand state expression
//              typename ST3, typename TDT>
//    inline decltype(auto) operator+(SO3TimeDeltaMultExpr<ST3, TDT>& lhs,
//                                    SO3RotRotAddExpr<STL, STR>& rhs) {
//      static_assert(
//          has_type_v<AddTrait<result_type_t<SO3TimeDeltaMultExpr<ST3,
//          TDT>>,
//                              result_type_t<SO3RotRotAddExpr<STL, STR>>>>,
//          "+ operation not allowed");
//
//      // Split such that STR and ST3 are together, i.e
//      // S1 + (STR + ST3) with brackets
//      return (*rhs).leftOperand() + ((*rhs).rightOperand() + *lhs);
//    }

// Never reached
//    // Associativity always from left to right
//    // omega1 * dt + Q + omega2 * dt
//    // ---------------   -----------
//    //     STL     STR       ST3
//    template <
//        typename STL,  // Type of the left-hand state expression
//        typename STR,  // Type of the right-hand state expression
//        typename ST3,
//        Requires<std::is_same<tag_type_t<STR>, PrimitiveTag>::value> =
//        nullptr>
//    inline decltype(auto) operator+(SO3RotRotAddExpr<STL, STR>& lhs,
//                                    SO3Base<ST3>& rhs) {
//      // Split such that STL and ST3 are together
//      // STR + (STL + ST3) with brackets
//      return (*lhs).leftOperand() + ((*lhs).rightOperand() + *rhs);
//    }

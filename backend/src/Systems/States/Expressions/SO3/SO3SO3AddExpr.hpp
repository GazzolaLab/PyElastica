#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <stdexcept> // for invalid argument
#include <type_traits> // for conditional_t

/// Types always first
#include "Systems/States/Expressions/SO3/Types.hpp"
///
#include "Systems/States/Expressions/Expr/StateAddExpr.hpp"
#include "Systems/States/Expressions/OrderTags.hpp"
#include "Systems/States/Expressions/OrderTags/TypeTraits.hpp"
#include "Systems/States/Expressions/SO3/SO3Base.hpp"
#include "Systems/States/TypeTraits/Aliases.hpp"
#include "Systems/States/TypeTraits/IsExpression.hpp"
#include "Systems/States/TypeTraits/IsPrimitive.hpp"
#include "Systems/States/TypeTraits/IsVectorized.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits/Cpp17.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Expression template object for additions between two States.
     * \ingroup states
     *
     * SO3SO3AddExpr class represents the compile time expression for
     * additions between two States objects
     *
     * \tparam STL Type of the left-hand side state
     * \tparam STR Type of the right-hand side state
     */
    template <typename STL, typename STR>
    class SO3SO3AddExpr
        : public StateAddExpr<SO3Base<SO3SO3AddExpr<STL, STR>>> {
     public:
      //**Type definitions******************************************************
      //! This type
      using This = SO3SO3AddExpr<STL, STR>;
      //! Base type of this instance.
      using BaseType = StateAddExpr<SO3Base<This>>;
      //! Operand type of the left-hand side dense vector expression.
      using LeftOperand =
          std::conditional_t<tt::is_expression_v<STL>, const STL, const STL&>;
      //! Operand type of the right-hand side dense vector expression.
      using RightOperand =
          std::conditional_t<tt::is_expression_v<STR>, const STR, const STR&>;
      //! Order Tag
      using Order = tt::common_order_t<STL, STR>;
      //! Vectorized Dispatch type
      using is_vectorized =
          cpp17::conjunction<tt::IsVectorized<STL>, tt::IsVectorized<STR>>;
      //************************************************************************

      //**Constructor***********************************************************
      /*!\brief Constructor for the SO3SO3AddExpr class.
       *
       * \param lhs The left-hand side operand of the addition expression.
       * \param rhs The right-hand side operand of the addition expression.
       */
      inline SO3SO3AddExpr(const STL& lhs, const STR& rhs) noexcept
          : lhs_(lhs), rhs_(rhs) {
        // ELASTICA_ASSERT(lhs.size() == rhs.size(), "Invalid state sizes");
      }
      //************************************************************************

      //**Size function*********************************************************
      /*!\brief Returns the current size/dimension of the vector.
       *
       * \return The size of the vector.
       */
      inline auto size() const noexcept -> std::size_t { return lhs_.size(); }
      //************************************************************************

      //**Get functions*********************************************************
      /*!\name Get functions */
      //@{

      //**Get function**********************************************************
      /*!\brief Get function for access to the evaluation of the operand
       * elements.
       *
       * \return The evaluated value of expression
       */
      template <bool B = is_vectorized{}, Requires<B> = nullptr>
      inline constexpr auto get() const noexcept {
        return lhs_.get() + rhs_.get();
      }
      //************************************************************************

      //**Get function**********************************************************
      /*!\brief Get function for access to the evaluation of the operand
       * elements.
       *
       * \param index Index of data
       * \return The evaluated value of expression
       */
      template <bool B = not is_vectorized{}, Requires<B> = nullptr>
      inline constexpr auto get(std::size_t index) const noexcept {
        return lhs_.get(index) + rhs_.get(index);
      }
      //************************************************************************

      //@}
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

    //==========================================================================
    //
    //  GLOBAL BINARY ARITHMETIC OPERATORS
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Operator for the addition of two non-primitive SO3 states
     * \ingroup states
     *
     * \details
     * This operator represents the addition of two SO3 states:
     * (\f$ \textrm{c} = \textrm{a} + \textrm{b} \f$).
     * used as shown below:
     *
     * \usage
     * \code
     * using namespace elastica::states;
     * SO3<std::vector<double>, tags::DerivativeTag> a, b;
     * elastica::TimeDelta td{2.0};
     *
     * //... Resizing and initialization
     * auto c = dt * a + dt * b;
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
    template <typename STL,  // Type of the left-hand state expression
              typename STR,  // Type of the right-hand state expression
              Requires<cpp17::conjunction_v<tt::IsNotPrimitive<STL>,
                                            tt::IsNotPrimitive<STR>>> = nullptr>
    inline decltype(auto) operator+(SO3Base<STL> const& lhs,
                                    SO3Base<STR> const& rhs) {
      if ((*lhs).size() != (*rhs).size()) {
        throw std::invalid_argument("State sizes do not match");
      }
      using ReturnType = const SO3SO3AddExpr<STL, STR>;
      return ReturnType(*lhs, *rhs);
    }
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

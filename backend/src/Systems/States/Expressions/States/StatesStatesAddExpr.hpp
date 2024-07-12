#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <stdexcept>  // throw
#include <type_traits>

///
#include "Systems/States/Expressions/States/Types.hpp"
///
#include "ErrorHandling/Assert.hpp"
#include "Systems/States/Expressions/Expr/StateAddExpr.hpp"
#include "Systems/States/Expressions/OrderTags/TypeTraits.hpp"
#include "Systems/States/Expressions/States/StatesBase.hpp"
#include "Systems/States/TypeTraits/IsExpression.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITIONS
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Expression template object for additions between two States.
     * \ingroup states
     *
     * StatesStatesAddExpr class represents the compile time expression for
     * additions between two States objects
     *
     * \tparam STL Type of the left-hand side state
     * \tparam STR Type of the right-hand side state
     */
    template <typename STL, typename STR>
    class StatesStatesAddExpr
        : public StateAddExpr<StatesBase<StatesStatesAddExpr<STL, STR>>> {
     public:
      //**Type definitions******************************************************
      //! Type of this StatesStatesAddExpr instance.
      using This = StatesStatesAddExpr<STL, STR>;
      //! Base type of this StatesStatesAddExpr instance.
      using BaseType = StateAddExpr<StatesBase<This>>;
      //! Operand type of the left-hand side state expression.
      using LeftOperand =
          std::conditional_t<tt::is_expression_v<STL>, const STL, const STL&>;
      //! Operand type of the right-hand side state expression.
      using RightOperand =
          std::conditional_t<tt::is_expression_v<STR>, const STR, const STR&>;
      //! Order type
      using Order = tt::common_order_t<STL, STR>;
      //************************************************************************

      //**Static members********************************************************
      //!< Number of groups / dimensions associated with this expression
      static constexpr unsigned int dimensions = STL::dimensions;
      //************************************************************************

      //**Constructor***********************************************************
      /*!\brief Constructor for the StatesStatesAddExpr class.
       *
       * \param lhs The left-hand side operand of the addition expression.
       * \param rhs The right-hand side operand of the addition expression.
       */
      inline StatesStatesAddExpr(const STL& lhs, const STR& rhs) noexcept
          :  // Left-hand side state of the addition expression
            lhs_(lhs),
            // Right-hand side state of the addition expression
            rhs_(rhs) {
        ELASTICA_ASSERT(lhs.size() == rhs.size(), "Invalid state sizes");
      }
      //************************************************************************

      //**Size function*********************************************************
      /*!\brief Returns the current size/dimension of the state.
       *
       * \return The size of the state.
       */
      inline auto size() const noexcept { return lhs_.size(); }
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
        return lhs_.template get<Idx>() + rhs_.template get<Idx>();
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
        return lhs_.template get<Group>() + rhs_.template get<Group>();
      }
      //************************************************************************

      //@}
      //************************************************************************

      //**Left operand access***************************************************
      /*!\brief Returns the left-hand side state operand.
       *
       * \return The left-hand side state operand.
       */
      inline auto leftOperand() const noexcept -> LeftOperand { return lhs_; }
      //************************************************************************

      //**Right operand access**************************************************
      /*!\brief Returns the right-hand side state operand.
       *
       * \return The right-hand side state operand.
       */
      inline auto rightOperand() const noexcept -> RightOperand { return rhs_; }
      //************************************************************************

     private:
      //**Member variables******************************************************
      //! Left-hand side state of the addition expression.
      LeftOperand lhs_;
      //! Right-hand side state of the addition expression.
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
    /*!\brief Operator for the addition of two States
     * \ingroup states
     *
     * \details
     * This operator represents the addition of two states:
     * (\f$ \textrm{c} = \textrm{a} + \textrm{b} \f$).
     * used as shown below:
     *
     * \usage
     * \code
     * using namespace elastica::states;
     * States<SO3<std::vector<double>, tags::PrimitiveTag>,
     * SE3<std::vector<float>, tags::PrimitiveTag>> a, b, c;
     *
     * //... Resizing and initialization
     * auto c = a + b;
     * \endcode
     *
     * The function returns an expression `c` representing an addition between
     * the two argument states. In case the dimensions of the two given states
     * don't match, a compiler-error is generated.
     * In case the current sizes of the two given states don't match, a
     * `std::invalid_argument` is thrown.
     *
     * \param lhs The left-hand side state for the state addition.
     * \param rhs The right-hand side state for the state addition.
     * \return The sum of the two states.
     * \exception std::invalid_argument State sizes do not match.
     */
    template <typename STL,  // Type of the left-hand state expression
              typename STR>  // Type of the right-hand state expression
    inline decltype(auto) operator+(StatesBase<STL> const& lhs,
                                    StatesBase<STR> const& rhs) {
      static_assert(STL::dimensions == STR::dimensions, "Error in state expr!");

      if ((*lhs).size() != (*rhs).size()) {
        throw std::invalid_argument("State sizes do not match");
      }

      using ReturnType = const StatesStatesAddExpr<STL, STR>;
      return ReturnType(*lhs, *rhs);
    }
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

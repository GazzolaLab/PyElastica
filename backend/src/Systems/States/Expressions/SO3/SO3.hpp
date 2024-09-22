#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <cstddef>

/// Forward declaration first
#include "Systems/States/Expressions/SO3/Types.hpp"
///
#include "Systems/States/Expressions/SO3/SO3Base.hpp"
#include "Systems/States/Expressions/SO3/SO3RotRotAddExpr.hpp"
#include "Systems/States/Expressions/SO3/SO3SO3AddExpr.hpp"
// #include "Systems/States/Expressions/SO3/SO3TimeDeltaMultExpr.hpp"
#include "Systems/States/Expressions/backends.hpp"
#include "Systems/States/TypeTraits/SupportsVectorizedOperations.hpp"
#include "Utilities/NonCopyable.hpp"
//#include "Utilities/Overloader.hpp"
#include "Utilities/Requires.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Temporally evolving states with dynamics in SO3 groups
     * \ingroup states
     *
     * \details
     * The SO3 class template represents temporal states evolving according to
     * dynamics in the SO3 Lie group. The type of the element and the group tag
     * of the state are specified via the template parameters:
     *
     * \code
     * template< class Type , class DerivativeOrder >
     * class StateCollection;
     * \endcode
     *
     * where
     *  - `Type` : specifies the type of the SO3 elements. SO3 can be used with
     * any non-cv-qualified, non-reference, non-pointer element type.
     *  - `Tag`  : Tag Type parameter to tag the SO3 state. This is usually one
     * of the tags defined in elastica::states::tags:
     *     - elastica::states::tags::PrimitiveTag,
     *     - elastica::states::tags::DerivativeTag,
     *     - elastica::states::tags::DoubleDerivativeTag.
     *
     * SO3 has the same interface of the States class, and they can work
     * together. The `dimensions`, in accordance with States, of the SO3 group
     * is 1. We note that SO3 does not store the `Type` object, but rather has
     * a non-owning pointer---the memory for the underlying data must then be
     * externally managed. This design ensures that SO3 is as non-intrusive as
     * possible, and all data is stored with the classes in @ref systems.
     * The data pointed by SO3 can be directly accessed with the get method,
     * with the get index defaulting to 0.
     *
     * SO3 states evolve according to the following dynamics, based on the
     * DerivativeOrder which are one of
     * - elastica::states::tags::PrimitiveTag,
     * - elastica::states::tags::DerivativeTag,
     * - elastica::states::tags::DoubleDerivativeTag.
     *
     * - Primitives when added to "lowered" Derivatives (obtained from
     *   multiplication of a DerivativeTag with a TimeDelta, see
     *   elastica::states::tt::lower_order_t) is exponentiated
     *   i.e Q + omega* dt == Q * exp(omega * dt)
     *   where Q is a primitive and omega is a derivative.
     * - Derivatives when added to Promoted Double Derivatives (obtained from
     *   multiplication of a DoubleDerivativeTag with a TimeDelta) has normal
     *   element-wise addition semantics.
     *
     * The use of SO3 is very natural and intuitive and is meant for use either
     * wrapped within States (if there are other states in the system) or
     * even separately (when the system being integrated has only SO3 dynamics,
     * as the case is with rotations.). All feasible operations (addition with
     * another SO3 group of same/promoted tag, multiplication with a TimeDelta)
     * can be performed on select SO3 and TimeDelta types. The following example
     * gives an impression of the use of SO3:
     *
     * \example
     * \code
     * using elastica::states::SO3;
     * using elastica::states::tags;
     *
     * using matrix_type = // your favorite matrix type;
     * using vector_type = // your favorite vector type;
     * matrix_type d(...), e(...);
     * vector_type f(...);
     * SO3<matrix_type, PrimitiveTag> a(&d);
     * SO3<matrix_type, PrimitiveTag> b(&e);
     * SO3<vector_type, DerivativeTag> x(&f);
     *
     * auto c = a + b;  // Addition between states of equal element type is
     * allowed
     *
     * // Subtraction, multiplication, division etc. is not allowed in
     * // accordance with semantics of temporal evolution
     * //// d = a - b
     * //// d = a * b
     * //// d = a / b
     *
     * // Scaling of States with a scalar directly is not allowed, even if its
     * // meant as a time-step
     *
     * //// double dt = 2.0;
     * //// c  = a * 2.0 * dt;
     * //// c  = 2.0 * dt * a;
     *
     * // Scaling of States with TimeDelta is allowed, in accordance with
     * // semantics of temporal evolution
     *
     * elastica::TimeDelta dt(2.0);
     * auto s  = dt * a;
     *
     * // Scalars acting with TimeDelta is allowed via implicit conversion
     * s  = 2.0 * dt * a;
     * s  = a * dt * 2.0;
     *
     * // Scaling with multiple TimeDelta is not allowed, in accordance with
     * // semantics of temporal evolution
     *
     * //// c  = TimeDelta<float>(2.0) * dt * a;
     * //// c  = a * TimeDelta<float>(2.0) * dt;
     * \endcode
     *
     * \tparam Type Type of the data pointed to
     * \tparam DerivativeOrder  order of derivative
     *
     * \see elastica::states::States, elastica::states::SE3
     */
    template <typename Type, typename DerivativeOrder>
    class SO3 : public SO3Base<SO3<Type, DerivativeOrder>> {
     private:
      //**Type definitions******************************************************
      //! NonOwningPointer Type of the Groups instantiated
      using NonOwningPointer = Type*;
      //************************************************************************
     public:
      //**Type definitions******************************************************
      //! Order Tag
      using Order = DerivativeOrder;
      //! Dispatch backend
      using is_vectorized = tt::SupportsVectorizedOperations<Type>;
      //************************************************************************
     public:
      //**Constructors**********************************************************
      /*!\name Constructors */
      //@{
      SO3() = default;
      explicit SO3(NonOwningPointer p);
      SO3(SO3 const&) = default;
      SO3(SO3&&) noexcept = default;
      //@}
      //************************************************************************

      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~SO3() = default;
      //@}
      //************************************************************************

      //**Resize function*******************************************************
      /*!\brief Changing the size of the group.
       *
       * \param n The new size of the group.
       * \return void
       * \exception std::runtime_error If data cannot be resized
       */
      inline void resize(std::size_t n) const {
        if (n == size())
          return;
        resize_backend(*p_, n);
      };
      //************************************************************************

      //**Get functions*********************************************************
      /*!\name Get functions */
      //@{

      //**Get function**********************************************************
      /*!\brief Get function for mutable groups.
       *
       * \return Mutable reference to pointed data
       */
      template <bool B = is_vectorized{}, Requires<B> = nullptr>
      inline constexpr auto get() & noexcept -> Type& {
        return *p_;
      }
      //************************************************************************

      //**Get function**********************************************************
      /*!\brief Get function for mutable groups.
       *
       * \param index Index of data
       * \return Mutable reference to pointed data
       */
      template <bool B = not is_vectorized{}, Requires<B> = nullptr>
      inline constexpr decltype(auto) get(std::size_t index) & noexcept {
        // todo backend
        return *p_[index];
      }
      //************************************************************************

      //**Get function**********************************************************
      /*!\brief Get function for constant groups.
       *
       * \return Const reference to pointed data
       */
      template <bool B = is_vectorized{}, Requires<B> = nullptr>
      inline constexpr auto get() const& noexcept -> Type const& {
        return *p_;
      }
      //************************************************************************

      //**Get function**********************************************************
      /*!\brief Get function for constant groups.
       *
       * \param index Index of data
       * \return Const reference to pointed data
       */
      template <bool B = not is_vectorized{}, Requires<B> = nullptr>
      inline constexpr decltype(auto) get(std::size_t index) const& noexcept {
        // todo backend
        return *p_[index];
      }
      //************************************************************************

      //@}
      //************************************************************************

      //**Size function*********************************************************
      /*!\brief Returns the current size of the state type.
      //
      // \return The size of the state
      */
      inline auto size() const noexcept -> std::size_t {
        return size_backend(*p_);
      }
      //************************************************************************

      //**Assignment operators**************************************************
      /*!\name Assignment operators */
      //@{
      template <typename ST>
      inline SO3& operator=(SO3Base<ST> const& rhs) &;
      template <typename ST>
      inline SO3& operator+=(SO3Base<ST> const& rhs) &;
      //@}
      //************************************************************************

      /*
      template <typename ST>
      SO3& operator=(const BaseState<ST>& expr) & {
        // Do not worry about resizing, memory allocation etc.
        // They will be taken care of in the interior scope
        const ST& st(*expr);
        // We should "resize" here if necessary?
        resize(st.size());

        EXPAND_PACK_LEFT_TO_RIGHT(std::get<Groups>(groups_) =
                                      st.template eval<Groups>());

        return *this;
      }
      */

     private:
      //      template <typename ST1, typename ST2, nullptr_t>
      //      inline auto assign(Type& vector, SO3SO3AddExpr<ST1, ST2> const&
      //      rhs)
      //          -> void;
      //      template <typename ST1, typename ST2, nullptr_t>
      //      inline auto assign(Type& vector, SO3RotRotAddExpr<ST1, ST2>
      //      const& rhs)
      //          -> void;

      //**Type definitions******************************************************
      //! Type marking template parameter `Type` to be vectorized
      using vectorized = std::true_type;
      //! Type marking template parameter `Type` to be not vectorized
      using not_vectorized = std::false_type;
      //************************************************************************

      //**Assign functions******************************************************
      /*!\name Assign functions */
      //@{
      template <typename ST1, typename ST2, typename BoolConstant>
      inline auto assign(SO3RotRotAddExpr<ST1, ST2> const& rhs,
                         BoolConstant /* meta */
                         ) noexcept -> void {
        // leftOperand is usually a Q, right operand is \sum coeff * dt * w
        // Doesnt matter if its vectorized or not here, since the backend takes
        // care of the implementation details
        // ugliest piece of code I have written so far
        SO3_primitive_assign(this->get(), rhs.leftOperand().get(),
                             rhs.rightOperand().get());
      }
      template <typename ST1, typename ST2>
      inline auto assign(SO3SO3AddExpr<ST1, ST2> const& rhs,
                         vectorized /* meta */) noexcept -> void {
        this->get() = rhs.get();
      }
      template <typename ST1, typename ST2>
      inline auto assign(SO3SO3AddExpr<ST1, ST2> const& rhs,
                         not_vectorized /* meta */) noexcept -> void {
        // leftOperand is usually a Q, right operand is \sum coeff * dt * w
        const std::size_t vec_size(size());
        for (auto i = 0UL; i < vec_size; ++i)
          get(i) = rhs.get(i);
      }
      //@}
      //************************************************************************

      //**Add Assign functions**************************************************
      /*!\name Add Assign functions */
      //@{
      template <typename ST, typename TDT>
      inline auto add_assign(
          SO3TimeDeltaMultExpr<ST, TDT> const& rhs_expr,
          tags::internal::DerivativeMultipliedByTimeTag /* meta */) noexcept
          -> void {
        SO3_primitive_add_assign(this->get(), rhs_expr.get());
      }

      template <typename ST, typename TDT>
      inline auto add_assign_impl(SO3TimeDeltaMultExpr<ST, TDT> const& rhs,
                                  vectorized /* meta */) noexcept {
        this->get() += rhs.get();
      }

      template <typename ST, typename TDT>
      inline auto add_assign_impl(SO3TimeDeltaMultExpr<ST, TDT> const& rhs,
                                  not_vectorized /* meta */) noexcept {
        const std::size_t vec_size(size());
        for (auto i = 0UL; i < vec_size; ++i)
          get(i) += rhs.get(i);
      }

      template <typename ST, typename TDT,
                typename LHS = SO3TimeDeltaMultExpr<ST, TDT>>
      inline auto add_assign(
          SO3TimeDeltaMultExpr<ST, TDT> const& rhs_expr,
          tags::internal::
              DoubleDerivativeMultipliedByTimeTag /* meta */) noexcept -> void {
        add_assign_impl(
            rhs_expr,
            cpp17::conjunction<is_vectorized, tt::is_vectorized_t<LHS>>{});
      }
      //@}
      //************************************************************************

      //**Member variables******************************************************
      //! Storage of type instantiated in the group
      NonOwningPointer p_{nullptr};
      //************************************************************************
    };
    //**************************************************************************

    //==========================================================================
    //
    //  CONSTRUCTORS
    //
    //==========================================================================

    //**Constructor*************************************************************
    /*!\brief Constructor for the Group class.
     *
     * \param p NonOwningPointer to the value to be stored
     */
    template <class Type,                // storage type
              typename DerivativeOrder>  // order of derivative
    SO3<Type, DerivativeOrder>::SO3(NonOwningPointer p) : p_(p) {}
    //**************************************************************************

    //==========================================================================
    //
    //  ASSIGNMENT OPERATORS
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Assignment operator for SO3 group.
     *
     * \param rhs SO3 group to be copied.
     * \return Reference to the assigned SO3 group.
     * \exception std::bad_alloc when resizing fails
     *
     * \note
     * The current groups is resized according to the rhs group and
     * initialized as a copy of the rhs group.
     *
     */
    template <class Type,                // storage type
              typename DerivativeOrder>  // order of derivative
    template <typename SO3T>             // Type of the right-hand SO3 group
    SO3<Type, DerivativeOrder>& SO3<Type, DerivativeOrder>::operator=(
        const SO3Base<SO3T>& rhs) & {
      resize((*rhs).size());
      assign(*rhs,
             cpp17::conjunction<is_vectorized, tt::is_vectorized_t<SO3T>>{});
      return *this;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Addition assignment operator for the addition of the SO3 groups
     * (\f$ \textrm{a}+=\textrm{b} \f$).
     *
     * \param rhs The right-hand side SO3 group to be added to the current SO3
     *            group
     * \return    Reference to the SO3 group.
     * \exception std::invalid_argument State sizes do not match.
     */
    template <class Type,                // storage type
              typename DerivativeOrder>  // order of derivative
    template <typename ST>               // Type of the right-hand SO3 group
    SO3<Type, DerivativeOrder>& SO3<Type, DerivativeOrder>::operator+=(
        SO3Base<ST> const& rhs_expr) & {
      if (size() != (*rhs_expr).size()) {
        throw std::invalid_argument("SO3 State sizes do not match");
      }
      add_assign(*rhs_expr, tt::order_t<ST>{});
      // using promoted_derivative = std::true_type;
      // using promoted_double_derivative = std::false_type;
      // make_overloader(
      //     [&matrix = *p_, &rhs = *rhs_expr](
      //         vectorized /* meta*/,
      //         promoted_derivative /*meta*/) noexcept -> void {
      //       // leftOperand is usually a Q, right operand is \sum coeff * dt *
      //       w
      //       // SO3_primitive_add_assign(matrix, rhs.get());
      //     },
      //     [&vector = *p_, &rhs = *rhs_expr](
      //         vectorized /* meta*/,
      //         promoted_double_derivative /*meta*/) noexcept -> void {
      //       vector += rhs.get();
      //     },
      //     [&matrix = *p_, &rhs = *rhs_expr](
      //         not_vectorized /* meta*/, promoted_derivative /*meta*/)
      //         noexcept
      //     -> void { SO3_primitive_add_assign(matrix, rhs.get()); },
      //     [this, &rhs = *rhs_expr, vec_size = size()](
      //         not_vectorized /* meta*/,
      //         promoted_double_derivative /*meta*/) noexcept -> void {
      //       for (auto i = 0UL; i < vec_size; ++i)
      //         get(i) = rhs.get(i);
      //     })(is_vectorized{},
      //        std::is_same<order_t<ST>,
      //        tags::DerivativeMultipliedByTimeTag>{});
      return *this;
    }
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

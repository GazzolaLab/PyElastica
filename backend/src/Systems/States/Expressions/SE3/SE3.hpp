#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <cstddef>

/// Forward declaration first
#include "Systems/States/Expressions/SE3/Types.hpp"
///
#include "Systems/States/Expressions/SE3/SE3Base.hpp"
#include "Systems/States/Expressions/SE3/SE3SE3AddExpr.hpp"
// #include "Systems/States/Expressions/SE3/SE3TimeDeltaMultExpr.hpp"
#include "Systems/States/Expressions/backends.hpp"
#include "Systems/States/TypeTraits/SupportsVectorizedOperations.hpp"
#include "Utilities/NonCopyable.hpp"
#include "Utilities/Requires.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Temporally evolving states with dynamics in SE3 groups
     * \ingroup states
     *
     * \details
     * The SE3 class template represents temporal states evolving according to
     * dynamics in the SE3 Lie group. The type of the element and the group tag
     * of the state are specified via the template parameters:
     *
     * \code
     * template< class Type , class DerivativeOrder >
     * class StateCollection;
     * \endcode
     *
     * where
     *  - `Type` : specifies the type of the SE3 elements. SE3 can be used with
     * any non-cv-qualified, non-reference, non-pointer element type.
     *  - `Tag`  : Tag Type parameter to tag the SE3 state. This is usually one
     * of the tags defined in elastica::states::tags:
     *     - elastica::states::tags::PrimitiveTag,
     *     - elastica::states::tags::DerivativeTag,
     *     - elastica::states::tags::DoubleDerivativeTag.
     *
     * SE3 has the same interface of the States class, and they can work
     * together. The `dimensions`, in accordance with States, of the SE3 group
     * is 1. We note that SE3 does not store the `Type` object, but rather has
     * a non-owning pointer---the memory for the underlying data must then be
     * externally managed. This design ensures that SE3 is as non-intrusive as
     * possible, and all data is stored with the classes in @ref systems.
     * The data pointed by SE3 can be directly accessed with the get method,
     * with the get index defaulting to 0.
     *
     * SE3 states evolve according to the following dynamics, based on the
     * DerivativeOrder which are one of
     * - elastica::states::tags::PrimitiveTag,
     * - elastica::states::tags::DerivativeTag,
     * - elastica::states::tags::DoubleDerivativeTag.
     *
     * - Primitives when added to Promoted Derivatives (obtained from
     *   multiplication of a DerivativeTag with a TimeDelta) has normal
     *   element-wise addition semantics.
     * - Derivatives when added to Promoted Double Derivatives (obtained from
     *   multiplication of a DoubleDerivativeTag with a TimeDelta) has normal
     *   element-wise addition semantics.
     *
     * These rules contrast with those of elastica::states::SO3, hence the
     * reason for this class.
     *
     * The use of SE3 is very natural and intuitive and is meant for use either
     * wrapped within States (if there are other states in the system) or
     * even separately (when the system being integrated has only SE3 dynamics,
     * as the case is with rotations.). All feasible operations (addition with
     * another SE3 group of same/promoted tag, multiplication with a TimeDelta)
     * can be performed on select SE3 and TimeDelta types. The following example
     * gives an impression of the use of SE3:
     *
     * \example
     * \code
     * using elastica::states::SE3;
     * using elastica::states::tags;
     *
     * using vector_type = // your favorite vector type;
     * vector_type d(...), e(...), f(...);
     * SE3<vector_type, PrimitiveTag> a(&d);
     * SE3<vector_type, PrimitiveTag> b(&e);
     * SE3<vector_type, DerivativeTag> x(&f);
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
     * \see elastica::states::States, elastica::states::SO3
     */
    template <typename Type, typename DerivativeOrder>
    class SE3 : public SE3Base<SE3<Type, DerivativeOrder>> {
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
      SE3() = default;
      explicit SE3(NonOwningPointer p);
      SE3(SE3 const& other) = default;
      SE3(SE3&&) noexcept = default;
      //@}
      //************************************************************************

      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~SE3() = default;
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
      inline SE3& operator=(SE3Base<ST> const& rhs) &;
      template <typename ST>
      inline SE3& operator+=(SE3Base<ST> const& rhs) &;
      //@}
      //************************************************************************

     private:
      //**Type definitions******************************************************
      //! Type marking template parameter `Type` to be vectorized
      using vectorized = std::true_type;
      //! Type marking template parameter `Type` to be not vectorized
      using not_vectorized = std::false_type;
      //************************************************************************

      //**Assign functions******************************************************
      /*!\name Assign functions */
      //@{
      template <typename ST1, typename ST2>
      inline auto assign(SE3SE3AddExpr<ST1, ST2> const& rhs,
                         vectorized /* meta */) noexcept -> void {
        this->get() = rhs.get();
      }
      template <typename ST1, typename ST2>
      inline auto assign(SE3SE3AddExpr<ST1, ST2> const& rhs,
                         not_vectorized /* meta */) noexcept -> void {
        const std::size_t vec_size(size());
        for (auto i = 0UL; i < vec_size; ++i)
          this->get(i) = rhs.get(i);
      }
      //@}
      //************************************************************************

      //**Add Assign functions**************************************************
      /*!\name Add Assign functions */
      //@{
      template <typename ST, typename TDT>
      inline auto add_assign(SE3TimeDeltaMultExpr<ST, TDT> const& rhs,
                             vectorized /* meta */) noexcept {
        this->get() += rhs.get();
      }

      template <typename ST, typename TDT>
      inline auto add_assign(SE3TimeDeltaMultExpr<ST, TDT> const& rhs,
                             not_vectorized /* meta */) noexcept {
        const std::size_t vec_size(size());
        for (auto i = 0UL; i < vec_size; ++i)
          get(i) += rhs.get(i);
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
    SE3<Type, DerivativeOrder>::SE3(NonOwningPointer p) : p_(p) {}
    //**************************************************************************

    //==========================================================================
    //
    //  ASSIGNMENT OPERATORS
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Assignment operator for SE3 group.
     *
     * \param rhs SE3 group to be copied.
     * \return Reference to the assigned SE3 group.
     * \exception std::bad_alloc when resizing fails
     *
     * \note
     * The current groups is resized according to the rhs group and
     * initialized as a copy of the rhs group.
     *
     */
    template <class Type,                // storage type
              typename DerivativeOrder>  // order of derivative
    template <typename SE3T>             // Type of the right-hand SE3 group
    SE3<Type, DerivativeOrder>& SE3<Type, DerivativeOrder>::operator=(
        const SE3Base<SE3T>& rhs) & {
      resize((*rhs).size());
      assign(*rhs,
             cpp17::conjunction<is_vectorized, tt::is_vectorized_t<SE3T>>{});
      return *this;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Addition assignment operator for the addition of the SE3 groups
     * (\f$ \textrm{a}+=\textrm{b} \f$).
     *
     * \param rhs The right-hand side SE3 group to be added to the current SE3
     *            group
     * \return    Reference to the SE3 group.
     * \exception std::invalid_argument State sizes do not match.
     */
    template <class Type,                // storage type
              typename DerivativeOrder>  // order of derivative
    template <typename SE3T>             // Type of the right-hand SE3 group
    SE3<Type, DerivativeOrder>& SE3<Type, DerivativeOrder>::operator+=(
        SE3Base<SE3T> const& rhs_expr) & {
      if (size() != (*rhs_expr).size()) {
        throw std::invalid_argument("SE3 State sizes do not match");
      }
      add_assign(
          *rhs_expr,
          cpp17::conjunction<is_vectorized, tt::is_vectorized_t<SE3T>>{});
      return *this;
    }
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

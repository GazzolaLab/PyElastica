#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <cstddef>  // std::size_t
#include <tuple>
#include <utility>  // std::move

/// Forward declaration first
#include "Systems/States/Expressions/States/Types.hpp"
///
#include "Systems/States/Expressions/States/StatesBase.hpp"
#include "Systems/States/TypeTraits/Aliases.hpp"
///
#include "Utilities/Invoke.hpp"  // index_apply
#include "Utilities/NonCopyable.hpp"
#include "Utilities/TMPL.hpp"  // for expand pack
//#include "Utilities/Requires.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief A collection of temporally evolving state-groups
     * \ingroup states
     *
     * \details
     * The States class template represents an arbitrary- sized collection of
     * statically allocated groups (set at compile time) of SE3 and SO3 types,
     * that represent temporally evolving states of a system.
     *
     * \usage
     * The type of the groups are specified via the template parameters:
     * \code
     * template< typename ... Groups > // Each of this is a Group, see below
     * class States;
     * \endcode
     *
     * Groups here specifies the type of the `Group` within the parameter pack.
     * Groups are valid Lie groups on which algebra (an addition between states
     * of the same group and multiplication operator with increments in time)
     * is well-defined. States can be used with any non-cv-qualified,
     * non-reference, non-pointer group types. These groups are then value
     * constructed inside the States class---i.e. references are not preserved.
     * The groups need not be distinct i.e. the same group can be used multiple
     * times in the parameter pack.
     * \code
     * // can repeat SE3 or SO3
     * using MyState = States<SE3<float>, SE3<double>, SO3<matrix_type>>;
     * \endcode
     *
     * The number of stored groups is a compile-time constant and is accessible
     * via the nested "dimensions" parameter. The stored groups can be directly
     * accessed with the get method, with the get index less than the dimensions
     * of the States. The numbering of the groups is
     *
     *                      \f[\left(\begin{array}{*{5}{c}}
     *                      0 & 1 & 2 & \cdots & dimensions-1 \\
     *                      \end{array}\right)\f]
     *
     * The use of States is natural and intuitive, and follows the interface of
     * a std::tuple. All feasible operations (addition with another States,
     * multiplication with a TimeDelta ...) can be performed on select States
     * and TimeDelta types. The following example gives an impression of the use
     * of States:
     *
     * \example
     * \code
     * using elastica::states::States;
     * using elastica::states::SO3;
     * using elastica::states::SE3;
     *
     * // Non-initialized state collection of dimension 2
     * States<SO3<double>, SE3<float>> a( );
     * double d(1.0), float f(2.0F);
     * a.get<0UL>(a) = &d;            // Initialization of the first element
     * a.get<1UL>(a) = &f;            // Initialization of the second element
     *
     * // Directly initialized dimension 2 state, preferred
     * States<SO3<double>, SE3<float>> b(&d, &f);
     * States<SO3<double>, SE3<float>> c;
     *
     * c = a + b;  // Addition between states of equal element type
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
     * \tparam Groups Parameter-pack of groups, either elastica::states::SE3 or
     * elastica::states::SO3
     *
     * \see elastica::states::SE3, elastica::states::SO3
     */
    template <class... Groups>
    class States : public StatesBase<States<Groups...>> {
     private:
      //**Type definitions******************************************************
      //! Storage Type of the Groups instantiated
      using GroupsStorage = std::tuple<Groups...>;
      //************************************************************************

     public:
      //**Type definitions******************************************************
      using This = States<Groups...>;  //!< Type of this State instance.
      using Order = std::common_type_t<tt::order_t<Groups>...>;  //!< Order type
      //************************************************************************

     public:
      //**Static members********************************************************
      //!< Number of groups / dimensions associated with this State
      static constexpr unsigned int dimensions = sizeof...(Groups);
      //************************************************************************

     public:
      //**Constructors**********************************************************
      /*!\name Constructors */
      //@{
      inline States() : groups_(){};
      explicit States(Groups&&... group_elements);
      //@}
      //************************************************************************

      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~States() = default;
      //@}
      //************************************************************************

      //**Get functions*********************************************************
      /*!\name Get functions */
      //@{

      //**Get function**********************************************************
      /*!\brief Get function for mutable state elements.
       *
       * \tparam Idx Access index. The index has to be in the range
       * \f$[0..dimensions]\f$, else a compile-time error is thrown.
       *
       * \return Mutable reference of group at location Idx.
       */
      template <unsigned int Idx /*, Requires<(Idx < dimensions)> = nullptr*/>
      inline constexpr auto get() noexcept
          -> std::tuple_element_t<Idx, GroupsStorage>& {
        return std::get<Idx>(groups_);
      }
      //************************************************************************

      //**Get function**********************************************************
      /*!\brief Get function for constant state elements.
       *
       * \tparam Idx Access index. The index has to be in the range
       * \f$[0..dimensions]\f$, else a compile-time error is thrown.
       *
       * \return Const reference of group at location Idx.
       */
      template <unsigned int Idx /*, Requires<(Idx < dimensions)> = nullptr*/>
      inline constexpr auto get() const noexcept
          -> std::tuple_element_t<Idx, GroupsStorage> const& {
        return std::get<Idx>(groups_);
      }
      //************************************************************************

      //**Get function**********************************************************
      /*!\brief Get function for mutable state elements.
       *
       * \tparam Group Access type. The type has to correspond to a constructed
       * type, else a compile-time error is thrown.
       *
       * \return Mutable reference of Group.
       */
      template <typename Group>
      inline constexpr auto get() noexcept -> Group& {
        return std::get<Group>(groups_);
      }
      //************************************************************************

      //**Get function**********************************************************
      /*!\brief Get function for constant state elements.
       *
       * \tparam Group Access type. The type has to correspond to a constructed
       * type, else a compile-time error is thrown.
       *
       * \return Const reference of Group.
       */
      template <typename Group>
      inline constexpr auto get() const noexcept -> Group const& {
        return std::get<Group>(groups_);
      }
      //************************************************************************

      //@}
      //************************************************************************

      //**Size function*********************************************************
      /*!\brief Returns the current size across all dimensions of the state.
       *
       * \return The size of the groups as a tuple of std::size_t
       */
      inline constexpr auto size() const noexcept {
        return std::make_tuple(std::get<Groups>(groups_).size()...);
      }
      //************************************************************************

      //**Assignment operators**************************************************
      /*!\name Assignment operators */
      //@{
      template <typename ST>
      inline States& operator=(const StatesBase<ST>& rhs) &;
      template <typename ST>
      inline States& operator+=(const StatesBase<ST>& rhs) &;
      //@}
      //************************************************************************

      /*
      template <typename... Initializer>
      explicit SymplecticState(std::piecewise_construct_t,
      std::tuple<FirstInitializers...> first_initializers,
          PointersToGroupTypes&&... elements)
          :
      groups_(std::make_tuple()std::forward<PointersToGroupTypes>(elements)...){};
          */

      // should return ref
      /*
      template <typename Type>
      inline constexpr auto get() noexcept ->
      decltype(std::get<Type>(groups_)){ return std::get<Type>(groups_);
      }
       */

      /*
      friend void swap(SymplecticState& f, SymplecticState& s) noexcept {
        using std::swap;
        swap(f.x_dofs_, s.x_dofs_);
        swap(f.v_dofs_, s.v_dofs_);
        swap(f.omega_dofs_, s.omega_dofs_);
        swap(f.Q_dofs_, s.Q_dofs_);  // implicit copy when called from const.
        f.compute_all_norms();
        s.compute_all_norms();
      }

      SymplecticState& operator=(SymplecticState other) {
        swap(*this, other);
        return *this;
      }
      */
     private:
      //**Member variables******************************************************
      //!< Storage of all groups instantiated in the state
      GroupsStorage groups_;
      //************************************************************************
    };
    //**************************************************************************

    //==========================================================================
    //
    //  CONSTRUCTORS
    //
    //==========================================================================

    //**Constructor*************************************************************
    /*!\brief Constructor for the State class.
    //
    // \param elements Initializers of groups
    */
    template <typename... Groups>  // Parameter-pack of groups
    // TODO : Revisit, this doesnt seem right that you need to pass in
    // Groups by forwarding references.
    // template <typename... ValueInitializers>  // Pack of group initializers
    States<Groups...>::States(Groups&&... group_elements)
        : groups_(std::forward<Groups>(group_elements)...) {}  // copy by value
    //**************************************************************************

    //==========================================================================
    //
    //  ASSIGNMENT OPERATORS
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Assignment operator for states.
     *
     * \param rhs State to be copied.
     * \return Reference to the assigned state.
     * \exception std::bad_alloc  when resizing fails
     *
     * \note
     * The composing groups can be resized according to the rhs state and
     * initialized as a copy of the rhs state.
     */
    template <typename... Groups>  // Parameter-pack of groups
    template <typename ST>         // Type of the right-hand side state
    inline States<Groups...>& States<Groups...>::operator=(
        const StatesBase<ST>& rhs) & {
      // usually we prevent self-assignment, but that cannot be the case
      // anywhere

      // DO NOT DO THIS!
      /* Problem with access using "Groups" is that there might be a mismatch
       * in the tags. For example consider
       * State<SO3<X, Primitive >> state, another_state;
       * State<SO3<T, Derivative>> derivative;
       * state = another_state + derivative * time;
       * expr on the RHS
       * AddExpr<decltype(another_state), MultExpr<decltype(derivative), TDT>>
       *
       * Upon doing get<SO3<X, Primitive>> on the RHS we do
       *
       * AddExpr.get<SO3<X, Primitive>>
       * => decltype(another_state).get<SO3<X, Primitive>> which is well formed
       *    +
       *    MultExpr.get<SO3<X, Primitive>>
       *
       * MultExpr.get<SO3<X, Primitive>>
       * => decltype(derivative).get<SO3<X, Primitive>> which is not well
       *    formed as derivative does not have a SO3<X, Primitive> state
       */
      //// Do not worry about resizing, memory allocation etc.
      //// They will be taken care of in the interior scope
      //// const ST& st(*rhs);
      //// EXPAND_PACK_LEFT_TO_RIGHT(get<Groups>() = st.template get<Groups>());
      // DO NOT DO THIS!
      index_apply<This::dimensions>(
          [this, &st = static_cast<const ST&>(*rhs)](auto... Is) {
            EXPAND_PACK_LEFT_TO_RIGHT(this->get<Is>() = st.template get<Is>());
          });
      return *this;
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Addition assignment operator for the addition of the state
     * (\f$ \textrm{a}+=\textrm{b} \f$).
     *
     * \details
     * In case the current sizes of the two given states don't match, a \a
     * std::invalid_argument is thrown.
     *
     * \param rhs The right-hand side state to be added to the current state.
     * \return Reference to the state.
     * \exception std:invalid_argument when state sizes do not match.
     */
    template <typename... Groups>  // Parameter-pack of groups
    template <typename ST>         // Type of the right-hand side state
    inline States<Groups...>& States<Groups...>::operator+=(
        const StatesBase<ST>& rhs) & {
      index_apply<This::dimensions>(
          [this, &st = static_cast<const ST&>(*rhs)](auto... Is) {
            EXPAND_PACK_LEFT_TO_RIGHT(this->get<Is>() += st.template get<Is>());
          });
      return *this;
    }
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

/*
 * TagType
 *
 * struct so3{ValueType* x};
 *
 * so3 + so3 => so3
 *
 * struct se3{ValueType* y};
 * se3 + so3 => so3
 *
 * <so3, se3>
 * StateType {
 *
 *  operator+(StateType<First,Second>){
 *    so3 + se3;
 *    se3 + se3;
 *    return Proxy<this, second>
 *  };
 * };
 *
 * operator += (TagType, DerivType)
 *
 * StateType operator *(real_t, DerivType)
 * StateType operator *(real_t, DerivType)
 */

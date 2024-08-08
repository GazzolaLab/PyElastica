#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/States/Expressions/SE3/Types.hpp"
#include "Utilities/CRTP.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Base class for temporal states in SE3 group.
     * \ingroup states
     *
     * The SE3Base class is a base class for all temporally evolving state
     * classes belonging to the SE3 group (rotations) within \elastica. It
     * provides an abstraction from the actual type of the state, but enables a
     * conversion back to this type via the 'Curiously Recurring Template
     * Pattern' (CRTP). Any SE3 group state or operations on SE3 groups need to
     * inherit from this class for seamlessly interfacing with the expression
     * template system.
     *
     * \tparam SE3ST Type of the derived state belonging to SE3 group
     */
    template <typename SE3ST>
    class SE3Base : CRTPHelper<SE3ST, SE3Base> {
     private:
      //**Type definitions******************************************************
      using This = SE3Base<SE3ST>;  //!< Type of this group instance.
      using CRTP = CRTPHelper<SE3ST, SE3Base>;  //!< CRTP type
      using CRTP::self;                         //!< CRTP methods
      //************************************************************************

     public:
      //**Type definitions******************************************************
      using StateType = SE3ST;  //!< Type of the state.
      //************************************************************************

      //**Conversion operators**************************************************
      /*!\name Conversion operators */
      //@{
      constexpr auto operator*() noexcept -> SE3ST&;
      constexpr auto operator*() const noexcept -> SE3ST const&;
      //@}
      //************************************************************************

     protected:
      //**Constructors**********************************************************
      /*!\name Constructors*/
      //@{
      SE3Base() = default;
      SE3Base(const SE3Base&) = default;
      SE3Base(SE3Base&&) noexcept = default;
      //@}
      //************************************************************************

      //**Assignment operators**************************************************
      /*!\name Assignment operators */
      //@{
      SE3Base& operator=(const SE3Base&) noexcept = default;
      SE3Base& operator=(SE3Base&&) noexcept = default;
      //@}
      //************************************************************************

      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~SE3Base() = default;
      //@}
      //************************************************************************
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief CRTP-based conversion operation for non-constant states.
     *
     * \details
     * This operator performs the CRTP-based type-safe downcast to the derived
     * state type `SE3ST`.
     *
     * \return Mutable reference to the derived type of the state.
     */
    template <typename SE3ST>  // Type of the SE3 state
    ELASTICA_ALWAYS_INLINE constexpr auto SE3Base<SE3ST>::operator*() noexcept
    -> SE3ST& {
      return static_cast<SE3ST&>(*this);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief CRTP-based conversion operation for constant states.
     *
     * \details
     * This operator performs the CRTP-based type-safe downcast to the derived
     * state type `SE3ST`.
     *
     * \return const reference to the derived type of the state.
     */
    template <typename SE3ST>  // Type of the SE3 state
    ELASTICA_ALWAYS_INLINE constexpr auto SE3Base<SE3ST>::operator*()
    const noexcept -> SE3ST const& {
      return static_cast<SE3ST const&>(*this);
    }
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

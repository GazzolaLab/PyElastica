#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/States/Expressions/SO3/Types.hpp"
#include "Utilities/CRTP.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Base class for temporal states in SO3 group.
     * \ingroup states
     *
     * The SO3Base class is a base class for all temporally evolving state
     * classes belonging to the SO3 group (rotations) within \elastica. It
     * provides an abstraction from the actual type of the state, but enables a
     * conversion back to this type via the 'Curiously Recurring Template
     * Pattern' (CRTP). Any SO3 group state or operations on SO3 groups need to
     * inherit from this class for seamlessly interfacing with the expression
     * template system.
     *
     * \tparam SO3ST Type of the derived state belonging to SO3 group
     */
    template <typename SO3ST>
    class SO3Base : CRTPHelper<SO3ST, SO3Base> {
     private:
      //**Type definitions******************************************************
      using This = SO3Base<SO3ST>;  //!< Type of this group instance.
      using CRTP = CRTPHelper<SO3ST, SO3Base>;  //!< CRTP type
      using CRTP::self;                         //!< CRTP methods
      //************************************************************************

     public:
      //**Type definitions******************************************************
      using StateType = SO3ST;  //!< Type of the state.
      //************************************************************************

      //**Conversion operators**************************************************
      /*!\name Conversion operators */
      //@{
      constexpr auto operator*() noexcept -> SO3ST&;
      constexpr auto operator*() const noexcept -> SO3ST const&;
      //@}
      //************************************************************************

     protected:
      //**Constructors**********************************************************
      /*!\name Constructors*/
      //@{
      SO3Base() = default;
      SO3Base(const SO3Base&) = default;
      SO3Base(SO3Base&&) noexcept = default;
      //@}
      //************************************************************************

      //**Assignment operators**************************************************
      /*!\name Assignment operators */
      //@{
      SO3Base& operator=(const SO3Base&) noexcept = default;
      SO3Base& operator=(SO3Base&&) noexcept = default;
      //@}
      //************************************************************************

      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~SO3Base() = default;
      //@}
      //************************************************************************
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief CRTP-based conversion operation for non-constant states.
     *
     * \details
     * This operator performs the CRTP-based type-safe downcast to the derived
     * state type `SO3ST`.
     *
     * \return Mutable reference to the derived type of the state.
     */
    template <typename SO3ST>  // Type of the SO3 state
    ELASTICA_ALWAYS_INLINE constexpr auto SO3Base<SO3ST>::operator*() noexcept
        -> SO3ST& {
      return static_cast<SO3ST&>(*this);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief CRTP-based conversion operation for constant states.
     *
     * \details
     * This operator performs the CRTP-based type-safe downcast to the derived
     * state type `SO3ST`.
     *
     * \return const reference to the derived type of the state.
     */
    template <typename SO3ST>  // Type of the SO3 state
    ELASTICA_ALWAYS_INLINE constexpr auto SO3Base<SO3ST>::operator*()
        const noexcept -> SO3ST const& {
      return static_cast<SO3ST const&>(*this);
    }
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/States/Expressions/States/Types.hpp"
#include "Utilities/CRTP.hpp"

namespace elastica {

  namespace states {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Base class for a collection of temporal states.
     * \ingroup states
     *
     * StatesBase is a base class for a collection of temporally evolving state
     * classes within \elastica. It provides an abstraction from the actual type
     * of the state, but enables a conversion back to the actual type via the
     * 'Curiously Recurring Template Pattern' (CRTP). Any state collection or
     * operations on state collections need to inherit from this class for
     * seamlessly interfacing with the expression template system.
     *
     * \tparam ST Type of the derived state
     */
    template <typename ST>
    class StatesBase : public CRTPHelper<ST, StatesBase> {
     private:
      //**Type definitions******************************************************
      using CRTP = CRTPHelper<ST, StatesBase>;  //!< CRTP type
      using CRTP::self;                         //!< Methods
      //************************************************************************

     public:
      //**Type definitions******************************************************
      using StateType = ST;  //!< Type of the state.
      //************************************************************************

      //**Conversion operators**************************************************
      /*!\name Conversion operators */
      //@{
      constexpr auto operator*() noexcept -> ST&;
      constexpr auto operator*() const noexcept -> ST const&;
      //@}
      //************************************************************************

     protected:
      //**Constructors**********************************************************
      /*!\name Constructors*/
      //@{
      StatesBase() = default;
      StatesBase(const StatesBase&) = default;
      StatesBase(StatesBase&&) noexcept = default;
      //@}
      //************************************************************************

      //**Assignment operators**************************************************
      /*!\name Assignment operators */
      //@{
      StatesBase& operator=(const StatesBase&) noexcept = default;
      StatesBase& operator=(StatesBase&&) noexcept = default;
      //@}
      //************************************************************************

      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~StatesBase() = default;
      //@}
      //************************************************************************
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief CRTP-based conversion operation for non-constant states.
     *
     * \details
     * This operator performs the CRTP-based type-safe downcast to the derived
     * state type `ST`.
     *
     * \return Mutable reference to the derived type of the state.
     */
    template <typename ST>  // Type of the state
    ELASTICA_ALWAYS_INLINE constexpr auto StatesBase<ST>::operator*() noexcept
        -> ST& {
      return static_cast<ST&>(*this);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief CRTP-based conversion operation for constant states.
     *
     * \details
     * This operator performs the CRTP-based type-safe downcast to the derived
     * state type `ST`.
     *
     * \return const reference to the derived type of the state.
     */
    template <typename ST>  // Type of the state
    ELASTICA_ALWAYS_INLINE constexpr auto StatesBase<ST>::operator*()
        const noexcept -> ST const& {
      return static_cast<ST const&>(*this);
    }
    //**************************************************************************

  }  // namespace states

}  // namespace elastica

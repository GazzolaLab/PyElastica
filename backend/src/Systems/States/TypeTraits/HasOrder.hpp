#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/States/TypeTraits/Aliases.hpp"
#include "Utilities/TypeTraits/IsDetected.hpp"

namespace elastica {

  namespace states {

    namespace tt {

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Check whether the given state `ST` has `order`
       * \ingroup states_tt
       *
       * \details
       * Inherits from std::true_type if the state expression `ST` has some
       * `order` bulit-in, otherwise inherits from std::false_type.
       *
       * \usage
       * For any type `ST`,
       * \code
       * using result = ::elastica::states::tt::HasOrder<ST>;
       * \endcode
       *
       * \metareturns
       * cpp17::bool_constant
       *
       * \semantics
       * If the state type `ST` has a bulit-in order in the ODE system, i.e. has
       * a nested typedef `Order` which is one of the `Tags` in
       * elastica::states::tags, then
       * \code
       * typename result::type = std::true_type;
       * \endcode
       * otherwise
       * \code
       * typename result::type = std::false_type;
       * \endcode
       *
       * \example
       * \snippet Test_HasOrder.cpp has_order_eg
       *
       * \tparam ST : the type to check
       */
      template <typename ST>
      struct HasOrder : public ::tt::is_detected<order_t, ST> {};
      //************************************************************************

    }  // namespace tt

  }  // namespace states

}  // namespace elastica

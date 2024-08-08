#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <type_traits>

#include "Aliases.hpp"
#include "Systems/Block/BlockVariables/Protocols.hpp"
#include "Utilities/IgnoreUnused.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"
#include "Utilities/TypeTraits/IsDetected.hpp"

namespace blocks {

  namespace protocols {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Block plugin protocol.
     * \ingroup block_protocols
     *
     * Class to enforce adherence to interface expected of a Plugin in blocks.
     * Any valid blocks Plugin within the \elastica library should (publicly)
     * conform to this class using
     * \code
     * tt::ConformsTo<protocols::Plugin>
     * \endcode
     * to indicate it qualifies as a block Plugin. Only in case a class
     * expresses such conformance, the ::tt::conforms_to and
     * ::tt::assert_conforms_to type traits recognizes the class as valid
     * Plugin.
     *
     * Requires that conforming type has the following types:
     * \snippet Block/Aliases.hpp variables_t
     * which should be a type list and each type within that list in turn
     * conforms to protocols::Variable
     *
     * The following shows an example of minimal conformance to
     * protocols::Plugin.
     *
     * \example
     * \snippet Block/Test_Protocols.cpp plugin_protocol_eg
     */
    struct Plugin {
      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Auxiliary helper struct for enforcing protocols.
      // \ingroup protocols
      */
      template <typename ConformingType>
      struct test {
       public:
        // Check for nested types of the ConfirmingType
        // static_assert(std::is_same<ConformingType, bool>::value, "Failure!");
        static_assert(::tt::is_detected_v<variables_t, ConformingType>,
                      "Not a conforming Block Plugin, doesn't have a "
                      "nested type called `Variables`");
        using Variables = variables_t<ConformingType>;
        // the variables_t should be a tmpl::list
        static_assert(::tt::is_a_v<tmpl::list, Variables>,
                      "Not a conforming Block Plugin, nested `Variables` type "
                      "is not a `tmpl::list`");

        // and all of its contents should conform to a block variable protocol
        template <typename Var>
        struct ReportVariableProtocolConformation : std::true_type {
          static_assert(
              tt::assert_conforms_to<Var, ::blocks::protocols::Variable>,
              "Variable called `Var` above does not conform to the protocols "
              "expected by a Variable!");
        };

        using variable_checks IGNORE_UNUSED = decltype(
            tmpl::transform<Variables,
                            ReportVariableProtocolConformation<tmpl::_1>>{});
      };
      /*! \endcond */
      //************************************************************************
    };
    //**************************************************************************

  }  // namespace protocols

}  // namespace blocks

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/Block/BlockVariables/Aliases.hpp"
//
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsDetected.hpp"

namespace blocks {

  namespace protocols {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Block variable protocol.
     * \ingroup block_protocols
     *
     * Class to enforce adherence to interface expected by Variables in blocks.
     * Any valid blocks Variable within the \elastica library should (publicly)
     * conform to this class using
     * \code
       tt::ConformsTo<protocols::Variable>
       \endcode
     * to indicate it qualifies as a block Variable. Only in case a class
     * expresses such conformance, the ::tt::conforms_to and
     * ::tt::assert_conforms_to type traits recognizes the class as valid
     * Variable.
     *
     * Requires that conforming type has the following types:
     * \snippet this expected_types
     *
     * The following shows an example of minimal conformance to
     * protocols::Variable. We remark that this alone does not guarantee a
     * correct program, and so its advisable to use blocks::Variable which
     * conforms to protocols::Variable.
     *
     * \example
     * \snippet BlockVariables/Test_Protocols.cpp variable_protocol_eg
    */
    struct Variable {
      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Auxiliary helper struct for enforcing protocols.
      // \ingroup protocols
      */
      template <typename ConformingType>
      struct test {
       public:
        // Check for nested types of the ConfirmingType
        /// [expected_types]
        static_assert(
            ::tt::is_detected_v<::blocks::parameter_t, ConformingType>,
            R"error(
Not a conforming Block Variable, doesn't have a nested type called `Parameter`
)error");
        static_assert(::tt::is_detected_v<::blocks::rank_t, ConformingType>,
                      R"error(
Not a conforming Block Variable, doesn't have a nested type called `Rank`
)error");
        // has a nested type called `type`
        static_assert(::tt::is_detected_v<tmpl::type_from, ConformingType>,
                      R"error(
Not a conforming Block Variable, doesn't have a nested type called `type`
)error");
        static_assert(
            ::tt::is_detected_v<::blocks::slice_type_t, ConformingType>,
            R"error(
Not a conforming Block Variable, doesn't have a nested type called `SliceType`
)error");
        static_assert(
            ::tt::is_detected_v<::blocks::const_slice_type_t, ConformingType>,
            R"error(
Not a conforming Block Variable, doesn't have a nested type called `ConstSliceType`
)error");
        /// [expected_types]
      };
      /*! \endcond */
      //************************************************************************
    };
    //**************************************************************************

  }  // namespace protocols

}  // namespace blocks

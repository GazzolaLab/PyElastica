#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cstddef>  // std::size_t
///
#include "Systems/common/Access.hpp"
#include "Systems/common/IndexCheck.hpp"
///
#include "Utilities/End.hpp"
#include "Utilities/TypeTraits/Cpp17.hpp"
#include "Utilities/TypeTraits/IsDetected.hpp"

namespace elastica {

  namespace protocols {

    namespace detail {
      using elastica::angular_velocity;
      using elastica::director;
      using elastica::external_loads;
      using elastica::external_torques;
      using elastica::index_check;
      using elastica::position;
      using elastica::velocity;

      struct Signatures {
        // [signatures]
        using index = std::size_t;
        using index_from_end = elastica::from_end;
        template <typename ConformingType>
        using index_check_from_start_result = decltype(index_check(
            std::declval<ConformingType const&>(), std::declval<index>()));
        template <typename ConformingType>
        using index_check_from_end_result =
            decltype(index_check(std::declval<ConformingType const&>(),
                                 std::declval<index_from_end>()));
        template <typename ConformingType>
        using position_result =
            decltype(position(std::declval<ConformingType&>()));
        template <typename ConformingType>
        using const_position_result =
            decltype(position(std::declval<ConformingType const&>()));
        template <typename ConformingType>
        using director_result =
            decltype(director(std::declval<ConformingType&>()));
        template <typename ConformingType>
        using const_director_result =
            decltype(director(std::declval<ConformingType const&>()));
        template <typename ConformingType>
        using velocity_result =
            decltype(velocity(std::declval<ConformingType&>()));
        template <typename ConformingType>
        using const_velocity_result =
            decltype(velocity(std::declval<ConformingType const&>()));
        template <typename ConformingType>
        using angular_velocity_result =
            decltype(angular_velocity(std::declval<ConformingType&>()));
        template <typename ConformingType>
        using const_angular_velocity_result =
            decltype(angular_velocity(std::declval<ConformingType const&>()));
        template <typename ConformingType>
        using external_loads_result =
            decltype(external_loads(std::declval<ConformingType&>()));
        template <typename ConformingType>
        using const_external_loads_result =
            decltype(external_loads(std::declval<ConformingType const&>()));
        template <typename ConformingType>
        using external_torques_result =
            decltype(external_torques(std::declval<ConformingType&>()));
        template <typename ConformingType>
        using const_external_torques_result =
            decltype(external_torques(std::declval<ConformingType const&>()));
        // [signatures]
      };

    }  // namespace detail

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Protocol for (physical) Systems
     * \ingroup systems
     *
     * Class to enforce adherence to a PhysicalSystems protocol. Any valid
     * system within the \elastica library (such as a CosseratRod, RigidBody)
     * should (publicly) inherit from this class to indicate it qualifies as a
     * valid System. Only in case a class is derived publicly from
     * this base class, the tt::conforms_to and tt::assert_conforms_to
     * type traits recognizes the class as valid System .
     *
     * Requires that a conforming type `ConformingType` has these member
     * functions
     * \snippet this expected_functions
     * and these free functions
     * \snippet this signatures
     *
     * \example
     * \snippet Systems/common/Test_Protocols.cpp systems_protocol_eg
     */
    struct PhysicalSystem {
      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Auxiliary helper struct for enforcing protocols.
       * \ingroup systems
       */
      template <typename ConformingType>
      struct test {
        /// [expected_types]
        /// [expected_types]

        /// [expected_functions]
        /// [expected_functions]

        /// [expected_free_functions]
        using expected_return_type = std::size_t;

        static_assert(
            cpp17::is_same_v<detail::Signatures::index_check_from_start_result<
                                 ConformingType>,
                             expected_return_type>,
            R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`std::size_t index_check(ConformingType const&, std::size_t index_from_start)`
)error");
        static_assert(
            cpp17::is_same_v<
                detail::Signatures::index_check_from_end_result<ConformingType>,
                expected_return_type>,
            R"error(
Not a conforming PhysicalSystem, doesn't properly implement function
`std::size_t index_check(ConformingType const&, elastica::from_end index_from_end)`
)error");

        static_assert(::tt::is_detected_v<detail::Signatures::position_result,
                                          ConformingType>,
                      R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`decltype(auto) ::elastica::position(ConformingType&) noexcept`
)error");
        static_assert(
            ::tt::is_detected_v<detail::Signatures::const_position_result,
                                ConformingType>,
            R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`decltype(auto) ::elastica::position(ConformingType const&) noexcept`
)error");
        static_assert(::tt::is_detected_v<detail::Signatures::director_result,
                                          ConformingType>,
                      R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`decltype(auto) ::elastica::director(ConformingType&) noexcept`
)error");
        static_assert(
            ::tt::is_detected_v<detail::Signatures::const_director_result,
                                ConformingType>,
            R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`decltype(auto) ::elastica::director(ConformingType const&) noexcept`
)error");
        static_assert(::tt::is_detected_v<detail::Signatures::velocity_result,
                                          ConformingType>,
                      R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`decltype(auto) ::elastica::velocity(ConformingType&) noexcept`
)error");
        static_assert(
            ::tt::is_detected_v<detail::Signatures::const_velocity_result,
                                ConformingType>,
            R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`decltype(auto) ::elastica::velocity(ConformingType const&) noexcept`
)error");
        static_assert(
            ::tt::is_detected_v<detail::Signatures::angular_velocity_result,
                                ConformingType>,
            R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`decltype(auto) ::elastica::angular_velocity(ConformingType&) noexcept`
)error");
        static_assert(::tt::is_detected_v<
                          detail::Signatures::const_angular_velocity_result,
                          ConformingType>,
                      R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`decltype(auto) ::elastica::angular_velocity(ConformingType const&) noexcept`
)error");
        static_assert(
            ::tt::is_detected_v<detail::Signatures::external_loads_result,
                                ConformingType>,
            R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`decltype(auto) ::elastica::external_loads(ConformingType&) noexcept`
)error");
        static_assert(
            ::tt::is_detected_v<detail::Signatures::const_external_loads_result,
                                ConformingType>,
            R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`decltype(auto) ::elastica::external_loads(ConformingType const&) noexcept`
)error");
        static_assert(
            ::tt::is_detected_v<detail::Signatures::external_torques_result,
                                ConformingType>,
            R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`decltype(auto) ::elastica::external_torques(ConformingType&) noexcept`
)error");
        static_assert(::tt::is_detected_v<
                          detail::Signatures::const_external_torques_result,
                          ConformingType>,
                      R"error(
Not a conforming PhysicalSystem, doesn't have a free-function
`decltype(auto) ::elastica::external_torques(ConformingType const&) noexcept`
)error");
        /// [expected_free_functions]
      };
      /*! \endcond */
      //************************************************************************
    };
    //**************************************************************************

  }  // namespace protocols

}  // namespace elastica

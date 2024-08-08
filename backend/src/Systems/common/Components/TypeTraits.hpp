#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Utilities/TypeTraits/IsCRTP.hpp"
#include "Utilities/TypeTraits/IsDetected.hpp"

///
#include "Systems/common/Components/Types.hpp"
///
#include "Systems/common/Components/Component.hpp"

namespace elastica {

  namespace tt {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Check whether a given type `C` is a valid Component
     * \ingroup systems
     *
     * \details
     * Inherits from std::true_type if `C` is a valid component template,
     * deriving from Component, otherwise inherits from std::false_type.
     *
     * \usage
     * For any type `C`,
     * \code
     * using result = IsComponent<C>;
     * \endcode
     *
     * \metareturns
     * cpp17::bool_constant
     *
     * \semantics
     * If the type `C` is derived from Component<C>, then
     * \code
     * typename result::type = std::true_type;
     * \endcode
     * otherwise
     * \code
     * typename result::type = std::false_type;
     * \endcode
     *
     * \example
     * \snippet Test_Components.cpp is_component_example
     *
     * \tparam C : the type to check
     *
     * \see Component
     */
    template <typename C>
    struct IsComponent : public ::tt::IsCRTP<::elastica::Component, C> {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Check whether a given type `C` is a valid GeometryComponent
     * \ingroup systems
     *
     * \details
     * Inherits from std::true_type if `C` is a valid geometry component
     * template, deriving from GeometryComponent, otherwise inherits from
     * std::false_type.
     *
     * \usage
     * For any type `C`,
     * \code
     * using result = IsGeometryComponent<C>;
     * \endcode
     *
     * \metareturns
     * cpp17::bool_constant
     *
     * \semantics
     * If the type `C` is derived from GeometryComponent<C>, then
     * \code
     * typename result::type = std::true_type;
     * \endcode
     * otherwise
     * \code
     * typename result::type = std::false_type;
     * \endcode
     *
     * \example
     * \snippet Test_Components.cpp is_geometry_component_example
     *
     * \tparam C : the type to check
     *
     * \see GeometryComponent
     */
    template <typename C>
    struct IsGeometryComponent
        : public ::tt::IsCRTP<::elastica::GeometryComponent, C> {};
    //**************************************************************************

    //**************************************************************************
    /*!\brief Check whether a given type `C` is a valid KinematicsComponent
     * \ingroup systems
     *
     * \details
     * Inherits from std::true_type if `C` is a valid kinematics component
     * template, deriving from KinematicsComponent, otherwise inherits from
     * std::false_type.
     *
     * \usage
     * For any type `C`,
     * \code
     * using result = IsKinematicsComponent<C>;
     * \endcode
     *
     * \metareturns
     * cpp17::bool_constant
     *
     * \semantics
     * If the type `C` is derived from KinematicsComponent<C>, then
     * \code
     * typename result::type = std::true_type;
     * \endcode
     * otherwise
     * \code
     * typename result::type = std::false_type;
     * \endcode
     *
     * \example
     * \snippet Test_Components.cpp is_kinematics_component_example
     *
     * \tparam C : the type to check
     *
     * \see KinematicsComponent
     */
    template <typename C>
    struct IsKinematicsComponent
        : public ::tt::IsCRTP<::elastica::KinematicsComponent, C> {};
    //**************************************************************************

    template <typename T>
    using geometry_component_t = typename T::geometry_component;

    template <typename T>
    struct GeometryComponentTrait
        : public ::tt::detected_or<NoSuchType, geometry_component_t, T> {};

    template <typename T>
    using kinematics_component_t = typename T::kinematics_component;

    template <typename T>
    struct KinematicsComponentTrait
        : public ::tt::detected_or<NoSuchType, kinematics_component_t, T> {};

  }  // namespace tt

}  // namespace elastica

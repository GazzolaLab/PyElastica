#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/common/Components/Types.hpp"
//
#include "Systems/common/Components/Component.hpp"

namespace elastica {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Base class for all kinematics components.
   * \ingroup systems
   *
   * \details
   * The Kinematics component is the base class for all kinematics components
   * used in programming systems within \elastica. All classes that represent a
   * kinematics component and that are used within the @ref blocks environment
   * of \elastica library have to derive publicly from this class in order to
   * qualify as a component.
   *
   * Only in case a class is derived publicly from the Component base class, the
   * IsComponent type trait recognizes the class as valid component.
   *
   * \tparam DerivedKinematicsComponent A downstream component that derives from
   * the Component class using the CRTP pattern.
   */
  template <typename DerivedKinematicsComponent>
  struct KinematicsComponent : public Component<DerivedKinematicsComponent> {
    //**Type definitions********************************************************
    //! Type of kinematics component
    using kinematics_component = DerivedKinematicsComponent;
    //**************************************************************************
  };
  //****************************************************************************

}  // namespace elastica

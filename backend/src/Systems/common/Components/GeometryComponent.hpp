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
  /*!\brief Base class for all geometry components.
   * \ingroup systems
   *
   * \details
   * The Geometry component is the base class for all geometry components
   * used in programming systems within \elastica. All classes that represent a
   * geometry component and that are used within the @ref blocks environment of
   * \elastica library have to derive publicly from this class in order to
   * qualify as a component.
   *
   * Only in case a class is derived publicly from the Component base class, the
   * IsComponent type trait recognizes the class as valid component.
   *
   * \tparam DerivedGeometryComponent A downstream component that derives from
   * the Component class using the CRTP pattern.
   */
  template <typename DerivedGeometryComponent>
  struct GeometryComponent : public Component<DerivedGeometryComponent> {
    // for ease in programming, ideally we shouldnt need it
    //**Type definitions********************************************************
    //! Type of geometry component
    using geometry_component = DerivedGeometryComponent;
    //**************************************************************************
  };
  //****************************************************************************

}  // namespace elastica

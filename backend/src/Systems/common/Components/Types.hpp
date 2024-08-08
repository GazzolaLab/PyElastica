#pragma once

namespace elastica {

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  template <typename /* DerivedComponent*/>
  struct Component;
  template <typename /* DerivedComponent*/>
  struct GeometryComponent;
  template <typename /* DerivedComponent*/>
  struct KinematicsComponent;
  template <typename /* DerivedComponent*/>
  struct NameComponent;
  /*! \endcond */
  //****************************************************************************

  // This is an anti-pattern, but is required for aliases to not include
  // definition
  namespace detail {
    template <typename /* Traits */, typename /* Block */,
              typename /*NameImpl*/>
    struct NameAdapter;
  }

}  // namespace elastica

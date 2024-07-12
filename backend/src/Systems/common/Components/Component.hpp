#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/common/Components/Types.hpp"

namespace elastica {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Base class for all component templates.
   * \ingroup systems
   *
   * \details
   * The Component class is the base class for all component templates
   * used in programming systems within \elastica. All classes that represent a
   * component and that are used within the @ref blocks environment of \elastica
   * library have to derive publicly from this class in order to qualify as a
   * component.
   *
   * Only in case a class is derived publicly from the Component base
   * class, the IsComponent type trait recognizes the class as valid
   * component.
   *
   * \tparam DerivedComponent A downstream component that derives from
   * the Component class using the CRTP pattern.
   */
  template <typename DerivedComponent>
  struct Component;
  //****************************************************************************

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Specialized of Component for Cosserat Rod components
   * \ingroup systems
   */
  template <template <typename, typename, typename...> class DerivedComponent,
            typename Traits, typename InstantiatedBlock,
            typename... DerivedComponentMetaArgs>
  struct Component<DerivedComponent<Traits, InstantiatedBlock,
                                    DerivedComponentMetaArgs...>> {};
  /*! \endcond */
  //****************************************************************************

}  // namespace elastica

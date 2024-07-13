#pragma once

//******************************************************************************
// Includes
//******************************************************************************
///
#include "Systems/CosseratRods/Components/helpers/Types.hpp"
///
#include "Systems/CosseratRods/Components/helpers/Component.hpp"
#include "Systems/CosseratRods/Components/helpers/TypeTraits.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Helper class for adapting Component templates to Adapter and
       * Interfaces
       * \ingroup cosserat_rod_component
       *
       * \details
       * Adapt helps a valid Cosserat rod `Component` template (geometry,
       * elasticity etc) adapt as new components (such as adding new behavior)
       * or meet interfaces for working in sync with the rest of the Cosserat
       * rod hierarchy.
       *
       * \usage
       * For any component template `Component` and adapters `Adapters...`, with
       * the Component template of two types `Traits` and `Block`
       * \code
       * using result = typename
       * Adapt<Component<Traits, Block>>::template with<Adapters...>;
       * \endcode
       * forms an adapted Component `result` adapting `Component` with
       * `Adapters...` with `Adapters` expanded right to left (i.e. the
       * left-most Adapter is applied last and has the most immediate effect)
       *
       * \metareturns
       * A type `result` which is an adapted Component, with the class signature
       * `result = LeftMostAdapter<NextLeftMostAdapter<...<Component>...>>`
       *
       * \example
       * \snippet Test_AdapterHelpers.cpp adapter_helper_eg
       *
       * \tparam ComponentParam A valid Cosserat rod Component template
       */
      template <typename ComponentParam,
                bool B = tt::IsComponent<ComponentParam>::value>
      struct Adapt;

      //**********************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Helper for associating from `Component<Traits, Block>` to
       * multiple adapters (recursively) of the form
       * `Adapter<Traits, Block, Component>`
       * \ingroup cosserat_rod_component
       */
      template <template <typename /* Traits */, typename /* Block*/>
                class Component,
                typename Traits, typename DerivedBlock>
      struct Adapt<Component<Traits, DerivedBlock>, true> {
        //**********************************************************************
        /*!\brief Helper for associating from `Component<Traits, Block>` to
         * multiple adapters (recursively) of the form
         * `Adapter<Traits, Block, Component>`
         * \ingroup cosserat_rod_component
         */
        template <template <typename, typename, typename> class FirstAdapter,
                  template <typename, typename, typename>
                  class... OtherAdapters>
        struct __with_helper {
          //      template <typename Traits, typename Block, typename...
          //      ComponentMetaArgs> using type = FirstAdapter<Traits, Block,
          //                                typename
          //                                with<OtherAdapters...>::template
          //                                type<
          //                                    Traits, Block,
          //                                    ComponentMetaArgs...>>;

          //**Type definitions**************************************************
          //! Wrapper type to generate Adapted component
          using type =
              FirstAdapter<Traits, DerivedBlock,
                           typename __with_helper<OtherAdapters...>::type>;
          //********************************************************************
        };
        //**********************************************************************

        //**********************************************************************
        /*!\brief Helper for associating from `Component<Traits, Block>` to
         * `Adapter<Traits, Block, Component>`
         * \ingroup cosserat_rod_component
         */
        template <template <typename, typename, typename> class Adapter>
        struct __with_helper<Adapter> {
          //      template <typename Traits, typename Block, typename...
          //      ComponentMetaArgs> using type = Adapter<Traits, Block,
          //      Component<Traits, Block, ComponentMetaArgs...>>;
          //**Type definitions**************************************************
          //! Wrapper type to generate Adapted component
          using type =
              Adapter<Traits, DerivedBlock, Component<Traits, DerivedBlock>>;
          //********************************************************************
        };
        //**********************************************************************

        //**Type definitions****************************************************
        //! Template type to generate Adapted component
        template <template <typename, typename, typename> class... Adapters>
        using with = typename __with_helper<Adapters...>::type;
        //**********************************************************************
      };
      /*! \endcond */
      //************************************************************************

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica

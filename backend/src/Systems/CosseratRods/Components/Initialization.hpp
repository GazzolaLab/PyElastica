#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/common/Initialization.hpp"
//
#include "Systems/CosseratRods/CosseratRodPlugin.hpp"

namespace elastica {

  namespace cosserat_rod {

    //**************************************************************************
    /*!\brief Component initializer for a Cosserat-rod component
     * \ingroup cosserat_rods
     *
     * \tparam Variables Typelist of `BlockVariables` to initialize.
     * \param block_like Cosserat-rod block-slice.
     * \param initializer Cosserat-rod block-initializer.
     */
    template <typename Variables, typename CRT,
              template <typename /*CRT*/, typename /* ComputationalBlock */>
              class... Components,
              // typename Initializer>
              typename... Initializers>
    inline void initialize_component(
        ::blocks::BlockSlice<cosserat_rod::CosseratRodPlugin<
            CRT, ::blocks::BlockSlice, Components...>>& block_like,
        ::blocks::BlockInitializer<cosserat_rod::CosseratRodPlugin<
                                       CRT, ::blocks::Block, Components...>,
                                   Initializers...>
            initializers) {
      ::elastica::detail::initialize_component<Variables>(
          block_like, std::move(initializers));
    }
    //**************************************************************************

    // FIXME : https://github.com/tp5uiuc/elasticapp/issues/460
    // Don't use initialize on model directly, so remove this function.
    //**************************************************************************
    /*!\brief Component initializer for a Cosserat-rod component
     * \ingroup cosserat_rods
     *
     * \tparam Variables Typelist of `BlockVariables` to initialize.
     * \param block_like Cosserat-rod block.
     * \param initializer Cosserat-rod block-initializer.
     */
    template <typename Variables, typename CRT,
              template <typename /*CRT*/, typename /* ComputationalBlock */>
              class... Components,
              // typename Initializer>
              typename... Initializers>
    void initialize_component(
        ::blocks::Block<cosserat_rod::CosseratRodPlugin<
            CRT, ::blocks::Block, Components...>>& block_like,
        ::blocks::BlockInitializer<cosserat_rod::CosseratRodPlugin<
                                       CRT, ::blocks::Block, Components...>,
                                   Initializers...>
            initializers) {
      ::elastica::detail::initialize_component<Variables>(
          block_like, std::move(initializers));
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Initialize method for the current component
     *
     * \details
     * initialize() is responsible for initializing the variables present
     * in the current component (in the final block of data). For
     * `InitializedVariables`, it fetches the corresponding initializer from
     * the set of `initializers` to fill in the data. For
     * `ComputedVariables` the default value is usually set. In this way,
     * each component is responsible for initializing its own data in a
     * manner that does not intrude with initialization from other
     * components.
     *
     * \param this_component The current component to be initialized
     * \param initializer Object with parameters to initialize `InitializedVar`
     */
    struct ComponentInitializationDocStub {};
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

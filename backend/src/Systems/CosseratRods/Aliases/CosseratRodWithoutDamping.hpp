#pragma once

//******************************************************************************
// Includes
//******************************************************************************
// common
#include "Systems/common/Types.hpp"
// blocks
#include "Systems/Block/Types.hpp"
// cosserat rod components
#include "Systems/CosseratRods/_Types.hpp"
#include "Systems/CosseratRods/Components/Names/CosseratRodWithoutDamping.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      template <typename T, typename B>
      using CosseratRodWithoutDampingComponents = ::elastica::SymplecticPolicy<
          T, B, ::elastica::cosserat_rod::component::WithCircularCosseratRod,
          ::elastica::cosserat_rod::component::
              WithDiagonalLinearHyperElasticModel,
          ::elastica::cosserat_rod::component::WithRodKinematics,
          ::elastica::cosserat_rod::component::
              CosseratRodWithoutDampingNameAdapted>;

      using CosseratRodWithoutDamping =
          ::elastica::cosserat_rod::CosseratRodPlugin<
              ::elastica::cosserat_rod::DefaultCosseratRodTraits,
              ::blocks::Block, detail::CosseratRodWithoutDampingComponents>;
      using CosseratRodWithoutDampingSlice =
          ::elastica::cosserat_rod::CosseratRodPlugin<
              ::elastica::cosserat_rod::DefaultCosseratRodTraits,
              ::blocks::BlockSlice,
              detail::CosseratRodWithoutDampingComponents>;
      using CosseratRodWithoutDampingConstSlice =
          ::elastica::cosserat_rod::CosseratRodPlugin<
              ::elastica::cosserat_rod::DefaultCosseratRodTraits,
              ::blocks::ConstBlockSlice,
              detail::CosseratRodWithoutDampingComponents>;
      using CosseratRodWithoutDampingView =
          ::elastica::cosserat_rod::CosseratRodPlugin<
              ::elastica::cosserat_rod::DefaultCosseratRodTraits,
              ::blocks::BlockView, detail::CosseratRodWithoutDampingComponents>;
      using CosseratRodWithoutDampingConstView =
          ::elastica::cosserat_rod::CosseratRodPlugin<
              ::elastica::cosserat_rod::DefaultCosseratRodTraits,
              ::blocks::ConstBlockView,
              detail::CosseratRodWithoutDampingComponents>;

      using CosseratRodWithoutDampingBlock =
          ::blocks::Block<CosseratRodWithoutDamping>;
      using CosseratRodWithoutDampingBlockSlice =
          ::blocks::BlockSlice<CosseratRodWithoutDampingSlice>;
      using CosseratRodWithoutDampingBlockConstSlice =
          ::blocks::ConstBlockSlice<CosseratRodWithoutDampingConstSlice>;
      using CosseratRodWithoutDampingBlockView =
          ::blocks::BlockView<CosseratRodWithoutDampingView>;
      using CosseratRodWithoutDampingBlockConstView =
          ::blocks::ConstBlockView<CosseratRodWithoutDampingConstView>;

    }  // namespace detail

    // clang-format off
    //**************************************************************************
    /*!\brief Cosserat rod without internal damping in \elastica.
     * \ingroup cosserat_rod
     *
     * \details
     * CosseratRodWithoutDamping models CosseratRods without internal
     * dissipation in \elastica. It leverages the performance the @ref blocks
     * machinery and the flexibility of CosseratRodPlugin enhanced with the
     * composability of @ref cosserat_rod_component to generate a fast, compact
     * and efficient CosseratRod data-structure. Users can utilize this to add
     * cosserat rods to a simulator, as detailed in the tutorials.
     *
     * ### Tags
     * The member data can be accessed using @ref tags. For example, accessing
     * the position of a rod is done as shown below
     * \snippet tutorial_single_rod/tutorial_single_rod.cpp tag_example
     *
     * The Cosserat rod supports other tags, one for each variable it contains,
     * documented below.
     * \copydetails elastica::cosserat_rod::CosseratRodPluginTagsDocsStub
     * \copydetails elastica::cosserat_rod::component::WithCircularCosseratRodTagsDocsStub
     * \copydetails elastica::cosserat_rod::component::WithDiagonalLinearHyperElasticModelTagsDocsStub
     * \copydetails elastica::cosserat_rod::component::WithRodKinematicsTagsDocsStub
     *
     * ### Member functions
     * Additionally, you can also access variables by its member functions. The
     * convention for the member function is `get_{name}` where `{name}` is the
     * tag name in small letters. For example, to access the variable
     * elastica::tags::Position above, you can invoke the `get_position()`
     * member function
     */
    // clang-format on
    using CosseratRodWithoutDamping = detail::CosseratRodWithoutDamping;
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

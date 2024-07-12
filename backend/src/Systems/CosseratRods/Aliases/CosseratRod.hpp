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
#include "Systems/CosseratRods/Components/Names/CosseratRod.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace detail {

      template <typename T, typename B>
      using CosseratRodComponents = ::elastica::SymplecticPolicy<
          T, B, ::elastica::cosserat_rod::component::WithCircularCosseratRod,
          ::elastica::cosserat_rod::component::
              WithExplicitlyDampedDiagonalLinearHyperElasticModel,
          ::elastica::cosserat_rod::component::WithRodKinematics,
          ::elastica::cosserat_rod::component::CosseratRodNameAdapted>;

      using CosseratRod = ::elastica::cosserat_rod::CosseratRodPlugin<
          ::elastica::cosserat_rod::DefaultCosseratRodTraits, ::blocks::Block,
          detail::CosseratRodComponents>;
      using CosseratRodSlice = ::elastica::cosserat_rod::CosseratRodPlugin<
          ::elastica::cosserat_rod::DefaultCosseratRodTraits,
          ::blocks::BlockSlice, detail::CosseratRodComponents>;
      using CosseratRodConstSlice = ::elastica::cosserat_rod::CosseratRodPlugin<
          ::elastica::cosserat_rod::DefaultCosseratRodTraits,
          ::blocks::ConstBlockSlice, detail::CosseratRodComponents>;
      using CosseratRodView = ::elastica::cosserat_rod::CosseratRodPlugin<
          ::elastica::cosserat_rod::DefaultCosseratRodTraits,
          ::blocks::BlockView, detail::CosseratRodComponents>;
      using CosseratRodConstView = ::elastica::cosserat_rod::CosseratRodPlugin<
          ::elastica::cosserat_rod::DefaultCosseratRodTraits,
          ::blocks::ConstBlockView, detail::CosseratRodComponents>;

      using CosseratRodBlock = ::blocks::Block<CosseratRod>;
      using CosseratRodBlockSlice = ::blocks::BlockSlice<CosseratRodSlice>;
      using CosseratRodBlockConstSlice =
          ::blocks::ConstBlockSlice<CosseratRodConstSlice>;
      using CosseratRodBlockView = ::blocks::BlockView<CosseratRodView>;
      using CosseratRodBlockConstView =
          ::blocks::ConstBlockView<CosseratRodConstView>;

    }  // namespace detail

    // clang-format off

    //**************************************************************************
    /*!\brief The main type of Cosserat rod in \elastica.
     * \ingroup cosserat_rod
     *
     * \details
     * CosseratRod is the preferred type for modeling CosseratRods with internal
     * dissipation in \elastica. It leverages the performance the @ref blocks
     * machinery and the flexibility of CosseratRodPlugin enhanced with the
     * composability of  @ref cosserat_rod_component to generate a fast, compact
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
     * \copydetails elastica::cosserat_rod::component::WithExplicitlyDampedDiagonalLinearHyperElasticModelTagsDocsStub
     * \copydetails elastica::cosserat_rod::component::WithRodKinematicsTagsDocsStub
     *
     * ### Member functions
     * Additionally, you can also access variables by its member functions. The
     * convention for the member function is `get_{name}` where `{name}` is the
     * tag name in small letters. For example, to access the variable
     * elastica::tags::Position above, you can invoke the `get_position()` member
     * function
     */
    // clang-format on
    using CosseratRod = detail::CosseratRod;
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

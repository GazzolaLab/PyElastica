#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <type_traits>

#include "Systems/CosseratRods/Traits/PlacementTraits/PlacementTraits.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace tt {

      //************************************************************************
      /*!\brief Placement traits of variables in a staggered Cosserat rod grid
       * \ingroup cosserat_rod_tt
       *
       * \usage
       * For any PlacementTrait collecting placements together (each of which
       * adheres to placement protocol)
       * \snippet PlacementTraits/Test_TypeTraits.cpp type_trait_def
       * \code
       * using result = PlacementTypeTrait<PlacementTraits>;
       * \endcode
       * gives access to type traits within `TypeTrait`, such as IsOnNode,
       * IsOnElement, IsOnVoronoi, IsOnRod
       *
       * \tparam PlacementTraits Traits controlling the placement of Variables,
       * for example see ::elastica::cosserat_rod::PlacementTrait
       */
      template <typename PlacementTraits>
      struct PlacementTypeTrait {
        //**Type definitions****************************************************
        //! Traits of placement
        using PT = PlacementTraits;
        //**********************************************************************

        //**********************************************************************
        /*!\brief Check if `Var` is placed on the nodes in the Cosserat rod grid
         * \ingroup cosserat_rod_tt
         *
         * \requires `Var` is a CosseratRodVariable
         * \effects If `Var` inherits from `OnNode`, then inherits from
         * std::true_type, otherwise inherits from std::false_type
         *
         * \usage
         * For any Cosserat rod variable `Var`
         * \code
         * using result = IsOnNode<Var>;
         * \endcode
         * \metareturns
         * cpp17::bool_constant
         *
         * \semantics
         * If the type `Var` is placed `OnNode` (achieved by inheritance), then
         * \code
         * typename result::type = std::true_type;
         * \endcode
         * otherwise
         * \code
         * typename result::type = std::false_type;
         * \endcode
         *
         * \example
         * \snippet PlacementTraits/Test_TypeTraits.cpp on_node_eg
         *
         * \tparam Var Variable to be checked for placement on node
         */
        template <typename Var>
        struct IsOnNode : std::is_base_of<typename PT::OnNode, Var> {};
        //**********************************************************************

        //**********************************************************************
        /*!\brief Check if `Var` is placed on the elements in the Cosserat rod
         * grid
         * \ingroup cosserat_rod_tt
         *
         * \requires `Var` is a CosseratRodVariable
         * \effects If `Var` inherits from `OnElement`, then inherits from
         * std::true_type, otherwise inherits from std::false_type
         *
         * \usage
         * For any Cosserat rod variable `Var`
         * \code
         * using result = IsOnElement<Var>;
         * \endcode
         * \metareturns
         * cpp17::bool_constant
         *
         * \semantics
         * If the type `Var` is placed `OnElement` (achieved by inheritance),
         * then
         * \code
         * typename result::type = std::true_type;
         * \endcode
         * otherwise
         * \code
         * typename result::type = std::false_type;
         * \endcode
         *
         * \example
         * \snippet PlacementTraits/Test_TypeTraits.cpp on_element_eg
         *
         * \tparam Var Variable to be checked for placement on element
         */
        template <typename Var>
        struct IsOnElement : std::is_base_of<typename PT::OnElement, Var> {};
        //**********************************************************************

        //**********************************************************************
        /*!\brief Check if `Var` is placed on the voronois in the Cosserat rod
         * grid
         * \ingroup cosserat_rod_tt
         *
         * \requires `Var` is a CosseratRodVariable
         * \effects If `Var` inherits from `OnVoronoi`, then inherits from
         * std::true_type, otherwise inherits from std::false_type
         *
         * \usage
         * For any Cosserat rod variable `Var`
         * \code
         * using result = IsOnVoronoi<Var>;
         * \endcode
         * \metareturns
         * cpp17::bool_constant
         *
         * \semantics
         * If the type `Var` is placed `OnVoronoi` (achieved by inheritance),
         * then
         * \code
         * typename result::type = std::true_type;
         * \endcode
         * otherwise
         * \code
         * typename result::type = std::false_type;
         * \endcode
         *
         * \example
         * \snippet PlacementTraits/Test_TypeTraits.cpp on_voronoi_eg
         *
         * \tparam Var Variable to be checked for placement on voronoi
         */
        template <typename Var>
        struct IsOnVoronoi : std::is_base_of<typename PT::OnVoronoi, Var> {};
        //**********************************************************************

        //**********************************************************************
        /*!\brief Check if `Var` is placed on the rods in the Cosserat rod
         * grid
         * \ingroup cosserat_rod_tt
         *
         * \requires `Var` is a CosseratRodVariable
         * \effects If `Var` inherits from `OnRod`, then inherits from
         * std::true_type, otherwise inherits from std::false_type
         *
         * \usage
         * For any Cosserat rod variable `Var`
         * \code
         * using result = IsOnRod<Var>;
         * \endcode
         * \metareturns
         * cpp17::bool_constant
         *
         * \semantics
         * If the type `Var` is placed `OnRod` (achieved by inheritance),
         * then
         * \code
         * typename result::type = std::true_type;
         * \endcode
         * otherwise
         * \code
         * typename result::type = std::false_type;
         * \endcode
         *
         * \example
         * \snippet PlacementTraits/Test_TypeTraits.cpp on_rod_eg
         *
         * \tparam Var Variable to be checked for placement on rod
         */
        template <typename Var>
        struct IsOnRod : std::is_base_of<typename PT::OnRod, Var> {};
        //**********************************************************************
      };
      //************************************************************************

    }  // namespace tt

  }  // namespace cosserat_rod

}  // namespace elastica

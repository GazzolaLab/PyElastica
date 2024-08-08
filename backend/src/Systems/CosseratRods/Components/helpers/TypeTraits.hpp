#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/CosseratRods/Components/helpers/Types.hpp"
#include "Systems/common/Components/TypeTraits.hpp"
#include "Utilities/TypeTraits/IsDetected.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace component {

      namespace tt {

        using ::elastica::tt::IsComponent;

        //**************************************************************************
        /*!\brief Check whether a given type `C` is a valid ElasticityComponent
         * \ingroup systems
         *
         * \details
         * Inherits from std::true_type if `C` is a valid elasticity component
         * template, deriving from ElasticityComponent, otherwise inherits from
         * std::false_type.
         *
         * \usage
         * For any type `C`,
         * \code
         * using result = IsElasticityComponent<C>;
         * \endcode
         *
         * \metareturns
         * cpp17::bool_constant
         *
         * \semantics
         * If the type `C` is derived from ElasticityComponent<C>, then
         * \code
         * typename result::type = std::true_type;
         * \endcode
         * otherwise
         * \code
         * typename result::type = std::false_type;
         * \endcode
         *
         * \example
         * \snippet helpers/Test_TypeTraits.cpp is_elasticity_component_example
         *
         * \tparam C : the type to check
         *
         * \see ElasticityComponent
         */
        template <typename C>
        struct IsElasticityComponent
            : public ::tt::IsCRTP<
                  ::elastica::cosserat_rod::component::ElasticityComponent, C> {
        };
        //**************************************************************************

        template <typename T>
        using elasticity_component_t = typename T::elasticity_component;

        template <typename T>
        struct ElasticityComponentTrait
            : public ::tt::detected_or<NoSuchType, elasticity_component_t, T> {
        };

        template <typename T>
        using elasticity_component_trait_t =
            typename ElasticityComponentTrait<T>::type;

      }  // namespace tt

    }  // namespace component

  }  // namespace cosserat_rod

}  // namespace elastica

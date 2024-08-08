#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsDetected.hpp"

namespace elastica {

  namespace cosserat_rod {

    namespace tt {

      //========================================================================
      //
      //  ALIAS DECLARATIONS
      //
      //========================================================================

      //************************************************************************
      /*!\brief Gets the nested `ElasticityModel` from `T`
       * \ingroup cosserat_rod
       *
       * The elasticity_model_t alias declaration provides a convenient shortcut
       * to access the nested `ElasticityModel` type definition of the given
       * type \a T. The following code example shows both ways to access the
       * nested type definition:
       *
       * \example
       * \code
       *   using Type1 = typename T::ElasticityModel;
       *   using Type2 = elasticity_model_t<T>;
       * \endcode
       *
       * \see Variable, blocks::protocols::Variable
       */
      /// [elasticity_model_t]
      template <typename T>
      using elasticity_model_t = typename T::ElasticityModel;
      /// [elasticity_model_t]
      //************************************************************************

      // not sure if this is needed in the first place
      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      template <class X>
      struct GetBaseElasticityModel {
        // if not detected use X, else use X::ElasticityModel
        using type = ::tt::detected_or_t<X, elasticity_model_t, X>;
      };

      template <class X>
      using GetBaseElasticityModel_t = typename GetBaseElasticityModel<X>::type;
      /*! \endcond */
      //************************************************************************

    }  // namespace tt

  }  // namespace cosserat_rod

}  // namespace elastica

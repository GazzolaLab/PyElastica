#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/States/TypeTraits/Aliases.hpp"
#include "Utilities/TypeTraits/IsDetected.hpp"

namespace elastica {

  namespace states {

    namespace tt {

      //************************************************************************
      /*!\brief Check whether the operations of a given state `ST` is vectorized
       * \ingroup states_tt
       *
       * \details
       * Inherits from std::true_type if the state expression `ST` executes its
       * operations in a vectorized fashion, otherwise inherits from
       * std::false_type.
       *
       * \usage
       * For any type `ST`,
       * \code
       * using result = ::elastica::states::tt::IsVectorized<ST>;
       * \endcode
       *
       * \metareturns
       * cpp17::bool_constant
       *
       * \semantics
       * If the state type `ST` carries out vectorized operations, then
       * \code
       * typename result::type = std::true_type;
       * \endcode
       * otherwise
       * \code
       * typename result::type = std::false_type;
       * \endcode
       * We define having "vectorized" operations for `ST` by obtaining
       * ::elastica::states::tt::is_vectorized_t for `ST` if it exists, else we
       * set the result to std::false_type
       *
       * \example
       * \snippet Test_IsVectorized.cpp is_vectorized_eg
       *
       * \tparam ST : the type to check
       */
      template <typename ST>
      struct IsVectorized
          : public ::tt::detected_or_t<std::false_type, is_vectorized_t, ST> {};
      //************************************************************************

    }  // namespace tt

  }  // namespace states

}  // namespace elastica

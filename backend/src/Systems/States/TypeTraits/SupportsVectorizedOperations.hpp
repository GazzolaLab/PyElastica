#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <type_traits>
#include <utility>  // declval

#include "Utilities/TypeTraits/IsDetected.hpp"

namespace elastica {

  namespace states {

    namespace tt {

      namespace detail {

        // [support_requirements]
        template <typename T>
        using addable = decltype(std::declval<T>() + std::declval<T>());
        template <typename T>
        using right_multiplicable =
            decltype(std::declval<T>() * std::declval<double>());
        template <typename T>
        using left_multiplicable =
            decltype(std::declval<double>() * std::declval<T>());
        // [support_requirements]

        template <typename T>
        using add_assignable =
            decltype(std::declval<T>() += (std::declval<T const&>()));

        // vectorized if it has, for the same type T, an addition operator
        // defined and a multiplication operator with a double.
        template <typename T>
        using type_supports_vectorized_operations =
            cpp17::void_t<addable<T>,
                          // for some reason add assignabliity fails with blaze
                          // types add_assignable<T>,
                          left_multiplicable<T>, right_multiplicable<T>>;
      }  // namespace detail

      //************************************************************************
      /*!\brief Check whether a given type `T` supports vectorized operations
       * \ingroup states_tt
       *
       * \details
       * Inherits from std::true_type if `T` supports vectorized operations
       * otherwise inherits from std::false_type.
       *
       * \usage
       * For any type `T`,
       * \code
       * using result = ::elastica::states::tt::SupportsVectorizedOperations<T>;
       * \endcode
       *
       * \metareturns
       * cpp17::bool_constant
       *
       * \semantics
       * If the type `T` supports vectorization, then
       * \code
       * typename result::type = std::true_type;
       * \endcode
       * otherwise
       * \code
       * typename result::type = std::false_type;
       * \endcode
       * We define "supports vectorization" by checking for the following
       * requirements on the type `T`
       * \snippet this support_requirements
       *
       * \example
       * \snippet Test_SupportsVectorizedOperations.cpp supports_vectorized_eg
       *
       * \tparam T : the type to check
       */
      template <typename T>
      struct SupportsVectorizedOperations
          : public ::tt::is_detected<
                detail::type_supports_vectorized_operations, T> {};
      //************************************************************************

    }  // namespace tt

  }  // namespace states

}  // namespace elastica

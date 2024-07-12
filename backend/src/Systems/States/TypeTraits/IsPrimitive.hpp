#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

///
#include "Systems/States/Expressions/OrderTags/Types.hpp"
///
#include "Systems/States/TypeTraits/Aliases.hpp"
#include "Utilities/TypeTraits/Cpp17.hpp"

namespace elastica {

  namespace states {

    namespace tt {

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Check whether the given state `ST` is a primitive state
       * \ingroup states_tt
       *
       * \details
       * Inherits from std::true_type if the state expression `ST` is a
       * primitive state otherwise inherits from std::false_type.
       *
       * \usage
       * For any type `ST`,
       * \code
       * using result = ::elastica::states::tt::IsPrimitive<ST>;
       * \endcode
       *
       * \metareturns
       * cpp17::bool_constant
       *
       * \semantics
       * If the state type `ST` is a primitive state in the ODE system, i.e.
       * tagged with elastica::states::tags::PrimitiveTag, then
       * \code
       * typename result::type = std::true_type;
       * \endcode
       * otherwise
       * \code
       * typename result::type = std::false_type;
       * \endcode
       *
       * \example
       * \snippet Test_IsPrimitive.cpp is_primitive_eg
       *
       * \tparam ST : the type to check
       */
      template <typename ST>
      struct IsPrimitive
          : public std::is_same<order_t<ST>, tags::PrimitiveTag> {};
      //************************************************************************

      //************************************************************************
      /*!\brief Check whether the given state `ST` is not a primitive state
       * \ingroup states_tt
       *
       * \details
       * Inherits from std::true_type if the state expression `ST` is NOT a
       * primitive state otherwise inherits from std::false_type.
       *
       * \see IsPrimitive
       */
      template <typename ST>
      struct IsNotPrimitive : public cpp17::negation<IsPrimitive<ST>> {};
      //************************************************************************

      //************************************************************************
      /*!\brief Auxiliary variable template for the IsPrimitive type trait.
       * \ingroup states_tt
       *
       * The is_primitive_v variable template provides a convenient shortcut to
       * access the nested `value` of the IsPrimitive class template. For
       * instance, given the type `T` the following two statements are
       * identical:
       *
       * \example
       * \code
       * using namespace elastica::states::tt;
       * constexpr bool value1 = IsPrimitive<T>::value;
       * constexpr bool value2 = is_primitive_v<T>;
       * \endcode
       *
       * \see IsPrimitive
       */
      template <typename ST>
      constexpr bool is_primitive_v = IsPrimitive<ST>::value;
      //************************************************************************

    }  // namespace tt

  }  // namespace states

}  // namespace elastica

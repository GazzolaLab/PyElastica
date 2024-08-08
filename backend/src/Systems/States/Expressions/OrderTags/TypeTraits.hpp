#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <utility>  // declval

#include "Systems/States/Expressions/OrderTags/Types.hpp"
#include "Systems/States/TypeTraits/Aliases.hpp"

namespace elastica {

  namespace states {

    namespace tt {

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      namespace detail {
        template <typename T>
        struct LowerOrder;

        template <>
        struct LowerOrder<tags::DerivativeTag> {
          using type = tags::internal::DerivativeMultipliedByTimeTag;
        };

        template <>
        struct LowerOrder<tags::DoubleDerivativeTag> {
          using type = tags::internal::DoubleDerivativeMultipliedByTimeTag;
        };
      }  // namespace detail

      //************************************************************************
      /*!\brief Metafunction lowering the derivative order of a state type `ST`
       * \ingroup states_tt
       *
       * \details
       * Lowers the derivative order of a state type `ST` by one, for use in
       * multiplication of `ST` derivative expressions with a time-delta type.
       *
       * \usage
       * For any state type `ST`,
       * \code
       * using result = ::elastica::states::tt::lower_order_t<ST>;
       * \endcode
       *
       * \semantics
       * For the state type `ST`, we effectively lower the order by 1. i.e. if
       * `ST` is `::elastica::states::states::DerivativeSTag` (a derivative of
       * order 1), `result` indicates a type with derivative of order 0.
       * Similarly if `ST` is `::elastica::states::states::DoubleDerivativeSTag`
       * (a derivative of order 2), `result` indicates a type with derivative of
       * order 1.
       *
       * \example
       * \snippet OrderTags/Test_TypeTraits.cpp lower_order_t_eg
       *
       * \tparam ST : the state type whose derivative order needs to be lowered.
       */
      template <typename ST>
      using lower_order_t = typename detail::LowerOrder<order_t<ST>>::type;
      //************************************************************************

      namespace detail {
        /// These two are for enabling order in StateCollections and are not
        /// executed in the SO3 path
        auto common_order_backend(
            tags::PrimitiveTag&, tags::internal::DerivativeMultipliedByTimeTag&)
            -> tags::internal::DerivativeMultipliedByTimeTag;
        auto common_order_backend(
            tags::internal::DerivativeMultipliedByTimeTag&, tags::PrimitiveTag&)
            -> tags::internal::DerivativeMultipliedByTimeTag;
        ///

        auto common_order_backend(
            tags::internal::DerivativeMultipliedByTimeTag&,
            tags::internal::DerivativeMultipliedByTimeTag&)
            -> tags::internal::DerivativeMultipliedByTimeTag;
        auto common_order_backend(
            tags::internal::DoubleDerivativeMultipliedByTimeTag&,
            tags::DerivativeTag&)
            -> tags::internal::DoubleDerivativeMultipliedByTimeTag;
        auto common_order_backend(
            tags::DerivativeTag&,
            tags::internal::DoubleDerivativeMultipliedByTimeTag&)
            -> tags::internal::DoubleDerivativeMultipliedByTimeTag;
        auto common_order_backend(
            tags::internal::DoubleDerivativeMultipliedByTimeTag&,
            tags::internal::DoubleDerivativeMultipliedByTimeTag&)
            -> tags::internal::DoubleDerivativeMultipliedByTimeTag;

        template <typename First, typename Second>
        struct common_order {
          using type = decltype(common_order_backend(std::declval<First&>(),
                                                     std::declval<Second&>()));
        };

      }  // namespace detail

      //************************************************************************
      /*!\brief Metafunction returning a common type if state types `ST1` and
       * `ST2` have the same order of derivatives.
       * \ingroup states_tt
       *
       * \details
       * Returns the common type of two state types `ST1` and `ST2` having the
       * same order of derivatives.
       *
       * \usage
       * For any two state types `ST1`, `ST2`,
       * \code
       * using result = ::elastica::states::tt::common_order_t<ST>;
       * \endcode
       *
       * \semantics
       * If `ST1` and `ST2` have the same order of derivatives, `result`
       * contains the common type between `ST1` and `ST2`. Else if `ST1` and
       * `ST2` do not have the same order of derivatives, a static error
       * is thrown and compilation is halted.
       *
       * \example
       * \snippet OrderTags/Test_TypeTraits.cpp common_order_t_eg
       *
       * \tparam ST1 : the first state type
       * \tparam ST1 : the second state type
       *
       * \see std::common_type_t
       */
      template <typename ST1, typename ST2>
      using common_order_t =
          typename detail::common_order<order_t<ST1>, order_t<ST2>>::type;
      //************************************************************************

    }  // namespace tt

  }  // namespace states

}  // namespace elastica

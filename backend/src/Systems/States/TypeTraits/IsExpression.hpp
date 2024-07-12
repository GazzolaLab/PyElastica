//==============================================================================
/*!
//  \file
//  \brief IsExpression type trait
//
//  Copyright (C) 2020-2020 Tejaswin Parthasarathy - All Rights Reserved
//  Copyright (C) 2020-2020 MattiaLab - All Rights Reserved
//
//  Distributed under the MIT License.
//  See LICENSE.txt for details.
//
//  The original file is abridged from the Blaze library with the following
//  license:
//
//  Copyright (C) 2012-2020 Klaus Iglberger - All Rights Reserved
//  This file is part of the Blaze library. You can redistribute it and/or
//  modify it under the terms of the New (Revised) BSD License. Redistribution
//  and use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//  1. Redistributions of source code must retain the above copyright notice,
//  this list of conditions and the following disclaimer.
//  2. Redistributions in
//  binary form must reproduce the above copyright notice, this list of
//  conditions and the following disclaimer in the documentation and/or other
//  materials provided with the distribution.
//  3. Neither the names of the Blaze development group nor the names of its
//  contributors may be used to endorse or promote products derived from this
//  software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//  AND ANY   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//  ARE DISCLAIMED. IN NO EVENT   SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
//  LIABLE FOR ANY DIRECT, INDIRECT,   INCIDENTAL, SPECIAL, EXEMPLARY, OR
//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
//  ARISING IN   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
*/
//==============================================================================

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>

#include "Systems/States/Expressions/Expr/StateExpr.hpp"
#include "Utilities/IgnoreUnused.hpp"
#include "Utilities/TypeTraits.hpp"

namespace elastica {

  namespace states {

    namespace tt {

      namespace detail {

        //**********************************************************************
        /*! \cond ELASTICA_INTERNAL */
        /*!\brief Auxiliary helper functions for the IsExpression type trait.
        // \ingroup math_type_traits
        */
        template <typename U>
        std::true_type is_expression_backend(const volatile StateExpr<U>*);
        IGNORE_UNUSED std::false_type is_expression_backend(...);
        /*! \endcond */
        //**********************************************************************

      }  // namespace detail

      //========================================================================
      //
      //  CLASS DEFINITION
      //
      //========================================================================

      //************************************************************************
      /*!\brief Check whether the given type `Expr` is a valid expression
       * \ingroup states_tt
       *
       * \details
       * Inherits from std::true_type if the type `Expr` is a valid state
       * expression, otherwise inherits from std::false_type.
       *
       * \usage
       * For any type `Expr`,
       * \code
       * using result = ::elastica::states::tt::IsExpression<Expr>;
       * \endcode
       *
       * \metareturns
       * cpp17::bool_constant
       *
       * \semantics
       * If the state type `Expr` is an expression within the states expression
       * template mechanism, i.e. it derives from ::elastica::states::StateExpr
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
       * \snippet Test_IsExpression.cpp is_expression_eg
       *
       * \tparam Expr : the type to check
       */
      template <typename Expr>
      struct IsExpression : public decltype(detail::is_expression_backend(
                                std::declval<Expr*>())) {};
      //************************************************************************

      //************************************************************************
      /*! \cond ELASTICA_INTERNAL */
      /*!\brief Specialization of the IsExpression type trait for references.
      // \ingroup states_tt
      */
      template <typename T>
      struct IsExpression<T&> : public std::false_type {};
      /*! \endcond */
      //************************************************************************

      //************************************************************************
      /*!\brief Auxiliary variable template for the IsExpression type trait.
       * \ingroup states_tt
       *
       * The is_expression_v variable template provides a convenient shortcut
       * to access the nested `value` of the IsExpression class template. For
       * instance, given the type `Expr` the following two statements are
       * identical:
       *
       * \example
       * \code
       * using namespace elastica::states::tt;
       * constexpr bool value1 = IsExpression<Expr>::value;
       * constexpr bool value2 = is_expression_v<Expr>;
       * \endcode
       *
       * \see IsExpression
       */
      template <typename Expr>
      constexpr bool is_expression_v = IsExpression<Expr>::value;
      //************************************************************************

    }  // namespace tt

  }  // namespace states

}  // namespace elastica

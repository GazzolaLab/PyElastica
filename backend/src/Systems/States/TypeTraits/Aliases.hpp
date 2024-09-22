//==============================================================================
/*!
//  \file
//  \brief Helper Aliases for state expressions
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

namespace elastica {

  namespace states {

    namespace tt {

      //========================================================================
      //
      //  ALIAS DECLARATIONS
      //
      //========================================================================

      //************************************************************************
      /*!\brief Alias declaration for nested `LeftOperand` type definition.
       * \ingroup states_tt
       *
       * The left_operand_t alias declaration provides a convenient shortcut to
       * access the nested `LeftOperand` type definition of the given type `T`.
       * The  following code example shows both ways to access the nested type
       * definition:
       *
       * \example
       * \code
       * using Type1 = typename T::LeftOperand;
       * using Type2 = left_operand_t<T>;
       * \endcode
       *
       * \tparam T The type containing the nested typedef
       */
      template <typename T>
      using left_operand_t = typename T::LeftOperand;
      //************************************************************************

      //************************************************************************
      /*!\brief Alias declaration for nested `RightOperand` type definition.
       * \ingroup states_tt
       *
       * The right_operand_t alias declaration provides a convenient shortcut to
       * access the nested `RightOperand` type definition of the given type `T`.
       * The following code example shows both ways to access the nested type
       * definition:
       *
       * \example
       * \code
       * using Type1 = typename T::RightOperand;
       * using Type2 = right_operand_t<T>;
       * \endcode
       *
       * \tparam T The type containing the nested typedef
       */
      template <typename T>
      using right_operand_t = typename T::RightOperand;
      //************************************************************************

      //************************************************************************
      /*!\brief Alias declaration for nested `Order` type definition.
       * \ingroup states_tt
       *
       * The order_t alias declaration provides a convenient shortcut to access
       * the nested `Order` type definition of the given type `T`. The following
       * code example shows both ways to access the nested type definition:
       *
       * \example
       * \code
       * using Type1 = typename T::Order;
       * using Type2 = order_t<T>;
       * \endcode
       *
       * \tparam T The type containing the nested typedef
       */
      template <typename T>
      using order_t = typename T::Order;
      //************************************************************************

      //************************************************************************
      /*!\brief Alias declaration for nested `is_vectorized` type definition.
       * \ingroup states_tt
       *
       * The is_vectorized_t alias declaration provides a convenient shortcut to
       * access the nested `is_vectorized` type definition of the given type
       * `T`. The following code example shows both ways to access the nested
       * type definition:
       *
       * \example
       * \code
       * using Type1 = typename T::is_vectorized;
       * using Type2 = is_vectorized_t<T>;
       * \endcode
       *
       * \tparam T The type containing the nested typedef
       */
      template <typename T>
      using is_vectorized_t = typename T::is_vectorized;
      //************************************************************************

    }  // namespace tt

  }  // namespace states

}  // namespace elastica

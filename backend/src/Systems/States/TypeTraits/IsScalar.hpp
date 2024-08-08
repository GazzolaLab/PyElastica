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

#include "Systems/States/TypeTraits/HasOrder.hpp"
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
      /*!\brief Check whether the given type `T` is a scalar type within the
       * context of the expression template states
       * \ingroup states_tt
       *
       * \details
       * Inherits from std::true_type if the state expression `T` is a
       * scalar, otherwise inherits from std::false_type.
       *
       * \usage
       * For any type `T`,
       * \code
       * using result = ::elastica::states::tt::IsScalar<T>;
       * \endcode
       *
       * \metareturns
       * cpp17::bool_constant
       *
       * \semantics
       * If the state type `T` is a scalar in the ODE system i.e. neither a
       * state, nor an expression, reference or pointer type, then
       * \code
       * typename result::type = std::true_type;
       * \endcode
       * otherwise
       * \code
       * typename result::type = std::false_type;
       * \endcode
       *
       * \example
       * \snippet Test_IsScalar.cpp is_scalar_eg
       *
       * \tparam T : the type to check
       */
      // We using De Morgan's laws here : not (A or B) = not A and not B;
      template <typename T>
      struct IsScalar
          : public cpp17::negation<cpp17::disjunction<
                HasOrder<T>, std::is_pointer<T>, std::is_reference<T>>> {};
      //************************************************************************

      //************************************************************************
      /*!\brief Auxiliary variable template for the IsScalar type trait.
       * \ingroup states_tt
       *
       * The is_scalar_v variable template provides a convenient shortcut to
       * access the nested `value` of the IsScalar class template. For instance,
       * given the type `T` the following two statements are identical:
       *
       * \example
       * \code
       * constexpr bool value1 = elastica::IsScalar<T>::value;
       * constexpr bool value2 = elastica::is_scalar_v<T>;
       * \endcode
       *
       * \see IsScalar
       */
      template <typename T>
      constexpr bool is_scalar_v = IsScalar<T>::value;
      //************************************************************************

    }  // namespace tt

  }  // namespace states

}  // namespace elastica

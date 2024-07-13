// Distributed under the MIT License.
// See LICENSE.txt for details.

// Source :
// https://raw.githubusercontent.com/sxs-collaboration/spectre/develop/src/Utilities/ForceInline.hpp

/// \file
/// Defines macro to always inline a function.

#pragma once

#if ELASTICA_USE_ALWAYS_INLINE
#if defined(__GNUC__)
/// \ingroup UtilitiesGroup
/// Always inline a function. Only use this if you benchmarked the code.
#define ELASTICA_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
/// \ingroup UtilitiesGroup
/// Always inline a function. Only use this if you benchmarked the code.
#define ELASTICA_ALWAYS_INLINE __forceinline
#endif
#else
/// \ingroup UtilitiesGroup
/// Always inline a function. Only use this if you benchmarked the code.
#define ELASTICA_ALWAYS_INLINE inline
#endif

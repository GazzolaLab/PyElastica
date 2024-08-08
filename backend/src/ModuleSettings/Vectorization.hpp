#pragma once

//******************************************************************************
// Includes
//******************************************************************************

// Ideally the check here would include system vectorization macro checks, such
// as in blaze.
// But directly including blaze's vectorization also includes intrinsics headers
// which we want to avoid.

//******************************************************************************
/*!\brief Vectorization flag for some Elastica++ kernels
 * \ingroup config
 *
 * This macro enables use of vectorized kernels when available.
 */
#if defined(__APPLE__) && defined(__arm64__)
#define ELASTICA_USE_VECTORIZATION 0
#else
#define ELASTICA_USE_VECTORIZATION BLAZE_USE_VECTORIZATION
#endif

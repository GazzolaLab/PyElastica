#pragma once

//******************************************************************************
/*!\brief Selection of the default backend for quadrature
 * \ingroup default_config
 *
 * The following execution policies are available:
 *   - scalar
 *   - simd
 *   - blaze
 *
 * This can be customized if needed in the client file.
 */
#ifndef ELASTICA_COSSERATROD_LIB_QUADRATURE_BACKEND
#define ELASTICA_COSSERATROD_LIB_QUADRATURE_BACKEND simd
#endif
//******************************************************************************

//******************************************************************************
/*!\brief Selection of the default backend for difference
 * \ingroup default_config
 *
 * The following execution policies are available:
 *   - scalar
 *   - simd
 *   - blaze
 *
 * This can be customized if needed in the client file.
 */
#ifndef ELASTICA_COSSERATROD_LIB_DIFFERENCE_BACKEND
#define ELASTICA_COSSERATROD_LIB_DIFFERENCE_BACKEND simd
#endif
//******************************************************************************

//******************************************************************************
/*!\brief Selection of the default backend for matvec
 * \ingroup default_config
 *
 * The following execution policies are available:
 *   - scalar
 *   - simd
 *   - blaze
 *
 * This can be customized if needed in the client file.
 */
#ifndef ELASTICA_COSSERATROD_LIB_MATVEC_BACKEND
#define ELASTICA_COSSERATROD_LIB_MATVEC_BACKEND simd
#endif
//******************************************************************************

//******************************************************************************
/*!\brief Selection of the default backend for vector-scalar division
 * \ingroup default_config
 *
 * The following execution policies are available:
 *   - scalar
 *   - simd
 *   - blaze
 *
 * This can be customized if needed in the client file.
 */
#ifndef ELASTICA_COSSERATROD_LIB_VECSCALARDIV_BACKEND
#define ELASTICA_COSSERATROD_LIB_VECSCALARDIV_BACKEND simd
#endif
//******************************************************************************

//******************************************************************************
/*!\brief Selection of the default backend for vector-scalar multiplication
 * \ingroup default_config
 *
 * The following execution policies are available:
 *   - scalar
 *   - simd
 *   - blaze
 *
 * This can be customized if needed in the client file.
 */
#ifndef ELASTICA_COSSERATROD_LIB_VECSCALARMULT_BACKEND
#define ELASTICA_COSSERATROD_LIB_VECSCALARMULT_BACKEND simd
#endif
//******************************************************************************

//******************************************************************************
/*!\brief Selection of the default backend for cross products
 * \ingroup default_config
 *
 * The following execution policies are available:
 *   - scalar
 *   - simd
 *   - blaze
 *
 * This can be customized if needed in the client file.
 */
#ifndef ELASTICA_COSSERATROD_LIB_CROSSPRODUCT_BACKEND
#define ELASTICA_COSSERATROD_LIB_CROSSPRODUCT_BACKEND simd
#endif
//******************************************************************************

//******************************************************************************
/*!\brief Selection of the default backend for inv rotate divide
 * \ingroup default_config
 *
 * The following execution policies are available:
 *   - scalar
 *   - simd
 *   - blaze
 *
 * This can be customized if needed in the client file.
 */
#ifndef ELASTICA_COSSERATROD_LIB_INV_ROTATE_DIVIDE_BACKEND
#define ELASTICA_COSSERATROD_LIB_INV_ROTATE_DIVIDE_BACKEND simd
#endif
//******************************************************************************

//******************************************************************************
/*!\brief Selection of the default backend for SO3+=
 * \ingroup default_config
 *
 * The following execution policies are available:
 *   - scalar
 *   - simd
 *
 * This can be customized if needed in the client file.
 */
#ifndef ELASTICA_COSSERATROD_LIB_SO3_ADDITION_BACKEND
#define ELASTICA_COSSERATROD_LIB_SO3_ADDITION_BACKEND simd
#endif
//******************************************************************************

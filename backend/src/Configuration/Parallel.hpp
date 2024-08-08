#pragma once

//******************************************************************************
/*!\brief Compilation switch for the (de-)activation of the shared-memory
 * parallelization
 * \ingroup config parallel
 *
 * This compilation switch enables/disables the shared-memory parallelization.
 * In case the switch is set to 1 (i.e. in case the shared-memory
 * parallelization is enabled), \elastica is allowed to execute operations in
 * parallel. In case the switch is set to 0 (i.e. parallelization
 * is disabled), \elastica is restricted from executing operations in parallel.
 *
 * Possible settings for the shared-memory parallelization switch:
 *  - Deactivated: \b 0
 *  - Activated  : \b 1 (default)
 *
 * \note It is possible to (de-)activate the shared-memory parallelization via
 * command line or by defining this symbol manually before including any
 * Blaze header file:
 *
 * \example
 * \code
 * g++ ... -DELASTICA_USE_SHARED_MEMORY_PARALLELIZATION=1 ...
 * \endcode
 *
 * \code
 * #define ELASTICA_USE_SHARED_MEMORY_PARALLELIZATION 1
 * #include <elastica/elastica.hpp>
 * \endcode
 *
 */
// #ifndef ELASTICA_USE_SHARED_MEMORY_PARALLELIZATION
// #define ELASTICA_USE_SHARED_MEMORY_PARALLELIZATION @ELASTICA_SMP@
// #endif
//******************************************************************************

//******************************************************************************
/*!\brief Compile time hint for number of threads to be used in an elastica
 * application
 * \ingroup parallel
 */
constexpr std::size_t elastica_threads_hint() { return 2UL; }
//******************************************************************************

//******************************************************************************
/*!\brief Environment variable to set a hint for the number of threads used
 * \ingroup parallel
 *
 * The macro defines the environment variable used to provide hint for the
 * number of threads to set for Elastica++ applications.
 * For user applications (i.e. when writing your own main-file), this variable
 * does not factor in : rather the user can set the number of threads they want
 * using the specific parallelism library they employed.
 */
#define ENV_ELASTICA_NUM_THREADS "ELASTICA_NUM_THREADS"
  //****************************************************************************

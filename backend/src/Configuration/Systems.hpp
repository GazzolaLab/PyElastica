#pragma once

//******************************************************************************
/*!\brief Selection of the default admissible systems
 * \ingroup default_config
 *
 * This macro selects the default admissible \systems. The purpose of this
 * \a typelist is to add tags indicating the admissible \systems in the
 * simulation that comprise the degrees of freedom. The following \system tags
 * are available:
 *
 *   - elastica::cosserat_rod::CosseratRod
 *   - elastica::cosserat_rod::CosseratRodWithoutDamping
 *   - elastica::rigid_body::Sphere
 *
 * Consult the documentation page of elastica::PhysicalSystemPlugins for more
 * information. This can be customized as needed in the client file.
 */
#ifndef ELASTICA_DEFAULT_ADMISSIBLE_PLUGINS_FOR_SYSTEMS
#define ELASTICA_DEFAULT_ADMISSIBLE_PLUGINS_FOR_SYSTEMS \
  tmpl::list<elastica::cosserat_rod::CosseratRod>
#endif
//******************************************************************************

//******************************************************************************
/*!\brief Selection of the default policy for adding new systems to Blocks
 * within the simulator
 * \ingroup default_config
 *
 * This macro select the default Blocking policy for adding Systems within
 * \elastica. The following (simple) blocking policies are available:
 *
 *   - elastica::configuration::RestrictSizeAcrossBlockTypesPolicy
 *   - elastica::configuration::LooselyRestrictSizeAcrossBlockTypesPolicy
 *   - elastica::configuration::AlwaysNewAcrossBlockTypesPolicy
 *   - elastica::configuration::AlwaysSameAcrossBlockTypesPolicy
 *
 * This can be customized if needed in the client file. More complicated
 * policies are also possible by composition
 */
#ifndef ELASTICA_DEFAULT_BLOCKING_POLICY_FOR_SYSTEMS
#define ELASTICA_DEFAULT_BLOCKING_POLICY_FOR_SYSTEMS \
  elastica::configuration::RestrictSizeAcrossBlockTypesPolicy
#endif
//******************************************************************************

//******************************************************************************
/*!\brief Selection of the size parameter for the default policy choosen in
 * ELASTICA_DEFAULT_BLOCKING_CRITERIA_FOR_SYSTEMS()
 * \ingroup default_config
 *
 * This value specifies the default size parameter for specifying the blocking
 * policy.
 */
constexpr std::size_t default_blocking_policy_size() { return 1024UL; }
//******************************************************************************

//******************************************************************************
/*!\brief Selection of the default iteration behavior over multiple types of
 * \elastica systems
 * \ingroup default_config
 *
 * This macro select the default behavior while iterating over multiple types
 * of systems (such as a CosseratRod, Sphere, Capsule) etc. The following
 * execution policies are available:
 *
 *   - elastica::configuration::sequential_policy
 *   - elastica::configuration::parallel_policy
 *
 * This can be customized if needed in the client file.
 */
#ifndef ELASTICA_DEFAULT_ITERATION_POLICY_ACROSS_SYSTEM_TYPES
#define ELASTICA_DEFAULT_ITERATION_POLICY_ACROSS_SYSTEM_TYPES \
  ::elastica::configuration::sequential_policy
#endif
//******************************************************************************

//******************************************************************************
/*!\brief Selection of the default iteration behavior over a container of
 * \elastica systems
 * \ingroup default_config
 *
 * This macro select the default behavior while iterating over multiple systems
 * of the same type. The following execution policies are available:
 *
 *   - elastica::configuration::sequential_policy
 *   - elastica::configuration::parallel_policy
 *   - elastica::configuration::hybrid_policy
 *
 * This can be customized if needed in the client file.
 */
#ifndef ELASTICA_DEFAULT_ITERATION_POLICY_FOR_EACH_SYSTEM_TYPE
#define ELASTICA_DEFAULT_ITERATION_POLICY_FOR_EACH_SYSTEM_TYPE \
  ::elastica::configuration::sequential_policy
#endif
//******************************************************************************

//******************************************************************************
/*!\brief Environment variable to control if potential warnings are shown during
 * system initialization
 * \ingroup systems
 *
 * The macro defines the environment variable that is searched for controlling
 * the behavior of (potential) warnings when initialization systems in
 * Elastica++. By default, potential warnings are enabled. To disable user
 * warnings the environment variable can be set to 1, i.e.
 * \code
 * ELASTICA_NO_SYSTEM_WARN=1 ./<your_application> [options]
 * \endcode
 */
#define ENV_ELASTICA_NO_SYSTEM_WARN "ELASTICA_NO_SYSTEM_WARN"
//****************************************************************************

//******************************************************************************
/*!\brief Selection of the default index checking behavior within systems module
 * \elastica systems
 * \ingroup default_config
 *
 * This macro select the default behavior while slicing/viewing indices within
 * a system. The following execution policies are available:
 *
 *   - checked (checks indices)
 *   - unchecked (does not check indices)
 *
 * This can be customized if needed in the client file.
 */
#ifndef ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK
#define ELASTICA_DEFAULT_SYSTEM_INDEX_CHECK checked
#endif
//****************************************************************************

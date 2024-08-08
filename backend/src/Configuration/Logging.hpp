#pragma once

//******************************************************************************
/*!\brief Selection of the level of logging.
 * \ingroup config logging
 *
 * This macro selects the level of logging, to effectively filter out logs that
 * are not deemed necessary. Any logs below the level chosen are not reported
 * while a logging level comprises all higher logging levels. For instance,
 * if the set ELASTICA_LOG_LEVEL() is `info`, then logs marked with `debug` are
 * not printed, but all errors and warning are also printed to the log file(s).
 * The following log levels can be chosen:
 *
 * - ::elastica::logging::inactive
 * - ::elastica::logging::error
 * - ::elastica::logging::warning
 * - ::elastica::logging::info
 * - ::elastica::logging::debug
 *
 * \see elastica::logging::LoggingLevel
 */
#ifndef ELASTICA_LOG_LEVEL
#define ELASTICA_LOG_LEVEL ::elastica::logging::info
#endif
//******************************************************************************

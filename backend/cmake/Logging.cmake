# Utilities for writing cmake log messages.

# Distributed under the MIT License. See LICENSE for details.

# Global variables
set(error_log_level "1" CACHE INTERNAL "Error log level")
set(warning_log_level "2" CACHE INTERNAL "Warning log level")
set(info_log_level "3" CACHE INTERNAL "Info log level")
set(debug_log_level "4" CACHE INTERNAL "Debug log level")
set(heavy_debug_log_level "5" CACHE INTERNAL "Heavy_debug log level")

set(current_log_level "${info_log_level}" CACHE INTERNAL "Current log level")

function(log_message message)
    if(LOG_MESSAGE_STATUS)
        set(LOG_MESSAGE_PREFIX "[${PROJECT_NAME}]")
        message(STATUS "${LOG_MESSAGE_PREFIX} ${message}")
    endif()
endfunction(log_message)

function(set_logs_on)
    set(LOG_MESSAGE_STATUS "ON" CACHE INTERNAL "Status for log messages")
endfunction(set_logs_on)

function(set_logs_off)
    set(LOG_MESSAGE_STATUS "OFF" CACHE INTERNAL "Status for log messages")
endfunction(set_logs_off)

function(set_log_level log_level)
    if(${log_level} MATCHES "^[0-9]+$")
        if((${log_level} EQUAL ${error_log_level})
           OR (${log_level} GREATER ${error_log_level}))
            set(current_log_level
                "${log_level}"
                CACHE INTERNAL "Current log level")
        endif()
    else()
        if(${log_level} STREQUAL "ERROR")
            set(local_log_level "${error_log_level}")
        elseif(${log_level} STREQUAL "WARNING")
            set(local_log_level "${warning_log_level}")
        elseif(${log_level} STREQUAL "INFO")
            set(local_log_level "${info_log_level}")
        elseif(${log_level} STREQUAL "DEBUG")
            set(local_log_level "${debug_log_level}")
        elseif(${log_level} STREQUAL "HEAVYDEBUG")
            set(local_log_level "${heavy_debug_log_level}")
        endif()
        set(current_log_level
            "${local_log_level}"
            CACHE INTERNAL "Current log level")
    endif()
endfunction(set_log_level)

# set_log_level( "${info_log_level}" )

function(log_error message)
    set(message_log_level "${error_log_level}")
    if((${current_log_level} EQUAL ${message_log_level})
       OR (${current_log_level} GREATER ${message_log_level}))
        set(LOG_MESSAGE_PREFIX "[${PROJECT_NAME}]")
        message(STATUS "${LOG_MESSAGE_PREFIX} ${message}")
    endif()
endfunction(log_error)

function(log_warning message)
    set(message_log_level "${warning_log_level}")
    if((${current_log_level} EQUAL ${message_log_level})
       OR (${current_log_level} GREATER ${message_log_level}))
        set(LOG_MESSAGE_PREFIX "[${PROJECT_NAME}]")
        message(STATUS "${LOG_MESSAGE_PREFIX} ${message}")
    endif()
endfunction(log_warning)

function(log_info message)
    set(message_log_level "${info_log_level}")
    if((${current_log_level} EQUAL ${message_log_level})
       OR (${current_log_level} GREATER ${message_log_level}))
        set(LOG_MESSAGE_PREFIX "[${PROJECT_NAME}]")
        message(STATUS "${LOG_MESSAGE_PREFIX} ${message}")
    endif()
endfunction(log_info)

function(log_debug message)
    set(message_log_level "${debug_log_level}")
    if((${current_log_level} EQUAL ${message_log_level})
       OR (${current_log_level} GREATER ${message_log_level}))
        set(LOG_MESSAGE_PREFIX "[${PROJECT_NAME}]")
        message(STATUS "${LOG_MESSAGE_PREFIX} ${message}")
    endif()
endfunction(log_debug)

function(log_heavy_debug message)
    set(message_log_level "${heavy_debug_log_level}")
    if((${current_log_level} EQUAL ${message_log_level})
       OR (${current_log_level} GREATER ${message_log_level}))
        set(LOG_MESSAGE_PREFIX "[${PROJECT_NAME}]")
        message(STATUS "${LOG_MESSAGE_PREFIX} ${message}")
    endif()
endfunction(log_heavy_debug)

# ================================================
# See if we have env vars to help us find packages
# ================================================
# Gets path, convert backslashes as ${ENV_${var}}
macro(getenv_path VAR)
    set(ENV_${VAR} $ENV{${VAR}})
    # replace won't work if var is blank
    if(ENV_${VAR})
        string(REGEX
                REPLACE "\\\\"
                "/"
                ENV_${VAR}
                ${ENV_${VAR}})
    endif()
endmacro()

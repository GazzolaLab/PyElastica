#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <string>

//******************************************************************************
/*!\brief Returns the demangled name of a compilation symbol
// \ingroup UtilitiesGroup
//
// \usage
   \code
     auto demangled_name = demangle(typeid(int).name());
   \endcode
*/
std::string demangle(char const* name);
//******************************************************************************

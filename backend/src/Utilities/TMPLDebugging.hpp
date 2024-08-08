//==============================================================================
/*!
//  \file
//  \brief Debugging facilities for template meta-programming
//
//  Copyright (C) 2020-2020 Tejaswin Parthasarathy - All Rights Reserved
//  Copyright (C) 2020-2020 MattiaLab - All Rights Reserved
//
//  Distributed under the MIT License.
//  See LICENSE.txt for details.
//
//  Reused with thanks from SpECTRE : https://spectre-code.org/
//  Distributed under the MIT License.
//  See LICENSE.txt for details.
*/
//==============================================================================

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Utilities/TMPL.hpp"

//******************************************************************************
/*!\brief Get compiler error with type of template parameter
// \ingroup utils type_traits
//
// \details
// The compiler error generated when using an object of type
// `TypeDisplayer<...>` contains the types of the template parameters. This
// effectively provides printf-debugging for metaprogramming. For example,
   \code
   TypeDisplayer<std::vector<double>> some_random_name;
   \endcode
// will produce a compiler error that contains the type `std::vector<double,
// std::allocator...>`. TypeDisplayer is extremely useful when debugging
// template metaprograms.
//
// \note
// The TypeDisplayer header should only be included during testing
// and debugging.
//
// \see make_list
*/
template <typename...>
struct TypeDisplayer;
//******************************************************************************

//******************************************************************************
/*!\brief Metafunction to turn a parameter pack into a typelist
// \ingroup utils
//
// This metafunction is really only useful for debugging metaprograms. For
// example, the desired algorithm might be:
//
   \code
   using variables_tags_from_single_tags = tmpl::filter<
       extracted_from_variables,
       tmpl::bind<tmpl::found, tmpl::pin<mutated_tags_list>,
                  tmpl::bind<std::is_same, tmpl::_1, tmpl::parent<tmpl::_1>>>>;
   \endcode
//
// However, getting the `tmpl::pin`, `tmpl::parent`, and `tmpl::bind` calls
// right can be extremely frustrating with little help as to what is going on.
// Let's introduce an error by pinning `tmpl::_1`:
//
   \code
   using variables_tags_from_single_tags = tmpl::filter<
       extracted_from_variables,
       tmpl::bind<tmpl::found, tmpl::pin<mutated_tags_list>,
                  tmpl::bind<std::is_same, tmpl::pin<tmpl::_1>,
                  tmpl::parent<tmpl::_1>>>>;
   \endcode
//
// The result is comparing all values in `extracted_from_variables` to
// themselves. To find this out, replace `tmpl::filter` and `tmpl::found` with
// `tmpl::transform`, and the metafunction `std::is_same` to `make_list`.
// You will then get back a "backtrace" of what the algorithm did, which is
// invaluable for getting the `tmpl::pin` and `tmpl::parent` right. That is,
//
   \code
   using variables_tags_from_single_tags2 = tmpl::transform<
       extracted_from_variables,
       tmpl::bind<tmpl::transform, tmpl::pin<mutated_tags_list>,
                  tmpl::bind<make_list, tmpl::_1, tmpl::parent<tmpl::_1>>>>;

   TypeDisplayer<variables_tags_from_single_tags2> aeou;
   \endcode
//
// You will get an output along the lines of:
//
   \code
   src/DataStructures/DataBox.hpp:1181:40: error: implicit instantiation of
   undefined template
   'TypeDisplayer<brigand::list<brigand::list<
       brigand::list<test_databox_tags::ScalarTag,
       test_databox_tags::Tag0>, brigand::list<test_databox_tags::ScalarTag,
       Tags::Variables<brigand::list<test_databox_tags::ScalarTag,
                       test_databox_tags::VectorTag> > > >,
       brigand::list<brigand::list<test_databox_tags::VectorTag,
                     test_databox_tags::Tag0>,
       brigand::list<test_databox_tags::VectorTag,
                     Tags::Variables<
                     brigand::list<test_databox_tags::ScalarTag,
                     test_databox_tags::VectorTag> > > > > >'
   \endcode
//
// \see TypeDisplayer
*/
template <class... Ts>
struct make_list {
  using type = tmpl::list<Ts...>;
};
//******************************************************************************

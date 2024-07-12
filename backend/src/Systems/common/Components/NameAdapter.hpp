#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/common/Components/NameComponent.hpp"
//
#include "Utilities/TMPL.hpp"

namespace elastica {

  namespace detail {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Provides a name to a System
     * \ingroup systems
     */
    template <typename /* Traits */, typename /* Block */, typename NameImpl>
    struct NameAdapter : public NameImpl {
     protected:
      //**Type definitions******************************************************
      //! List of computed variables
      using ComputedVariables = tmpl::list<>;
      //! List of initialized variables
      using InitializedVariables = tmpl::list<>;
      //! List of all variables
      using Variables = tmpl::list<>;
      //************************************************************************

      //************************************************************************
      /*!\brief Initialize method for the current component
       *
       * \tparam DownstreamBlock The final block-like object which is
       * derived from the current component
       * \tparam CosseratInitializer An Initializer corresponding to the
       * Cosserat rod hierarchy
       */
      template <typename DownstreamBlock, typename Initializer>
      static void initialize(DownstreamBlock&, Initializer&&) noexcept {}
      //************************************************************************

     public:
      //************************************************************************
      /*!\brief Human-readable name of the current plugin and all derivates
       *
       * \note This is intended to work with pretty_type::name<>
       */
      using NameImpl::name;
      //************************************************************************
    };
    //**************************************************************************

  }  // namespace detail

}  // namespace elastica

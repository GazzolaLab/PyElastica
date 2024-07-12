#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <type_traits>

#include "Systems/Block/Block.hpp"
//
#include "Systems/CosseratRods/Traits/DataOpsTraits.hpp"
#include "Systems/CosseratRods/Traits/PlacementTrait.hpp"
#include "Systems/CosseratRods/TypeTraits.hpp"
#include "Systems/CosseratRods/Types.hpp"
//
#include "Utilities/NonCreatable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

//==============================================================================
//
//  DOXYGEN DOCUMENTATION
//
//==============================================================================

//******************************************************************************
/*!\defgroup cosserat_rod_traits Cosserat Rod Traits
 * \ingroup cosserat_rod
 * \brief Traits and protocols for customizing behavior of Cosserat Rods
 */
//******************************************************************************

namespace elastica {

  namespace cosserat_rod {

    //==========================================================================
    //
    //  CLASS DEFINITIONS
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Traits class collecting common meta-functions for implementing
     * Cosserat Rods
     * \ingroup cosserat_rod_traits
     *
     * \details
     * CosseratRodTraits offers a compile-time customization point for the data
     * -structures, placement and algorithms needed to implement a Cosserat rod
     * hierarchy implemented using @ref blocks in \elastica. CosseratRodTraits
     * is templated on three types controlling orthogonal aspects of a Cosserat
     * Rod---the data structures used, where these are placed in the
     * finite-difference grids (similar to primal and dual grids, for example
     * on nodes/elements/voronoi) and memory allocation. This makes it possible
     * to customize the Traits class if needed.
     *
     * Finally, an instantiation of CosseratRodTraits is used as the first
     * template parameter in elastica::cosserat_rod::CosseratRodPlugin and
     * across all components defined in @ref cosserat_rod_component.
     * It does not need customization, but
     *
     * \tparam TraitsForData Traits controlling the data-structures
     * \tparam TraitsForPlacement Traits controlling the placement of variables
     * \tparam TraitsForAllocation Traits controlling memory allocation
     */
    template <typename TraitsForDataOps, typename TraitsForPlacement,
              typename TraitsForAllocation>
    struct CosseratRodTraits : private NonCreatable {
      //**Type definitions******************************************************
      //! Data type for seamless use in components
      using DataType = TraitsForDataOps;
      //! Real type
      using real_type = typename DataType::real_type;
      //! Type of index
      using index_type = typename DataType::index_type;
      //! Placement type for seamless use in components
      using Place = TraitsForPlacement;
      //! Type of size
      using size_type = typename Place::size_type;
      //! Operations type to act on the data, typically contains kernels
      using Operations = typename TraitsForDataOps::Operations;
      //************************************************************************

      //************************************************************************
      /*!\brief Variable for use in @ref cosserat_rod_component
       *
       * \details
       * CosseratRodVariable is a special blocks::Variable customized to the
       * need of cosserat rods
       *
       * \tparam ParameterTag A unique type parameter to mark the Variable with
       * \tparam RankTag      Type for customizing the data implementation, from
       * DataType
       * \tparam StaggerTag   Type(class) for customizing the placement on grid,
       * from Place
       * \tparam Tags...      Other types to provide customization by
       * inheritance
       *
       * \see blocks::Variable
       */
      template <typename ParameterTag, typename RankTag, typename StaggerTag,
                typename... Tags>
      struct CosseratRodVariable
          : public ::blocks::Variable<ParameterTag, RankTag, StaggerTag,
                                      Tags...> {
        //**Type definitions****************************************************
        //! Type recording the staggering of Cosserat rod variables
        using Stagger = StaggerTag;
        //**********************************************************************
      };
      //************************************************************************

      //************************************************************************
      /*!\brief InitializedVariable for use in @ref cosserat_rod_component
       *
       * \details
       * CosseratRodInitializedVariable is a special blocks::InitializedVariable
       * customized to the need of cosserat rods
       *
       * \tparam ParameterTag A unique type parameter to mark the
       * InitializedVariable with
       * \tparam RankTag      Type for customizing the data implementation, from
       * DataType
       * \tparam StaggerTag   Type(class) for customizing the placement
       * on grid, from Place
       * \tparam Tags... Other types to provide customization by inheritance
       *
       * \see blocks::InitializedVariable
       */
      template <typename ParameterTag, typename RankTag, typename StaggerTag,
                typename... Tags>
      struct CosseratRodInitializedVariable
          : public ::blocks::InitializedVariable<ParameterTag, RankTag,
                                                 StaggerTag, Tags...> {
        //**Type definitions****************************************************
        //! Type recording the staggering of Cosserat rod variables
        using Stagger = StaggerTag;
        //**********************************************************************
      };
      //************************************************************************

      //************************************************************************
      /*!\brief Check if `Var` is placed on the nodes in the Cosserat rod grid
       *
       * \tparam Var Variable to be checked for placement on node
       *
       * \see elastica::cosserat_rod::tt::PlacementTypeTrait
       */
      template <typename Var>
      using IsOnNode =
          typename tt::PlacementTypeTrait<Place>::template IsOnNode<Var>;
      //************************************************************************

      //************************************************************************
      /*!\brief Check if `Var` is placed on the elements in the Cosserat rod
       * grid
       *
       * \tparam Var Variable to be checked for placement on element
       *
       * \see elastica::cosserat_rod::tt::PlacementTypeTrait
       */
      template <typename Var>
      using IsOnElement =
          typename tt::PlacementTypeTrait<Place>::template IsOnElement<Var>;
      //************************************************************************

      //************************************************************************
      /*!\brief Check if `Var` is placed on the voronois in the Cosserat rod
       * grid
       *
       * \tparam Var Variable to be checked for placement on voronoi
       *
       * \see elastica::cosserat_rod::tt::PlacementTypeTrait
       */
      template <typename Var>
      using IsOnVoronoi =
          typename tt::PlacementTypeTrait<Place>::template IsOnVoronoi<Var>;
      //************************************************************************

      //************************************************************************
      /*!\brief Check if `Var` is placed on the rods in the Cosserat rod grid
       *
       * \tparam Var Variable to be checked for placement on rod
       *
       * \see elastica::cosserat_rod::tt::PlacementTypeTrait
       */
      template <typename Var>
      using IsOnRod =
          typename tt::PlacementTypeTrait<Place>::template IsOnRod<Var>;
      //************************************************************************
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Default traits for Cosserat Rods
     * \ingroup cosserat_rod_traits
     *
     * \details
     * DefaultCosseratRodTraits is our implementation of default traits for all
     * canned CosseratRod types. It uses the `blaze` library for matrix and
     * tensor operations, implemented in DataOpsTraits and uses PlacementTrait
     * to control the domain of variables (on node, elements, voronois etc).
     *
     * \see CosseratRodTraits
     */
    struct DefaultCosseratRodTraits
        : public CosseratRodTraits<DataOpsTraits, PlacementTrait, void> {};
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

//      template <typename VarTag>
//      struct TensorSlice;
//      template <typename VarTag>
//      struct TensorSlice {
//        using This = TensorSlice<VarTag>;
//        static_assert(std::is_same<typename VarTag::SliceType,
//        This>::value,
//                      "Invariant violation in slices");
//        using Root = TensorTag;
//        using type = typename Root::slice_type;
//        constexpr type& get() noexcept { return d_; }
//        type d_;
//      };

//      template <typename VarTag>
//      struct MatrixSlice;

// can be in traits class

//      // Useful aliases
//      using NodeMatrix = MatrixTag<NodeTag>;
//      using VoronoiMatrix = MatrixTag<VoronoiTag>;
//      using ElementMatrix = MatrixTag<ElementTag>;

//      template <typename ParameterTag, typename StaggeredTag>
//      struct MatrixSlice {
//        using TagType = StaggeredTag;
//        using type = typename MatrixTag<StaggeredTag>::slice_type;
//        constexpr type& get() noexcept { return d_; }
//        type d_;
//      };

//      // Useful aliases
//      using NodeMatrixSlice = MatrixSliceTag<NodeTag>;
//      using VoronoiMatrixSlice = MatrixSliceTag<VoronoiTag>;
//      using ElementMatrixSlice = MatrixSliceTag<ElementTag>;
//
// can be in traits class

//      template <typename VarTag>
//      struct VectorSlice;

//      // Useful aliases
//      using NodeVector = VectorTag<NodeTag>;
//      using VoronoiVector = VectorTag<VoronoiTag>;
//      using ElementVector = VectorTag<ElementTag>;
//
//      template <typename ParameterTag, typename StaggeredTag>
//      struct VectorSlice {
//        using TagType = StaggeredTag;
//        using type = typename VectorTag<StaggeredTag>::slice_type;
//        constexpr type& get() noexcept { return d_; }
//        type d_;
//      };

/*
template <blaze::AlignmentFlag AF,  // Alignment flag
          size_t I,          // Index of the first subvector element
          size_t N,          // Size of the subvector
          typename VT,       // Type of the vector
          bool TF,           // Transpose flag
          typename... RSAs>  // Optional subvector arguments
inline decltype(auto) custom_subvector(ElementVector::data_type& vector,
                                       RSAs... args) {
  using ReturnType = Subvector_<VT, AF, I, N>;
  return ReturnType(*vector, args...);
}
  */
//*************************************************************************************************

/*
template <typename... RSAs>  // Optional subvector arguments
static inline decltype(auto) custom_subvector(
    blaze::Vector<typename ElementVector::data_type,
                  blaze::columnVector> const& vector,
    std::size_t index, std::size_t size, RSAs... args) {
  using ReturnType =
      const blaze::Subvector_<const VoronoiVector::data_type,
                              blaze::unaligned>;
  return ReturnType(*vector, index, size, args...);
}
 */
//*************************************************************************************************

// TODO: Having different tag types seem painful for conversion of
//  operators from node->element->voronoi and vice-versa. Postpone for
//  now.

//      template <typename InputTagType = ElementVector,
//                typename OutputTagType = VoronoiVector>
//      static inline auto calculate_voronoi_length(
//          typename InputTagType::data_type const& phys) ->
//          typename OutputTagType::data_type {
//        // convolve with weights 0.5, 0.5
//        auto n_voronoi(OutputTagType::get_dofs(phys.size()));
//        return real_t(0.5) * (blaze::subvector(phys, 1UL, n_voronoi) +
//                              blaze::subvector(phys, 0UL, n_voronoi));
//      }

//      // Useful aliases
//      using NodeVectorSlice = VectorSliceTag<NodeTag>;
//      using VoronoiVectorSlice = VectorSliceTag<VoronoiTag>;
//      using ElementVectorSlice = VectorSliceTag<ElementTag>;
//

// can be in traits class
//      struct ScalarTag {
//        // To prevent cross operations
//        using data_type = std::vector<real_t>;
//        // A concrete slice type
//        using slice_type = real_t;
//      };

//      struct ScalarSliceTag {
//        using data_type = real_t&;
//
//        // Why can't I just use get
//        static inline constexpr auto get_dofs(std::size_t /*i*/)
//            -> std::size_t {
//          return 1UL;
//        }
//      };

//

// CRMT :-> CosseratRodMemberTag
// Can be free functions, but better to restrict scope here
//      template <typename CRMT, template <typename...> class Tuple,
//                typename... Slices>
//      static constexpr decltype(auto) get_slice(Tuple<Slices...>& t)
//      noexcept {
//        return std::get<GetSliceType<CRMT>>(t).get();
//      }

// Couple of lessons learnt here:
/*
 * Initially the design was to use an alias template inside the tags
 * to look at the initializer type, simply like so:
 *
 *
  struct MemberTag : Traits::VoronoiVector {

  template <typename Func>
  using InitializerType = tags::ReferenceCurvatureProfile<Func>;
  }

  And then in the call site, where we are passed in a tuple of
  these (instantiated initializer types with some lambda function
  Func), one can compare whether its an instantiation of the MemberTags
  Initializer template type
  like so

  // Alias for checking whether a tpye is an instantiaion of
 InitializerType using Checker = ::tt::detail::is_a<CRMT::template
 InitializerType>;

  // The templated check struct checks for an instantiation
  using Initializer = tmpl::front<
      // Find
      tmpl::find<InitializerTuple,
                 // Does it match the template?
                 tmpl::bind<Checker::template check, tmpl::_1>>
      //                       typename Checker::template check<tmpl::_1>>
      // End find
      >;

  This approach is volatile and causes an issue if CRMT::template
  InitializerType is an alias type as documented here :

  https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59498

  While there is no clear fix (and I don't know what the standard
  says about this), one  fix is to not use an alias iniside the class
  but rather a struct itself. This leads to a counter intutive design
  as we initialize the struct inside the Tag, which the initializer
 factory (for creating cosserat rods) needs to know about, which is less
 than ideal.


  The alternate route which I have gone below is to simply mark the
 MemberTag with an initializer Tag like so: struct PositionTag :
 Traits::NodeMatrix{ using InitializerTag = tags::PositionProfileTag;
  }

  Then when make the NamedGenerator with the same Tag
  ####### stands for some demangled name which we dont care about
  template <typename Func>
  aka PositionProgile##### = NamedType<Func, PositionProfileTag>;

  Now its possible to compare MemberTag::InitialzierTag to
  Initializer::ParameterType for a match
 */

//      template <typename CRMT, template <typename...> class Tuple,
//                typename... Initializers>
//      static constexpr decltype(auto) get_initializer(
//          Tuple<Initializers...>&& t) noexcept {
//        // Here initializers can be a ref too (in the case its not an
//        // immediate lambda, hence we first remove reference here
//
//        // TODO Is there no alternative than doing an O(n) search here?
//        using InitializerTuple = Tuple<Initializers...>;
//        using find_result =
//            tmpl::find<InitializerTuple,
//                       std::is_same<tmpl::pin<GetInitializerTagType<CRMT>>,
//                                    tt::GetParameterType<
//                                        std::remove_reference<tmpl::_1>>>>;
//
//        // TODO : Better display of failed component
//        static_assert(tmpl::size<find_result>::value, "Initializer not
//        found!");
//
//        using CRMTInitializer = tmpl::front<find_result>;
//
//        // The trailing get is to return the generator from the
//        enclosing
//        // strong type
//        return
//        std::get<CRMTInitializer>(std::forward<InitializerTuple>(t))
//            .get();
//      }

/*
using NodeVector = blaze::DynamicVector<real_t, blaze::columnVector,
                                        blaze::AlignedAllocator<real_t>,
                                        cosserat_rod::NodeTag>;
using NodeMatrix = blaze::DynamicMatrix<real_t, blaze::rowMajor,
                                        blaze::AlignedAllocator<real_t>,
                                        cosserat_rod::NodeTag>;

//
using ElementVector = blaze::DynamicVector<real_t, blaze::columnVector,
                                           blaze::AlignedAllocator<real_t>,
                                           cosserat_rod::ElementTag>;
using ElementMatrix = blaze::DynamicMatrix<real_t, blaze::rowMajor,
                                           blaze::AlignedAllocator<real_t>,
                                           cosserat_rod::ElementTag>;

//
using VoronoiMatrix = blaze::DynamicMatrix<real_t, blaze::rowMajor,
                                           blaze::AlignedAllocator<real_t>,
                                           cosserat_rod::VoronoiTag>;
                                           */

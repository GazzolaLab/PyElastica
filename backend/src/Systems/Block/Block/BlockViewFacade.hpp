#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Systems/Block/Block/Types.hpp"
//
#include "Systems/Block/Block/AsVariables.hpp"
#include "Systems/Block/Block/TypeTraits.hpp"
#include "Systems/Block/BlockVariables/TypeTraits.hpp"

namespace blocks {

  namespace detail {

    //**************************************************************************
    //! Region over a block
    struct Region {
      //! Start region
      std::size_t start;
      //! Region size
      std::size_t size;
    };
    //**************************************************************************

    //**Equality operator*******************************************************
    /*!\brief Equality comparison between two Region objects.
     *
     * \param lhs The left-hand side region.
     * \param rhs The right-hand side region.
     * \return \a true if the regions are over the same space, else \a false
     */
    inline constexpr auto operator==(Region const& lhs,
                                     Region const& rhs) noexcept -> bool {
      // Brings in the tuple header
      // return std::tie(lhs.start, lhs.size) == std::tie(rhs.start, rhs.size);
      return (lhs.start == rhs.start) && (lhs.size == rhs.size);
    }
    /*! \endcond */
    //****************************************************************************

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Implementation of a slice/view on an \elastica Block
     * \ingroup blocks
     *
     * \details
     * The BlockViewFacade template class provides a slice (or) view into a
     * Block of `data` and `operations`. It also models the `ComputationalBlock`
     * concept and hence can be used across places where a \elastica Block is
     * used as a template parameter.
     *
     * Similar to Block, BlockViewFacade can also be customized (specialized)
     * for any Lagrangian entity, but experience suggests that extensive
     * customization is not needed. For a concrete example, please see
     * specializations of BlockViewFacade in @ref cosserat_rod
     *
     * \usage
     * The intended usage of a BlockViewFacade is when a view is required into
     the data
     * held by a block---this is frequently the case when a user either
     * 1. adds an entity into the Simulator, or
     * 2. requires access (for e.g. reading/writing to disk) to only a portion
     of
     * the block.
     * The pattern that is most commonly seen in the use case (1) is for a
     * BlockViewFacade templated on some `Plugin` type to be only used with a
     Block
     * of the same `Plugin` type, when adding new entities to the Block.
     * For use case (2) we suggest the user to the blocks::slice() function,
     which
     * has an intuitive, explicit slice syntax, or even the subscript
     operator[].
     * We note that we might explicitly disable the subscript operator [] for
     * slicing in the future, as the semantics are potentially unclear.
     *
     * Finally, with no additional coding effort, the BlockViewFacade has
     exactly the
     * same operations as the mother Block (aka Block of the same `Plugin`
     type),
     * but now it operates only on that slice of the data. This means that a
     * BlockViewFacade can be used as a small Block in itself which greatly
     simplifies
     * interfacing different components of \elastica---the user need not care or
     * even know about whether the data that she has is a Block or a
     BlockViewFacade!
     * For example,
       \code
           // ... make simulator etc ...
           auto my_rod = simulator.emplace_back<CosseratRod>( * ...args... *);
           // the args go and form a Block
           // which in turn returns a BlockViewFacade
           // which the user gets as my_rod

           // use my_rod like a regular cosserat rod
           simulator->constrain(my_rod)->using<SomeConstraint>( *...args... *);
       \endcode
     * This abstraction helped us constrain \c my_rod, embedded in a Block of
     * data, using \c SomeConstraint just like any non-Blocked item of the
     * \elastica library.
     *
     * \tparam Plugin The computational plugin modeling a Lagrangian entity
     *
     * \see Block, blocks::slice
     */
    template <class Plugin>
    class BlockViewFacade {
     protected:
      //**Type definitions******************************************************
      //! Type of the slice
      using This = BlockViewFacade<Plugin>;
      //************************************************************************

     public:
      //**Type definitions******************************************************
      //! Type of the plugin
      using PluginType = Plugin;
      //! List of variables
      using Variables = typename PluginType::Variables;
      //! Typelist of all variable slices
      using VariableSlicesList = typename as_slices<Variables>::slices;
      //! Values of all variable slices
      using BlockVariableSlices = as_block_variables<VariableSlicesList>;
      //! Typelist of all variable slices
      using VariableConstSlicesList =
          typename as_slices<Variables>::const_slices;
      //! Values of all variable slices
      using BlockVariableConstViews =
          as_block_variables<VariableConstSlicesList>;
      //! Conformant mapping between tags and variables
      using VariableMap = VariableMapping<VariableSlicesList>;
      //! Type of the parent block
      using ParentBlock =
          typename PluginFrom<BlockView<Plugin>>::template to<Block>::type;
      //************************************************************************

     protected:
      //**Constructors**********************************************************
      /*!\name Constructors */
      //@{

      //************************************************************************
      /*!\brief The default constructor.
       *
       * \param parent_block Parent block for the slice
       * \param region Index of the slice
       */
      BlockViewFacade(ParentBlock& parent_block, std::size_t start_index,
                      std::size_t region_size) noexcept
          : region_{start_index, region_size}, parent_block_(parent_block) {}
      //************************************************************************

      //************************************************************************
      /*!\brief The copy constructor.
       *
       * \param other slice to copy
       */
      BlockViewFacade(BlockViewFacade const& other) noexcept
          : region_(other.region_), parent_block_(other.parent_block_){};
      //************************************************************************

      //************************************************************************
      /*!\brief The move constructor.
       *
       * \param other slice to move from
       */
      BlockViewFacade(BlockViewFacade&& other) noexcept
          : region_(std::move(other.region_)),
            parent_block_(other.parent_block_){};
      //************************************************************************

      //@}
      //************************************************************************

     public:
      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~BlockViewFacade() = default;
      //@}
      //************************************************************************

      //**Utility methods*******************************************************
      /*!\name Utility methods*/
      //@{

      //************************************************************************
      /*!\brief Gets the parent block
       */
      inline constexpr auto parent() & noexcept -> ParentBlock& {
        return parent_block_;
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Gets the parent block
       */
      inline constexpr auto parent() const& noexcept -> ParentBlock const& {
        return parent_block_;
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Gets the parent block
       *
       * \note
       * This method is useful when xvalue block-slices are generated
       * on-the-fly via BlockRefs. If this overload is not present, then the
       * const& is picked up, and it is not possible to assign values to the
       * parent block anymore.
       *
       * \note
       * Returning a reference is valid as the parent block outlives the block
       * slice.
       */
      inline constexpr auto parent() && noexcept -> ParentBlock& {
        return parent_block_;
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Gets the parent block
       *
       * \note
       * This method is provided for symmetry with the overload above.
       *
       * \note
       * Returning a reference is valid as the parent block outlives the block
       * slice.
       */
      inline constexpr auto parent() const&& noexcept -> ParentBlock const& {
        return parent_block_;
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Returns the slice region
       */
      inline constexpr auto region() const noexcept -> Region const& {
        return region_;
      }
      //************************************************************************

      //**Data access***********************************************************
      /*!\name Data access */
      //@{
     protected:
      template <typename... Vars>
      inline constexpr decltype(auto) generate_data(tmpl::list<Vars...> /*meta*/
                                                    ) & {
        using RT = BlockVariableSlices;
        return RT{
            parent().template slice<Vars>(region().start, region().size)...};
      }

      template <typename... Vars>
      inline constexpr decltype(auto) generate_data(tmpl::list<Vars...> /*meta*/
      ) const& {
        using RT = BlockVariableConstViews const;
        return RT{
            parent().template slice<Vars>(region().start, region().size)...};
      }

     public:
      //************************************************************************
      /*!\brief Access to the underlying data
      //
      // \return Underlying data
      */
      inline constexpr decltype(auto) data() & noexcept {
        return generate_data(Variables{});
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Access to the underlying data
      //
      // \return Constant lvalue reference to the underlying data
      */
      inline constexpr decltype(auto) data() const& noexcept {
        return generate_data(Variables{});
      }
      //************************************************************************

      //@}
      //************************************************************************

     private:
      //**Member variables******************************************************
      /*!\name Member variables */
      //@{
      //! View region
      detail::Region region_;
      //! Reference to the parent block
      ParentBlock& parent_block_;
      //@}
      //************************************************************************
    };
    /*! \endcond */
    //**************************************************************************

    //**************************************************************************
    /*! \cond ELASTICA_INTERNAL */
    /*!\brief Implementation of a slice/view on a const \elastica Block
     * \ingroup blocks
     *
     * \details
     * The ConstBlockViewFacade template class provides a slice (or) view into
     * a constant Block of \a data and \a operations. It also models the
     * `ComputationalBlock` concept and hence can be used across places where a
     * \elastica Block is used as a template parameter.
     *
     * Notably, it differs from BlockViewFacade in one key aspect : the
     * underlying data/view is always constant and cannot be modified (it is
     * only read only). Hence ConstBlock is useful for propagating
     * const-correctness throughout the code. It can be used in places where one
     * needs to pass (const)-data to the user which she can then copy and use it
     * for her own purposes.
     *
     * \tparam Plugin The computational plugin modeling a Lagrangian entity
     *
     * \see BlockViewFacade
     */
    template <class Plugin>  // The computational plugin type
    class ConstBlockViewFacade {
     protected:
      //**Type definitions******************************************************
      //! Type of the slice
      using This = ConstBlockViewFacade<Plugin>;
      //************************************************************************

     public:
      //**Type definitions******************************************************
      //! Type of the plugin
      using PluginType = Plugin;
      //! List of variables
      using Variables = typename PluginType::Variables;
      //! Typelist of all variable slices
      using VariableSlicesList = typename as_slices<Variables>::const_slices;
      //! Values of all variable slices
      using BlockVariableSlices = as_block_variables<VariableSlicesList>;
      //! Conformant mapping between tags and variables
      using VariableMap = VariableMapping<VariableSlicesList>;
      //! Type of the parent block
      using ParentBlock =
          typename PluginFrom<ConstBlockView<Plugin>>::template to<Block>::type;
      //************************************************************************

     protected:
      //**Constructors**********************************************************
      /*!\name Constructors */
      //@{

      //************************************************************************
      /*!\brief The default constructor.
       *
       * \param parent_block Parent block for the slice
       * \param region Index of the slice
       */
      ConstBlockViewFacade(ParentBlock const& parent_block,
                           std::size_t start_index,
                           std::size_t region_size) noexcept
          : region_{start_index, region_size}, parent_block_(parent_block) {}
      //************************************************************************

      //************************************************************************
      /*!\brief The copy constructor.
       *
       * \param other slice to copy
       */
      ConstBlockViewFacade(ConstBlockViewFacade const& other) noexcept
          : region_(other.region_), parent_block_(other.parent_block_){};
      //************************************************************************

      //************************************************************************
      /*!\brief The move constructor.
       *
       * \param other slice to move from
       */
      ConstBlockViewFacade(ConstBlockViewFacade&& other) noexcept
          : region_(std::move(other.region_)),
            parent_block_(other.parent_block_){};
      //************************************************************************

      //@}
      //************************************************************************

     public:
      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~ConstBlockViewFacade() = default;
      //@}
      //************************************************************************

      //**Utility methods*******************************************************
      /*!\name Utility methods*/
      //@{

      //************************************************************************
      /*!\brief Gets the parent block
       */
      inline constexpr auto parent() const noexcept -> ParentBlock const& {
        return parent_block_;
      }
      //************************************************************************

      //************************************************************************
      /*!\brief Returns the slice region
       */
      inline constexpr auto region() const noexcept -> Region const& {
        return region_;
      }
      //************************************************************************

      //@}
      //************************************************************************

      //**Data access***********************************************************
      /*!\name Data access */
      //@{
     protected:
      template <typename... Vars>
      inline constexpr decltype(auto) generate_data(tmpl::list<Vars...> /*meta*/
      ) const& {
        using RT = BlockVariableSlices;
        return RT{
            parent().template slice<Vars>(region().start, region().size)...};
      }

     public:
      //************************************************************************
      /*!\brief Access to the underlying data
      //
      // \return Constant lvalue reference to the underlying data
      */
      inline constexpr decltype(auto) data() const& noexcept {
        return generate_data(Variables{});
      }
      //************************************************************************

      //@}
      //************************************************************************

     private:
      //**Member variables******************************************************
      /*!\name Member variables */
      //@{
      //! View region
      detail::Region region_;
      //! Reference to the parent block
      ParentBlock const& parent_block_;
      //@}
      //************************************************************************
    };
    /*! \endcond */
    //**************************************************************************

    //==========================================================================
    //
    //  GLOBAL OPERATORS
    //
    //==========================================================================

    //**Equality operator*******************************************************
    /*!\brief Equality comparison between two BlockViewFacade objects.
     *
     * \param lhs The left-hand side slice.
     * \param rhs The right-hand side slice.
     * \return \a true if the slices are same, else \a false
     */
    template <class Plugin>
    inline constexpr auto operator==(
        BlockViewFacade<Plugin> const& lhs,
        BlockViewFacade<Plugin> const& rhs) noexcept -> bool {
      return (&lhs.parent() == &rhs.parent()) and lhs.region() == rhs.region();
    }
    //**************************************************************************

    //**Equality operator*******************************************************
    /*!\brief Equality comparison between two ConstBlockViewFacade objects.
     *
     * \param lhs The left-hand side const slice.
     * \param rhs The right-hand side const slice.
     * \return \a true if the const slices are same, else \a false
     */
    template <class Plugin>
    inline constexpr auto operator==(
        ConstBlockViewFacade<Plugin> const& lhs,
        ConstBlockViewFacade<Plugin> const& rhs) noexcept -> bool {
      return (&lhs.parent() == &rhs.parent()) and lhs.region() == rhs.region();
    }
    //**************************************************************************

  }  // namespace detail

}  // namespace blocks

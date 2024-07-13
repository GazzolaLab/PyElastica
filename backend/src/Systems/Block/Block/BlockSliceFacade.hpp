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
     * The BlockSliceFacade template class provides a slice (or) view into a
     * Block of `data` and `operations`. It also models the `ComputationalBlock`
     * concept and hence can be used across places where a \elastica Block is
     * used as a template parameter.
     *
     * Similar to Block, BlockSliceFacade can also be customized (specialized)
     * for any Lagrangian entity, but experience suggests that extensive
     * customization is not needed. For a concrete example, please see
     * specializations of BlockSliceFacade in @ref cosserat_rod
     *
     * \usage
     * The intended usage of a BlockSliceFacade is when a view is required into
     the data
     * held by a block---this is frequently the case when a user either
     * 1. adds an entity into the Simulator, or
     * 2. requires access (for e.g. reading/writing to disk) to only a portion
     of
     * the block.
     * The pattern that is most commonly seen in the use case (1) is for a
     * BlockSliceFacade templated on some `Plugin` type to be only used with a
     Block
     * of the same `Plugin` type, when adding new entities to the Block.
     * For use case (2) we suggest the user to the blocks::slice() function,
     which
     * has an intuitive, explicit slice syntax, or even the subscript
     operator[].
     * We note that we might explicitly disable the subscript operator [] for
     * slicing in the future, as the semantics are potentially unclear.
     *
     * Finally, with no additional coding effort, the BlockSliceFacade has
     exactly the
     * same operations as the mother Block (aka Block of the same `Plugin`
     type),
     * but now it operates only on that slice of the data. This means that a
     * BlockSliceFacade can be used as a small Block in itself which greatly
     simplifies
     * interfacing different components of \elastica---the user need not care or
     * even know about whether the data that she has is a Block or a
     BlockSliceFacade!
     * For example,
       \code
           // ... make simulator etc ...
           auto my_rod = simulator.emplace_back<CosseratRod>( * ...args... *);
           // the args go and form a Block
           // which in turn returns a BlockSliceFacade
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
    class BlockSliceFacade {
     protected:
      //**Type definitions******************************************************
      //! Type of the slice
      using This = BlockSliceFacade<Plugin>;
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
      using BlockVariableConstSlices =
          as_block_variables<VariableConstSlicesList>;
      //! Conformant mapping between tags and variables
      using VariableMap = VariableMapping<VariableSlicesList>;
      //! Type of the parent block
      using ParentBlock =
          typename PluginFrom<BlockSlice<Plugin>>::template to<Block>::type;
      //************************************************************************

     protected:
      //**Constructors**********************************************************
      /*!\name Constructors */
      //@{

      //************************************************************************
      /*!\brief The default constructor.
       *
       * \param parent_block Parent block for the slice
       * \param index Index of the slice
       */
      BlockSliceFacade(ParentBlock& parent_block, std::size_t index) noexcept
          : index_(index), parent_block_(parent_block) {}
      //************************************************************************

      //************************************************************************
      /*!\brief The copy constructor.
       *
       * \param other slice to copy
       */
      BlockSliceFacade(BlockSliceFacade const& other) noexcept
          : index_(other.index_), parent_block_(other.parent_block_){};
      //************************************************************************

      //************************************************************************
      /*!\brief The move constructor.
       *
       * \param other slice to move from
       */
      BlockSliceFacade(BlockSliceFacade&& other) noexcept
          : index_(other.index_), parent_block_(other.parent_block_){};
      //************************************************************************

      //@}
      //************************************************************************

     public:
      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~BlockSliceFacade() = default;
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
      /*!\brief Returns the slice index
       */
      inline constexpr auto index() const noexcept -> std::size_t {
        return index_;
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
                                                    ) & {
        using RT = BlockVariableSlices;
        return RT{parent().template slice<Vars>(index())...};
      }

      template <typename... Vars>
      inline constexpr decltype(auto) generate_data(tmpl::list<Vars...> /*meta*/
      ) const& {
        using RT = BlockVariableConstSlices const;
        return RT{parent().template slice<Vars>(index())...};
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
      //! Slice index
      std::size_t index_;
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
     * The ConstBlockSliceFacade template class provides a slice (or) view into
     * a constant Block of \a data and \a operations. It also models the
     * `ComputationalBlock` concept and hence can be used across places where a
     * \elastica Block is used as a template parameter.
     *
     * Notably, it differs from BlockSliceFacade in one key aspect : the
     * underlying data/view is always constant and cannot be modified (it is
     * only read only). Hence ConstBlock is useful for propagating
     * const-correctness throughout the code. It can be used in places where one
     * needs to pass (const)-data to the user which she can then copy and use it
     * for her own purposes.
     *
     * \tparam Plugin The computational plugin modeling a Lagrangian entity
     *
     * \see BlockSliceFacade
     */
    template <class Plugin>  // The computational plugin type
    class ConstBlockSliceFacade {
     protected:
      //**Type definitions******************************************************
      //! Type of the slice
      using This = ConstBlockSliceFacade<Plugin>;
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
      using ParentBlock = typename PluginFrom<
          ConstBlockSlice<Plugin>>::template to<Block>::type;
      //************************************************************************

      // ideally we make the constructors protected, but this interferes with
      // nothrow specifications
      // https://stackoverflow.com/questions/40441994/usage-of-noexcept-in-derived-classes
     protected:
      //**Constructors**********************************************************
      /*!\name Constructors */
      //@{

      //************************************************************************
      /*!\brief The default constructor.
       *
       * \param parent_block Parent block for the slice
       * \param index Index of the slice
       */
      ConstBlockSliceFacade(ParentBlock const& parent_block,
                            std::size_t index) noexcept
          : index_(index), parent_block_(parent_block) {}
      //************************************************************************

      //************************************************************************
      /*!\brief The copy constructor.
       *
       * \param other slice to copy
       */
      ConstBlockSliceFacade(ConstBlockSliceFacade const& other) noexcept
          : index_(other.index_), parent_block_(other.parent_block_){};
      //************************************************************************

      //************************************************************************
      /*!\brief The move constructor.
       *
       * \param other slice to move from
       */
      ConstBlockSliceFacade(ConstBlockSliceFacade&& other) noexcept
          : index_(other.index_), parent_block_(other.parent_block_){};
      //************************************************************************

      //@}
      //************************************************************************

     public:
      //**Destructor************************************************************
      /*!\name Destructor */
      //@{
      ~ConstBlockSliceFacade() = default;
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
      /*!\brief Returns the slice index
       */
      inline constexpr auto index() const noexcept -> std::size_t {
        return index_;
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
        return RT{parent().template slice<Vars>(index())...};
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
      //! Slice index
      std::size_t index_;
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
    /*!\brief Equality comparison between two BlockSliceFacade objects.
     *
     * \param lhs The left-hand side slice.
     * \param rhs The right-hand side slice.
     * \return \a true if the slices are same, else \a false
     */
    template <class Plugin>
    inline constexpr auto operator==(
        BlockSliceFacade<Plugin> const& lhs,
        BlockSliceFacade<Plugin> const& rhs) noexcept -> bool {
      return (&lhs.parent() == &rhs.parent()) and lhs.index() == rhs.index();
    }
    //**************************************************************************

    //**Equality operator*******************************************************
    /*!\brief Equality comparison between two ConstBlockSliceFacade objects.
     *
     * \param lhs The left-hand side const slice.
     * \param rhs The right-hand side const slice.
     * \return \a true if the const slices are same, else \a false
     */
    template <class Plugin>
    inline constexpr auto operator==(
        ConstBlockSliceFacade<Plugin> const& lhs,
        ConstBlockSliceFacade<Plugin> const& rhs) noexcept -> bool {
      return (&lhs.parent() == &rhs.parent()) and lhs.index() == rhs.index();
    }
    //**************************************************************************

  }  // namespace detail

}  // namespace blocks

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Systems/Block/Block.hpp"
#include "Systems/Types.hpp"
#include "Systems/common/Cardinality.hpp"
#include "Systems/common/Tags.hpp"
#include "Utilities/AsConst.hpp"
#include "Utilities/TMPL.hpp"
//
#include <utility>  // forward

namespace elastica {

  //****************************************************************************
  /*!\brief Generic component initializer for system with unit-cardinality.
   * \ingroup systems
   *
   * \tparam Variables Typelist of `BlockVariables` to initialize
   * \param block_like Data-structure conforming to the block concept.
   * \param initializer Initializer object conforming to block initialization.
   */
  template <typename Variables, typename BlockLike, typename Initializer>
  void initialize_component(BlockLike& block_like, Initializer&& initializer,
                            UnitCardinality /*meta*/
  ) {
    struct {
      Initializer value;
    } cap{std::forward<Initializer>(initializer)};
    // Note : need a mutable lambda as we can now "move" from cap
    tmpl::for_each<Variables>([&block_like, &cap](auto v) mutable -> void {
      using Variable = tmpl::type_from<decltype(v)>;
      using Tag = blocks::initializer_t<Variable>;

      // && since get() may be lazy and so can return either
      // lvalue of rvalue reference.
      auto&& variable(blocks::get<Tag>(block_like));
      // We may not need a ref here as it will be moved, but keep it for
      // symmetry.
      auto&& variable_initializer(blocks::get<Tag>(std::move(cap.value)));
      Variable::slice(variable, UnitCardinality::index()) = variable_initializer();
    });
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Generic component initializer for system with multiple-cardinality.
   * \ingroup systems
   *
   * \tparam Variables Typelist of `BlockVariables` to initialize
   * \param block_like Data-structure conforming to the block concept.
   * \param initializer Initializer object conforming to block initialization.
   */
  template <typename Variables, typename BlockLike, typename Initializer>
  void initialize_component(BlockLike& block_like, Initializer&& initializer,
                            MultipleCardinality /*meta*/
  ) {
    // Note : need a mutable lambda as we can now "move" from cap
    // Make a copy of the initializer as there is no guarantee to the number of
    // times it will be called.
    auto const n_elem =
        blocks::get<::elastica::tags::NElement>(cpp17::as_const(initializer))();
    struct {
      Initializer value;
    } cap{std::forward<Initializer>(initializer)};
    tmpl::for_each<Variables>(
        [n_elem, &block_like, &cap](auto v) mutable -> void {
          using Variable = tmpl::type_from<decltype(v)>;
          using Tag = blocks::initializer_t<Variable>;

          // && since get() may be lazy and so can return either
          // lvalue of rvalue reference.
          auto&& variable(blocks::get<Tag>(block_like));
          // We may not need a ref here as it will be moved, but keep
          // it for symmetry.
          auto&& variable_initializer(blocks::get<Tag>(std::move(cap.value)));

          const auto dofs = Variable::get_dofs(n_elem);
          for (std::size_t i = 0; i < dofs; ++i) {
            // TODO : This doesn't work on slices, consider having the
            // slice member in the variable_initializer itself
            Variable::slice(variable, i) = variable_initializer(i);
          }
        });
  }
  //****************************************************************************

  namespace detail {

    template <typename Variables, typename BlockLike, typename Initializer>
    inline void initialize_component(BlockLike& block_like,
                                     Initializer&& initializer) {
      initialize_component<Variables>(block_like,
                                      std::forward<Initializer>(initializer),
                                      typename BlockLike::cardinality{});
    }

  }  // namespace detail

}  // namespace elastica

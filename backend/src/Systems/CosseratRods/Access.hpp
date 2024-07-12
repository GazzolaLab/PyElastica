#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <cstddef>  // size_t

#include "Systems/common/Access.hpp"
/// Forward declarations
#include "Systems/CosseratRods/Types.hpp"
///
#include "Systems/CosseratRods/CosseratRodPlugin.hpp"

namespace elastica {

  //============================================================================
  //
  //  ACCESS API FUNCTIONS
  //
  //============================================================================

  //**Position functions********************************************************
  /*!\name Position functions */
  //@{
  /*!\brief Specialization of elastica::position() for Cosserat rods
   * \ingroup cosserat_rod
   */
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) position(
      ::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ComputationalBlock, Components...>& cosserat_rod) noexcept {
    // get_position() wraps around a free function, so the API is still not bad
    return cosserat_rod.self().get_position();
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) position(
      ::elastica::cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                                  Components...>& cosserat_rod,
      std::size_t index) noexcept {
    // tentative, can change
    // ideally should be position(blocks::slice(cosserat_rod, index));
    // but slice(cosserat_rod_slice) is not defined (and we do not need it per
    // se).

    // Other option is
    // blocks::slice(position(cosserat_rod), index);
    // which is very close to what we have below. The problem here is that
    // position(cosserat_rod) is not a block anymore, so the semantics are not
    // well formed.
    return cosserat_rod.self().get_position(index);
  }

  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) position(
      ::elastica::cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                                  Components...> const&
          cosserat_rod) noexcept {
    return cosserat_rod.self().get_position();
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) position(
      ::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ComputationalBlock, Components...> const& cosserat_rod,
      std::size_t index) noexcept {
    return cosserat_rod.self().get_position(index);
  }
  //@}
  //****************************************************************************

  //**Director functions********************************************************
  /*!\name Director functions */
  //@{
  /*!\brief Specialization of elastica::director() for Cosserat rods
   * \ingroup cosserat_rod
   */
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) director(
      ::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ComputationalBlock, Components...>& cosserat_rod) noexcept {
    return cosserat_rod.self().get_director();
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) director(
      ::elastica::cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                                  Components...>& cosserat_rod,
      std::size_t index) noexcept {
    return cosserat_rod.self().get_director(index);
  }

  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) director(
      ::elastica::cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                                  Components...> const&
          cosserat_rod) noexcept {
    return cosserat_rod.self().get_director();
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) director(
      ::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ComputationalBlock, Components...> const& cosserat_rod,
      std::size_t index) noexcept {
    return cosserat_rod.self().get_director(index);
  }
  //@}
  //****************************************************************************

  //**Velocity functions********************************************************
  /*!\name Velocity functions */
  //@{
  /*!\brief Specialization of elastica::velocity() for Cosserat rods
   * \ingroup cosserat_rod
   */
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) velocity(
      ::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ComputationalBlock, Components...>& cosserat_rod) noexcept {
    return cosserat_rod.self().get_velocity();
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) velocity(
      ::elastica::cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                                  Components...>& cosserat_rod,
      std::size_t index) noexcept {
    return cosserat_rod.self().get_velocity(index);
  }

  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) velocity(
      ::elastica::cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                                  Components...> const&
          cosserat_rod) noexcept {
    return cosserat_rod.self().get_velocity();
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) velocity(
      ::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ComputationalBlock, Components...> const& cosserat_rod,
      std::size_t index) noexcept {
    return cosserat_rod.self().get_velocity(index);
  }
  //@}
  //****************************************************************************

  //**Angular Velocity functions************************************************
  /*!\name Angular Velocity functions */
  //@{
  /*!\brief Specialization of elastica::angular_velocity() for Cosserat rods
   * \ingroup cosserat_rod
   */
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) angular_velocity(
      ::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ComputationalBlock, Components...>& cosserat_rod) noexcept {
    return cosserat_rod.self().get_angular_velocity();
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) angular_velocity(
      ::elastica::cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                                  Components...>& cosserat_rod,
      std::size_t index) noexcept {
    return cosserat_rod.self().get_angular_velocity(index);
  }

  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) angular_velocity(
      ::elastica::cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                                  Components...> const&
          cosserat_rod) noexcept {
    return cosserat_rod.self().get_angular_velocity();
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) angular_velocity(
      ::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ComputationalBlock, Components...> const& cosserat_rod,
      std::size_t index) noexcept {
    return cosserat_rod.self().get_angular_velocity(index);
  }
  //@}
  //****************************************************************************

  //**External loads functions**************************************************
  /*!\name External loads functions */
  //@{
  /*!\brief Specialization of elastica::external_loads() for Cosserat rods
   * \ingroup cosserat_rod
   */
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) external_loads(
      ::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ComputationalBlock, Components...>& cosserat_rod) noexcept {
    return cosserat_rod.self().get_external_loads();
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) external_loads(
      ::elastica::cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                                  Components...>& cosserat_rod,
      std::size_t index) noexcept {
    return cosserat_rod.self().get_external_loads(index);
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) external_loads(
      ::elastica::cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                                  Components...> const&
          cosserat_rod) noexcept {
    return cosserat_rod.self().get_external_loads();
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) external_loads(
      ::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ComputationalBlock, Components...> const& cosserat_rod,
      std::size_t index) noexcept {
    return cosserat_rod.self().get_external_loads(index);
  }
  //@}
  //****************************************************************************

  //**External torques functions************************************************
  /*!\name External torques functions */
  //@{
  /*!\brief Specialization of elastica::external_torques() for Cosserat rods
   * \ingroup cosserat_rod
   */
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) external_torques(
      ::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ComputationalBlock, Components...>& cosserat_rod) noexcept {
    return cosserat_rod.self().get_external_torques();
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) external_torques(
      ::elastica::cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                                  Components...>& cosserat_rod,
      std::size_t index) noexcept {
    return cosserat_rod.self().get_external_torques(index);
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) external_torques(
      ::elastica::cosserat_rod::CosseratRodPlugin<CRT, ComputationalBlock,
                                                  Components...> const&
          cosserat_rod) noexcept {
    return cosserat_rod.self().get_external_torques();
  }
  template <typename CRT, template <typename> class ComputationalBlock,
            template <typename /*CRT*/, typename /* ComputationalBlock */>
            class... Components>
  inline constexpr decltype(auto) external_torques(
      ::elastica::cosserat_rod::CosseratRodPlugin<
          CRT, ComputationalBlock, Components...> const& cosserat_rod,
      std::size_t index) noexcept {
    return cosserat_rod.self().get_external_torques(index);
  }
  //@}
  //****************************************************************************

}  // namespace elastica

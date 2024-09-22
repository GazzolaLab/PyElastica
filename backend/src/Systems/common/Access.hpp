#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <cstddef>  // size_t

namespace elastica {

  //============================================================================
  //
  //  ACCESS API FUNCTIONS
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Any physical system adhering to elastica::protocols::PhysicalSystem
  // \ingroup systems
  */
  struct AnyPhysicalSystem {};
  //****************************************************************************

  //**Position functions********************************************************
  /*!\name Position functions */
  //@{
  /*!\brief Retrieves position from the systems
   * \ingroup systems
   */
  inline constexpr decltype(auto) position(AnyPhysicalSystem& t) noexcept;
  inline constexpr decltype(auto) position(AnyPhysicalSystem const& t) noexcept;
  inline constexpr decltype(auto) position(AnyPhysicalSystem& t,
                                           std::size_t index) noexcept;
  inline constexpr decltype(auto) position(AnyPhysicalSystem const& t,
                                           std::size_t index) noexcept;
  //@}
  //****************************************************************************

  //**Director functions********************************************************
  /*!\name Director functions */
  //@{
  /*!\brief Retrieves directors from the systems
   * \ingroup systems
   */
  inline constexpr decltype(auto) director(AnyPhysicalSystem& t) noexcept;
  inline constexpr decltype(auto) director(AnyPhysicalSystem const& t) noexcept;
  inline constexpr decltype(auto) director(AnyPhysicalSystem& t,
                                           std::size_t index) noexcept;
  inline constexpr decltype(auto) director(AnyPhysicalSystem const& t,
                                           std::size_t index) noexcept;
  //@}
  //****************************************************************************

  //**Velocity functions********************************************************
  /*!\name Velocity functions */
  //@{
  /*!\brief Retrieves velocities from the systems
   * \ingroup systems
   */
  inline constexpr decltype(auto) velocity(AnyPhysicalSystem& t) noexcept;
  inline constexpr decltype(auto) velocity(AnyPhysicalSystem const& t) noexcept;
  inline constexpr decltype(auto) velocity(AnyPhysicalSystem& t,
                                           std::size_t index) noexcept;
  inline constexpr decltype(auto) velocity(AnyPhysicalSystem const& t,
                                           std::size_t index) noexcept;
  //@}
  //****************************************************************************

  //**Angular Velocity functions************************************************
  /*!\name Angular Velocity functions */
  //@{
  /*!\brief Retrieves angular velocities from the systems
   * \ingroup systems
   */
  inline constexpr decltype(auto) angular_velocity(
      AnyPhysicalSystem& t) noexcept;
  inline constexpr decltype(auto) angular_velocity(
      AnyPhysicalSystem const& t) noexcept;
  inline constexpr decltype(auto) angular_velocity(AnyPhysicalSystem& t,
                                                   std::size_t index) noexcept;
  inline constexpr decltype(auto) angular_velocity(AnyPhysicalSystem const& t,
                                                   std::size_t index) noexcept;
  //@}
  //****************************************************************************

  //**External loads functions**************************************************
  /*!\name External loads functions */
  //@{
  /*!\brief Retrieves external loads of a given system
   * \ingroup systems
   */
  inline constexpr decltype(auto) external_loads(AnyPhysicalSystem& t) noexcept;
  inline constexpr decltype(auto) external_loads(
      AnyPhysicalSystem const& t) noexcept;
  inline constexpr decltype(auto) external_loads(AnyPhysicalSystem& t,
                                                 std::size_t index) noexcept;
  inline constexpr decltype(auto) external_loads(AnyPhysicalSystem const& t,
                                                 std::size_t index) noexcept;
  //@}
  //****************************************************************************

  //**External torques functions************************************************
  /*!\name External torques functions */
  //@{
  /*!\brief Retrieves external torques of a given system
   * \ingroup systems
   */
  inline constexpr decltype(auto) external_torques(
      AnyPhysicalSystem& t) noexcept;
  inline constexpr decltype(auto) external_torques(
      AnyPhysicalSystem const& t) noexcept;
  inline constexpr decltype(auto) external_torques(AnyPhysicalSystem& t,
                                                   std::size_t index) noexcept;
  inline constexpr decltype(auto) external_torques(AnyPhysicalSystem const& t,
                                                   std::size_t index) noexcept;
  //@}
  //****************************************************************************

}  // namespace elastica

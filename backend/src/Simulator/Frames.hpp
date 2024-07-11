#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include "Simulator/Frames/EulerianFrame.hpp"
#include "Simulator/Frames/LagrangianFrame.hpp"
#include "Utilities/Math/Vec3.hpp"

namespace elastica {

  // ?
  using EulerianFrame = detail::EulerianFrame;
  using detail::cast_along;
  using LagrangianFrame = detail::LagrangianFrame;
  // ?

  //****************************************************************************
  /*!\brief Gets a unit vector along an Eulerian direction
   * \ingroup simulator
   *
   * \details
   * Gets the unit vector along a direction `Dir` of an Eulerian Frame
   *
   * \param dir Direction along which to get a unit vector
   */
  inline constexpr auto get_unit_vector_along(
      EulerianFrame::DirectionType dir) noexcept -> Vec3 {
    return {
        cast_along(dir) == 0 ? 1.0 : 0.0,
        cast_along(dir) == 1 ? 1.0 : 0.0,
        cast_along(dir) == 2 ? 1.0 : 0.0,
    };
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Gets a unit vector along an Eulerian direction
   * \ingroup simulator
   *
   * \details
   * Gets the unit vector along a direction `Dir` of an Eulerian Frame
   *
   * \tparam Dir Direction along which to get a unit vector
   */
  template <EulerianFrame::DirectionType Dir>
  inline constexpr auto get_unit_vector_along() noexcept -> Vec3 {
    return get_unit_vector_along(Dir);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Gets a unit vector along a Lagrangian direction
   * \ingroup simulator
   *
   * \details
   * Gets the unit vector along a direction `Dir` of an Lagrangian Frame
   *
   * \example
   * \snippet Test_LagrangianFrame.cpp vector_along_eg
   *
   * \param dir Direction along which to get a unit vector
   */
  inline constexpr auto get_unit_vector_along(
      LagrangianFrame::DirectionType dir) noexcept -> Vec3 {
    return {
        cast_along(dir) == 0 ? 1.0 : 0.0,
        cast_along(dir) == 1 ? 1.0 : 0.0,
        cast_along(dir) == 2 ? 1.0 : 0.0,
    };
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Gets a unit vector along a Lagrangian direction
   * \ingroup simulator
   *
   * \details
   * Gets the unit vector along a direction `Dir` of an Lagrangian Frame
   *
   * \tparam Dir Direction along which to get a unit vector
   */
  template <LagrangianFrame::DirectionType Dir>
  inline constexpr auto get_unit_vector_along() noexcept -> Vec3 {
    return get_unit_vector_along(Dir);
  }
  //****************************************************************************

  struct Frames {
    static_assert(LagrangianFrame::Dimension == EulerianFrame::Dimension,
                  "Wrong dimension configuration!");
    static constexpr auto Dimension = LagrangianFrame::Dimension;
  };

}  // namespace elastica

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <cstdint>

#include "Utilities/NonCreatable.hpp"

namespace elastica {

  namespace detail {

    //==========================================================================
    //
    //  CLASS DEFINITION
    //
    //==========================================================================

    //**************************************************************************
    /*!\brief Trait defining unique IDs for directions in Eulerian frame
     * \ingroup simulator
     *
     * EulerianFrame is a trait defining the type and number of directions in
     * 3D. It serves as a strong type for all directions within the \elastica
     * library.
     */
    struct EulerianFrame : public NonCreatable {
     public:
      //**Type definitions******************************************************
      //! The type of direction ID
      enum class DirectionType : std::uint8_t { x = 0, y, z, Count };
      //************************************************************************

      //**Static members********************************************************
      //! unique ID for X direction
      static constexpr DirectionType X = DirectionType::x;
      //! unique ID for Y direction
      static constexpr DirectionType Y = DirectionType::y;
      //! unique ID for Z direction
      static constexpr DirectionType Z = DirectionType::z;
      //! Dimension of simulation
      static constexpr auto Dimension =
          static_cast<std::uint8_t>(DirectionType::Count);
      //************************************************************************
    };
    //**************************************************************************

    //**************************************************************************
    /*!\brief Casts a Eulerian frame direction into an index value
     * \ingroup simulator
     *
     * \details
     * Converts a EulerianFrame::DirectionType into an index value in
     * associative arrays.
     *
     * \example
     * \snippet Test_EulerianFrame.cpp cast_along_eg
     *
     * \param dir A Direction to cast as index to
     */
    inline constexpr auto cast_along(EulerianFrame::DirectionType dir)
        -> std::size_t {
      return static_cast<std::size_t>(dir);
    }
    //**************************************************************************

    //**************************************************************************
    /*!\brief Casts a Eulerian frame direction into an index value
     * \ingroup simulator
     *
     * \details
     * Converts a EulerianFrame::DirectionType into an index value in
     * associative arrays.
     *
     * \example
     * \snippet Test_EulerianFrame.cpp cast_along_template_eg
     *
     * \tparam Dir A Direction to cast as index to
     */
    template <typename EulerianFrame::DirectionType Dir>
    inline constexpr auto cast_along() -> std::size_t {
      return cast_along(Dir);
    }
    //**************************************************************************

  }  // namespace detail

}  // namespace elastica

#pragma once

//******************************************************************************
// Includes
//******************************************************************************
#include <type_traits>
#include <utility>

namespace elastica {

  namespace cosserat_rod {

    // This is useful in unit-testing
    namespace protocols {

      /*!
       * \brief Has rod shape functions.
       *
       * Requires the class has these member functions:
       * - `get_dimension`: Returns the principal dimension of the rod (radius,
       * half-edge etc.)
       * - `get_area`: Returns the cross sectional area of a rod element
       * - `get_volume`: Returns the volume of a rod element
       * - `get_second_moment_of_area` : Templated function that returns the
       * moment of area along the template direction for a rod element
       * - `get_shape_factor` : Returns the shape factor for the CS shape
       *
       * and the following static members
       * - `D1`: Tag for the first orthogonal direction (normal)
       * - `D2`: Tag for the second orthogonal direction (binormal)
       * - `D3`: Tag for the third orthogonal direction (tangent)
       */
      struct Elastic1D {
        /*
         *  1. get_dimension
         *  2. get_area
         *  3. get_volume
         *  4. template function get_second_moment_of_inertia
         *  5. Direction Types
         *  5. shape_factor
         *  */
        template <typename ConformingType>
        struct test {
          // Try calling the `ConformingType::get_dimension` member function
          //          using dim_type =
          //              decltype(std::declval<ConformingType&>().get_dimension());
          using dimension_return_type =
              decltype(std::declval<ConformingType&>().get_dimension(
                  std::declval<std::size_t>()));

          // Try calling the `ConformingType::get_area` member function
          //          using area_type =
          //              decltype(std::declval<ConformingType&>().get_area());
          using area_return_type =
              decltype(std::declval<ConformingType&>().get_area(
                  std::declval<std::size_t>()));

          // Try calling the `ConformingType::get_volume` member function
          using volume_return_type =
              decltype(std::declval<ConformingType&>().get_volume(
                  std::declval<std::size_t>()));

          // Check for existence of static constexpr members
          static constexpr auto D1 = ConformingType::D1;
          static constexpr auto D2 = ConformingType::D2;
          static constexpr auto D3 = ConformingType::D3;

          // Try calling the `ConformingType::get_second_moment_of_inertia`
          // member function with templated D1
          using second_moment_of_area_return_type = std::common_type_t<
              decltype(std::declval<ConformingType&>()
                           .template get_second_moment_of_area<D1>(
                               std::declval<std::size_t>())),
              decltype(std::declval<ConformingType&>()
                           .template get_second_moment_of_area<D2>(
                               std::declval<std::size_t>())),
              decltype(std::declval<ConformingType&>()
                           .template get_second_moment_of_area<D3>(
                               std::declval<std::size_t>()))>;

          // Try calling the `ConformingType::get_shape_factor` member function
          using shape_factor_return_type =
              decltype(std::declval<ConformingType&>().get_shape_factor());

          using float_type = typename ConformingType::float_type;

          // Check the return type of the member functions
          static_assert(std::is_same<dimension_return_type, float_type>::value,
                        "Not a conforming domain, doesn't properly implement "
                        "`get_dimension`");
          static_assert(std::is_same<area_return_type, float_type>::value,
                        "Not a conforming domain, doesn't properly implement "
                        "`get_area`");
          static_assert(std::is_same<volume_return_type, float_type>::value,
                        "Not a conforming domain, doesn't properly implement "
                        "`get_volume`");
          static_assert(std::is_same<second_moment_of_area_return_type,
                                     float_type>::value,
                        "Not a conforming domain, doesn't properly implement "
                        "`get_second_moment_of_area`");
          static_assert(
              std::is_same<shape_factor_return_type, float_type>::value,
              "Not a conforming domain, doesn't properly implement "
              "`get_shape_factor`");
        };
      };

    }  // namespace protocols

  }  // namespace cosserat_rod

}  // namespace elastica

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include "Utilities/DefineTypes.h"

namespace elastica {

  //============================================================================
  //
  //  CLASS DEFINITION
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Collection of numerical threshold values.
   * \ingroup utils
   *
   * The Thresholds class defines numerical floating point thresholds for the
   * \elastica physics engine. The following thresholds can be used:
   *
   * - `collision_threshold`: Used for contact classification in distinguishing
   * between separating, resting and colliding contacts.
   * - `contact_threshold`: Used to tell whether two rigid bodies are in contact
   * or not. If the distance between two bodies is smaller than this threshold,
   * they are considered to be in contact with each other.
   * - `restitution_threshold`: In case the relative velocity between two
   * colliding rigid bodies is smaller than this threshold, a coefficient of
   * restitution of 0 is used to avoid an infinite number of collisions during a
   * single time step.
   * - `friction_threshold`: Represents the boundary between
   * static and dynamic friction. In case the relative tangential velocity of
   * two contacting rigid bodies is smaller than this threshold, static
   * friction is applied, else dynamic friction is used.
   * - `surface_threshold`: Used for surface checks. Only points with a distance
   * to the surface smaller than this threshold are considered surface point.
   * - `parallel_threshold`: Used for parallelism checks. If the scalar product
   * of two vectors is smaller than this threshold the vectors are considered to
   * be parallel.
   *
   * The Thresholds class is used in the following manner
   *
   * \example
   * \code
   * const double p = Thresholds<double>::parallel_threshold();
   * if( dist < Thresholds<double>::surface_threshold ) {...}
   * \endcode
   * \b Note: The Thresholds class is not defined for integral data types.
   */
  template <typename Float>
  struct Thresholds {};
  //****************************************************************************

  //============================================================================
  //
  //  FLOAT SPECIALIZATION
  //
  //============================================================================

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Thresholds<float> specialization.
   * \ingroup core
   */
  template <>
  struct Thresholds<float> {
   public:
    //! _threshold for the contact classification.
    /*! This threshold separates between separating, resting and colliding
     * contacts. */
    static inline constexpr float collision_threshold() { return 1E-8F; }

    //! _threshold for the distance between two rigid bodies.
    /*! Rigid bodies with a distance smaller than this threshold are in contact.
     */
    static inline constexpr float contact_threshold() { return 5E-7F; }

    //! _threshold for the restriction of the coefficient of restitution.
    /*! In case the relative velocity between two colliding rigid bodies is
       smaller than this threshold, a coefficient of restitution of 0 is used to
       avoid an infinite number of collisions during a single time step. */
    static inline constexpr float restitution_threshold() { return 1E-8F; }

    //! _threshold for the separation between static and dynamic friction.
    /*! This threshold represents the boundary between static and dynamic
     * friction. */
    static inline constexpr float friction_threshold() { return 1E-8F; }

    //! _threshold for surface points/checks.
    /*! Only points with a distance to the surface smaller than this threshold
       are considered surface point. */
    static inline constexpr float surface_threshold() { return 5E-7F; }

    //! _threshold for parallelism checks.
    /*! Scalar products smaller than this threshold value indicate parallel
     * vectors. */
    static inline constexpr float parallel_threshold() { return 1E-8F; }
  };
  /*! \endcond */
  //****************************************************************************

  //============================================================================
  //
  //  DOUBLE SPECIALIZATION
  //
  //============================================================================

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Thresholds<double> specialization.
   * \ingroup core
   */
  template <>
  struct Thresholds<double> {
   public:
    //! _threshold for the contact classification.
    /*! This threshold separates between separating, resting and colliding
     * contacts. */
    static inline constexpr double collision_threshold() { return 1E-8; }

    //! _threshold for the distance between two rigid bodies.
    /*! Rigid bodies with a distance smaller than this threshold are in contact.
     */
    static inline constexpr double contact_threshold() { return 1E-8; }

    //! _threshold for the restriction of the coefficient of restitution.
    /*! In case the relative velocity between two colliding rigid bodies is
       smaller than this threshold, a coefficient of restitution of 0 is used to
       avoid an infinite number of collisions during a single time step. */
    static inline constexpr double restitution_threshold() { return 1E-8; }

    //! _threshold for the separation between static and dynamic friction.
    /*! This threshold represents the boundary between static and dynamic
     * friction. */
    static inline constexpr double friction_threshold() { return 1E-8; }

    //! _threshold for surface points/checks.
    /*! Only points with a distance to the surface smaller than this threshold
       are considered surface point. */
    static inline constexpr double surface_threshold() { return 5E-7; }

    //! _threshold for parallelism checks.
    /*! Scalar products smaller than this threshold value indicate parallel
     * vectors. */
    static inline constexpr double parallel_threshold() { return 1E-8; }
  };
  /*! \endcond */
  //****************************************************************************

  //============================================================================
  //
  //  LONG DOUBLE SPECIALIZATION
  //
  //============================================================================

  //****************************************************************************
  /*! \cond ELASTICA_INTERNAL */
  /*!\brief Thresholds<long double> specialization.
   * \ingroup core
   */
  template <>
  struct Thresholds<long double> {
   public:
    //! _threshold for the contact classification.
    /*! This threshold separates between separating, resting and colliding
     * contacts. */
    static inline constexpr long double collision_threshold() { return 1E-10L; }

    //! _threshold for the distance between two rigid bodies.
    /*! Rigid bodies with a distance smaller than this threshold are in contact.
     */
    static inline constexpr long double contact_threshold() { return 5E-7L; }

    //! _threshold for the restriction of the coefficient of restitution.
    /*! In case the relative velocity between two colliding rigid bodies is
       smaller than this threshold, a coefficient of restitution of 0 is used to
       avoid an infinite number of collisions during a single time step. */
    static inline constexpr long double restitution_threshold() {
      return 1E-8L;
    }

    //! _threshold for the separation between static and dynamic friction.
    /*! This threshold represents the boundary between static and dynamic
     * friction. */
    static inline constexpr long double friction_threshold() { return 1E-8L; }

    //! _threshold for surface points/checks.
    /*! Only points with a distance to the surface smaller than this threshold
       are considered surface point. */
    static inline constexpr long double surface_threshold() { return 5E-7L; }

    //! _threshold for parallelism checks.
    /*! Scalar products smaller than this threshold value indicate parallel
     * vectors. */
    static inline constexpr long double parallel_threshold() { return 1E-8L; }
  };
  /*! \endcond */
  //****************************************************************************

  //============================================================================
  //
  //  GLOBAL THRESHOLD VALUES
  //
  //============================================================================

  //****************************************************************************
  /*! \brief _threshold for the contact classification.
   *  \ingroup utils
   *
   *  \details
   *  Used for contact classification in distinguishing
   *  between separating, resting and colliding contacts.
   */
  constexpr real_t collision_threshold =
      Thresholds<real_t>::collision_threshold();
  //****************************************************************************

  //****************************************************************************
  /*!\brief _threshold for the distance between two rigid bodies.
   * \ingroup utils
   *
   * \details
   *
   * Used to tell whether two rigid bodies are in contact
   * or not. If the distance between two bodies is smaller than this threshold,
   * they are considered to be in contact with each other.
   *
   */
  constexpr real_t contact_threshold = Thresholds<real_t>::contact_threshold();
  //****************************************************************************

  //****************************************************************************
  /*!\brief _threshold for restriction of the coefficient of restitution.
   * \ingroup utils
   *
   * \details
   * In case the relative velocity between two
   * colliding rigid bodies is smaller than this threshold, a coefficient of
   * restitution of 0 is used to avoid an infinite number of collisions during a
   * single time step.
   */
  constexpr real_t restitution_threshold =
      Thresholds<real_t>::restitution_threshold();
  //****************************************************************************

  //****************************************************************************
  /*!\brief _threshold for the separation between static and dynamic friction.
   * \ingroup utils
   *
   * \details
   * Represents the boundary between static and dynamic friction. In case the
   * relative tangential velocity of two contacting rigid bodies is smaller than
   * this threshold, static friction is applied, else dynamic friction is used.
   */
  constexpr real_t friction_threshold =
      Thresholds<real_t>::friction_threshold();
  //****************************************************************************

  //****************************************************************************
  /*!\brief _threshold for surface points/checks.
   * \ingroup utils
   *
   * \details
   * Used for surface checks. Only points with a distance
   * to the surface smaller than this threshold are considered surface point.
   */
  constexpr real_t surface_threshold = Thresholds<real_t>::surface_threshold();
  //****************************************************************************

  //****************************************************************************
  /*!\brief _threshold for parallelism checks.
   * \ingroup utils
   *
   * \details
   * Used for parallelism checks. If the scalar product of two vectors is
   * smaller than this threshold the vectors are considered to be parallel.
   */
  constexpr real_t parallel_threshold =
      Thresholds<real_t>::parallel_threshold();
  //****************************************************************************

  //****************************************************************************
  /*!\brief  _threshold for normalization checks.
   * \ingroup utils
   *
   * \details
   * A (length of a vector - 1) smaller than this threshold value indicate
   *  a normalized vector.
   */
  constexpr real_t normal_threshold = Thresholds<real_t>::parallel_threshold();
  //****************************************************************************

}  // namespace elastica

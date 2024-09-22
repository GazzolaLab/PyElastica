//==============================================================================
/*!
//  \file
//  \brief Header file for materials
//
//  Copyright (C) 2020-2020 Tejaswin Parthasarathy - All Rights Reserved
//  Copyright (C) 2020-2020 MattiaLab - All Rights Reserved
//
//  Distributed under the MIT License.
//  See LICENSE.txt for details.
//
//  This file contains implementation from the pe library with the following
//  copyrights
**
**  Project home: https://www.cs10.tf.fau.de/research/software/pe/
**
**  Copyright (C) 2009 Klaus Iglberger
**
**  This file is part of pe.
**
**  pe is free software: you can redistribute it and/or modify it under the
** terms of the GNU General Public License as published by the Free Software
** Foundation, either version 3 of the License, or (at your option) any later
** version.
**
**  pe is distributed in the hope that it will be useful, but WITHOUT ANY
** WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
** A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
**
**  You should have received a copy of the GNU General Public License along with
** pe. If not, see <http://www.gnu.org/licenses/>.
*/
//==============================================================================

#pragma once

//******************************************************************************
// Includes
//******************************************************************************

#include <blaze/math/DynamicMatrix.h>
#include <cstdint>
#include <string>
// #include <sys/types.h>
#include <utility>
#include <vector>

#include "ErrorHandling/Assert.hpp"
//
#include "Simulator/Materials/Types.hpp"
//
#include "Utilities/DefineTypes.h"
#include "Utilities/Math/Invert.hpp"
#include "Utilities/Math/Square.hpp"

namespace elastica {

  //****************************************************************************
  /*!\defgroup materials Materials
   * \ingroup simulator
   * \brief Contains functionality to implement and access data of new
   * materials that constitute soft and rigid-bodies.
   */
  //****************************************************************************

  //============================================================================
  //
  //  CLASS MATERIAL
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Rigid body material.
   * \ingroup materials
   *
   * A material specifies the properties of a rigid body: the density of the
   * body, the coefficient of restitution and the coefficients of static and
   * dynamic friction.
   *
   * The elastica engine provides several predefined materials that can be
   * directly used:
   *
   * - iron
   * - copper
   * - granite
   * - oak
   * - fir
   *
   * \usage
   * In order to create a new custom material use the create_material()
   * function: \example \code
   * // Creating a new material using the following material properties:
   * // - name/identifier: myMaterial
   * // - density: 2.54
   * // - coefficient of restitution: 0.8
   * // - coefficient of static friction: 0.1
   * // - coefficient of dynamic friction: 0.05
   * // - Poisson's ratio: 0.2
   * // - Young's modulus: 80.0
   * // - Contact stiffness: 100
   * // - dampingN: 10
   * // - dampingT: 11
   * MaterialID myMaterial = create_material("myMaterial", 2.54, 0.8, 0.1, 0.05,
   *                                        0.2, 80, 100, 10, 11 );
   * \endcode
   *
   * The following functions can be used to acquire a specific MaterialID or to
   * get a specific property of a material:
   * \code
   * // Searching a material
   * MaterialID myMaterial = Material::find( "myMaterial" );
   *
   * // Getting the density, coefficient of restitution, coefficient of static
   * // and dynamic friction, Poisson's ratio and Young's modulus of the
   * // material
   * real_t density = Material::get_density( myMaterial );
   * real_t cor     = Material::get_restitution( myMaterial );
   * real_t csf     = Material::get_static_friction( myMaterial );
   * real_t cdf     = Material::get_dynamic_friction( myMaterial );
   * real_t poisson = Material::get_poisson_ratio( myMaterial );
   * real_t young   = Material::get_youngs_modulus( myMaterial ):
   * \endcode
   */
  class Material {
   private:
    //**Type definitions********************************************************
    //! Size type of the Material class.
    using SizeType = std::size_t;
    //! Matrix type of the Material class.
    using MatrixType = blaze::DynamicMatrix<real_t>;
    //**************************************************************************

   public:
    //**Constructor*************************************************************
    /*!\name Constructor */
    //@{
    explicit inline Material(std::string name, real_t density, real_t cor,
                             real_t poisson, real_t young, real_t shear_modulus,
                             real_t stiffness, real_t normal_damping,
                             real_t tangential_damping, real_t csf, real_t cdf);
    explicit inline Material(std::string name, real_t density, real_t cor,
                             real_t poisson, real_t young, real_t shear_modulus,
                             real_t stiffness, real_t normal_damping,
                             real_t tangential_damping, real_t forward_csf,
                             real_t backward_csf, real_t lateral_csf,
                             real_t forward_cdf, real_t backward_cdf,
                             real_t lateral_cdf);
    // No explicitly declared copy constructor.
    //@}
    //**************************************************************************

    //**Destructor**************************************************************
    // No explicitly declared destructor.
    //**************************************************************************

    //**Get functions***********************************************************
    /*!\name Get functions */
    //@{
    inline const std::string& get_name() const noexcept;
    inline real_t get_density() const noexcept;
    inline real_t get_restitution() const noexcept;
    inline real_t get_poisson_ratio() const noexcept;
    inline real_t get_youngs_modulus() const noexcept;
    inline real_t get_shear_modulus() const noexcept;
    inline real_t get_stiffness() const noexcept;
    inline real_t get_normal_damping() const noexcept;
    inline real_t get_tangential_damping() const noexcept;
    inline real_t get_static_friction() const noexcept;
    inline real_t get_forward_static_friction() const noexcept;
    inline real_t get_backward_static_friction() const noexcept;
    inline real_t get_lateral_static_friction() const noexcept;
    inline real_t get_dynamic_friction() const noexcept;
    inline real_t get_forward_dynamic_friction() const noexcept;
    inline real_t get_backward_dynamic_friction() const noexcept;
    inline real_t get_lateral_dynamic_friction() const noexcept;

    static MaterialID find(std::string const& name) noexcept;
    static std::vector<MaterialID> find_prefix(std::string const& prefix);
    static inline const std::string& get_name(MaterialID material) noexcept;
    static inline real_t get_density(MaterialID material) noexcept;
    static inline real_t get_restitution(MaterialID material) noexcept;
    static inline real_t get_restitution(MaterialID material1,
                                         MaterialID material2) noexcept;
    static inline real_t get_static_friction(MaterialID material) noexcept;
    static inline real_t get_forward_static_friction(
        MaterialID material) noexcept;
    static inline real_t get_backward_static_friction(
        MaterialID material) noexcept;
    static inline real_t get_lateral_static_friction(
        MaterialID material) noexcept;
    static inline real_t get_static_friction(MaterialID material1,
                                             MaterialID material2) noexcept;
    static inline real_t get_forward_static_friction(
        MaterialID material1, MaterialID material2) noexcept;
    static inline real_t get_backward_static_friction(
        MaterialID material1, MaterialID material2) noexcept;
    static inline real_t get_lateral_static_friction(
        MaterialID material1, MaterialID material2) noexcept;
    static inline real_t get_dynamic_friction(MaterialID material) noexcept;
    static inline real_t get_forward_dynamic_friction(
        MaterialID material) noexcept;
    static inline real_t get_backward_dynamic_friction(
        MaterialID material) noexcept;
    static inline real_t get_lateral_dynamic_friction(
        MaterialID material) noexcept;
    static inline real_t get_dynamic_friction(MaterialID material1,
                                              MaterialID material2) noexcept;
    static inline real_t get_forward_dynamic_friction(
        MaterialID material1, MaterialID material2) noexcept;
    static inline real_t get_backward_dynamic_friction(
        MaterialID material1, MaterialID material2) noexcept;
    static inline real_t get_lateral_dynamic_friction(
        MaterialID material1, MaterialID material2) noexcept;

    static inline real_t get_poisson_ratio(MaterialID material) noexcept;
    static inline real_t get_youngs_modulus(MaterialID material) noexcept;
    static inline real_t get_youngs_modulus(MaterialID material1,
                                            MaterialID material2) noexcept;
    static inline real_t get_shear_modulus(MaterialID material) noexcept;
    static inline real_t get_stiffness(MaterialID material) noexcept;
    static inline real_t get_stiffness(MaterialID material1,
                                       MaterialID material2) noexcept;
    static inline real_t get_normal_damping(MaterialID material) noexcept;
    static inline real_t get_normal_damping(MaterialID material1,
                                            MaterialID material2) noexcept;
    static inline real_t get_tangential_damping(MaterialID material) noexcept;
    static inline real_t get_tangential_damping(MaterialID material1,
                                                MaterialID material2) noexcept;
    //@}
    //**************************************************************************

    //**Set functions***********************************************************
    /*!\name Set functions */
    //@{
    static inline void set_restitution(MaterialID material1,
                                       MaterialID material2,
                                       real_t cor) noexcept;
    static inline void set_static_friction(MaterialID material1,
                                           MaterialID material2,
                                           real_t csf) noexcept;
    static inline void set_forward_static_friction(MaterialID material1,
                                                   MaterialID material2,
                                                   real_t forward_csf) noexcept;
    static inline void set_backward_static_friction(
        MaterialID material1, MaterialID material2,
        real_t backward_csf) noexcept;
    static inline void set_lateral_static_friction(MaterialID material1,
                                                   MaterialID material2,
                                                   real_t lateral_csf) noexcept;
    static inline void set_dynamic_friction(MaterialID material1,
                                            MaterialID material2,
                                            real_t cdf) noexcept;
    static inline void set_forward_dynamic_friction(
        MaterialID material1, MaterialID material2,
        real_t forward_cdf) noexcept;
    static inline void set_backward_dynamic_friction(
        MaterialID material1, MaterialID material2,
        real_t backward_cdf) noexcept;
    static inline void set_lateral_dynamic_friction(
        MaterialID material1, MaterialID material2,
        real_t lateral_cdf) noexcept;
    //@}
    //**************************************************************************

    //**Calculate functions*****************************************************
    /*!\name Calculate functions */
    //@{
    static inline real_t calculate_shear_modulus(real_t young,
                                                 real_t poisson) noexcept;
    //@}
    //**************************************************************************

   private:
    //**Setup functions*********************************************************
    /*!\name Setup functions */
    //@{
    static bool activate_materials() noexcept;
    //@}
    //**************************************************************************

    //**Member variables********************************************************
    /*!\name Member variables */
    //@{
    //! The name of the material.
    std::string name_;
    //! The density of the material.
    real_t density_;
    //! The coefficient of restitution (COR) of a self-similar collision
    //! \f$ [0..1] \f$.
    /* The COR represents the energy dissipated during a collision between
     * self-similar bodies, that is bodies with similar materials. A value of
     * 0 corresponds to completely inelastic collision where all energy is
     * dissipated, a value of 1 corresponds to a completely elastic collision
     * where no energy is lost. The COR is assumed to be rate-independent. The
     * COR is often determined experimentally by measuring the pre- and
     * post-impact relative velocities:
     * \f[ C_R = \frac{V_{2,after}-V_{1,after}}{V_{2,before}-V_{1,before}} \f]
     * During a collision, the COR values of the two colliding
     * rigid bodies can be used by the collision response mechanism to
     * determine the restitution factor of the contact point.
     */
    real_t restitution_;
    //! The coefficient of static friction (CSF) \f [0..\infty) \f$
    /* The CSF is a dimensionless, non-negative quantity representing the
     * amount of static friction between two touching rigid bodies. Static
     * friction occurs in case the relative tangential velocity between the
     * two bodies is 0. Then the force magnitudes of the normal and friction
     * force are related by an inequality:
     * \f[ |\vec{f_t}| \leq \mu_s |\vec{f_n}| \f]
     * The direction of the friction must oppose acceleration if sliding is
     * imminent and is unresticted otherwise.
     */
    real_t static_;
    //! The coefficient of static friction (CSF) in the backward direction
    //! \f [0..\infty) \f$
    /* The CSF is a dimensionless, non-negative quantity representing the
     * amount of static friction between two touching rigid bodies. Static
     * friction occurs in case the relative tangential velocity between the
     * two bodies is 0. Then the force magnitudes of the normal and friction
     * force are related by an inequality:
     * \f[ |\vec{f_t}| \leq \mu_s |\vec{f_n}| \f]
     * The direction of the friction must oppose acceleration if sliding is
     * imminent and is unresticted otherwise.
     */
    real_t backward_static_;
    //! The coefficient of static friction (CSF) in the lateral direction
    //! \f [0..\infty) \f$
    /* The CSF is a dimensionless, non-negative quantity representing the
     * amount of static friction between two touching rigid bodies. Static
     * friction occurs in case the relative tangential velocity between the
     * two bodies is 0. Then the force magnitudes of the normal and friction
     * force are related by an inequality:
     * \f[ |\vec{f_t}| \leq \mu_s |\vec{f_n}| \f]
     * The direction of the friction must oppose acceleration if sliding is
     * imminent and is unresticted otherwise.
     */
    real_t lateral_static_;
    //! The coefficient of dynamic friction (CDF) \f [0..\infty) \f.$
    /* The CDF is a dimensionless, non-negative quantity representing the
     * amount of dynamic friction between two touching rigid bodies. Dynamic
     * friction occurs in case the relative tangential velocity between the
     * two bodies is greater than 0. Then the force magnitudes of the normal
     * and friction force are related by an inequality:
     * \f[ |\vec{f_t}| = -\mu_d |\vec{f_n}| \frac{\vec{v_t}}{|\vec{v_t}|} \f]
     */
    real_t dynamic_;
    //! The coefficient of dynamic friction (CDF) in the backward direction
    //! \f [0..\infty) \f.$
    /* The CDF is a dimensionless, non-negative quantity representing the
     * amount of dynamic friction between two touching rigid bodies. Dynamic
     * friction occurs in case the relative tangential velocity between the
     * two bodies is greater than 0. Then the force magnitudes of the normal
     * and friction force are related by an inequality:
     * \f[ |\vec{f_t}| = -\mu_d |\vec{f_n}| \frac{\vec{v_t}}{|\vec{v_t}|} \f]
     */
    real_t backward_dynamic_;
    //! The coefficient of dynamic friction (CDF) in the lateral direction
    //! \f [0..\infty) \f.$
    /* The CDF is a dimensionless, non-negative quantity representing the
     * amount of dynamic friction between two touching rigid bodies. Dynamic
     * friction occurs in case the relative tangential velocity between the
     * two bodies is greater than 0. Then the force magnitudes of the normal
     * and friction force are related by an inequality:
     * \f[ |\vec{f_t}| = -\mu_d |\vec{f_n}| \frac{\vec{v_t}}{|\vec{v_t}|} \f]
     */
    real_t lateral_dynamic_;
    //! The Poisson's ratio for the material \f$ [-1..0.5] \f$.
    /* When a material is compressed in one direction, it usually tends to
     * expand in the other two directions perpendicular to the direction of
     * compression. This effect is called Poisson effect. In this context, the
     * Poisson's ratio is the ratio of the contraction or transverse strain
     * (perpendicular to the applied load) to the extension or axial strain
     * (in the direction of the applied load). For stable, isotropic, linear
     * elastic materials this ratio cannot be less than -1.0 nor greater than
     * 0.5 due to the requirement that Young's modulus has positive values.
     */
    real_t poisson_;
    //! The Young's modulus for the material \f$ (0..\infty) \f$.
    /* The Young's modulus is a measure for the stiffness of an isotropic
     * elastic material. It is defined as the ratio of the uniaxial stress
     * over the uniaxial strain in the range of stress in which Hooke's law
     * holds. The SI unit for Young's modulus is \f$ Pa \f$ or \f$ N/m^2 \f$.
     */
    real_t young_;
    //! The shear modulus for the material \f$ (0..\infty) \f$.
    /* The shear modulus is a measure for the stiffness of an an-isotropic
     * elastic material. It is defined as the ratio of the shear stress
     * over the shear strain in the range of stress in which Hooke's law
     * holds. The SI unit for Young's modulus is \f$ Pa \f$ or \f$ N/m^2 \f$.
     */
    real_t shear_;
    //! The stiffness of the contact region \f$ (0..\infty) \f$.
    /* Rigid body theory assumes that the deformation during contact is
     * localized to the contact region. This local compliance can be modelled
     * simplified as a spring-damper. The spring constant corresponds to this
     * parameter.
     */
    real_t stiffness_;
    //! The damping at the contact zone in normal direction \f$ [0..\infty)\f$
    /* Rigid body theory assumes that the deformation during contact is
     * localized to the contact region. This local compliance in normal
     * direction can be modelled simplified as a spring-damper. The viscous
     * damping coefficient corresponds to this parameter.
     */
    real_t dampingN_;
    //! The damping at the contact zone in tangent direction \f$ [0..\infty)\f$
    /* Friction counteracts the tangential relative velocity and thus can be
     * modelled as a viscous damper with a limited damping force. The viscous
     * damping coefficient corresponds to this parameter.
     */
    real_t dampingT_;
    //! Vector for the registered materials.
    static Materials materials_;
    //! Lookup Table for the coefficients of restitution.
    static MatrixType corTable_;
    //! Lookup Table for the coefficients of static friction.
    static MatrixType csfTable_;
    //! Lookup Table for the coefficients of backward static friction.
    static MatrixType backward_csfTable_;
    //! Lookup Table for the coefficients of lateral static friction.
    static MatrixType lateral_csfTable_;
    //!< Lookup Table for the coefficients of dynamic friction.
    static MatrixType cdfTable_;
    //!< Lookup Table for the coefficients of backward dynamic friction.
    static MatrixType backward_cdfTable_;
    //!< Lookup Table for the coefficients of lateral dynamic friction.
    static MatrixType lateral_cdfTable_;
    //!< Helper variable for the automatic registration process.
    static bool materialsActivated_;
    //!< Counter for the amount of anonymous materials.
    static unsigned int anonymousMaterials_;
    //@}
    //**************************************************************************

    //**Friend declarations*****************************************************
    /*! \cond ELASTICA_INTERNAL */
    friend MaterialID create_material(std::string const& name, real_t density,
                                      real_t cor, real_t young, real_t shear,
                                      real_t stiffness, real_t dampingN,
                                      real_t dampingT, real_t csf, real_t cdf);
    friend MaterialID create_material(std::string const& name, real_t density,
                                      real_t cor, real_t young, real_t shear,
                                      real_t stiffness, real_t dampingN,
                                      real_t dampingT, real_t forward_csf,
                                      real_t backward_csf, real_t lateral_csf,
                                      real_t forward_cdf, real_t backward_cdf,
                                      real_t lateral_cdf);
    friend MaterialID create_material(real_t density, real_t cor, real_t young,
                                      real_t shear, real_t stiffness,
                                      real_t dampingN, real_t dampingT,
                                      real_t csf, real_t cdf);
    friend MaterialID create_material(real_t density, real_t cor, real_t young,
                                      real_t shear, real_t stiffness,
                                      real_t dampingN, real_t dampingT,
                                      real_t forward_csf, real_t backward_csf,
                                      real_t lateral_csf, real_t forward_cdf,
                                      real_t backward_cdf, real_t lateral_cdf);
    /*! \endcond */
    //**************************************************************************
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief The constructor of the Material class.
   *
   * \param name The name of the material.
   * \param density The density of the material \f$ (0..\infty) \f$.
   * \param cor The coefficient of restitution (COR) of the material
   * \f$ [0..1] \f$.
   * \param poisson The Poisson's ratio of the material \f$ [-1..0.5] \f$.
   * \param young The Young's modulus of the material \f$ (0..\infty) \f$.
   * \param shear The Shear modulus of the material \f$ (0..\infty) \f$.
   * \param stiffness The stiffness in normal direction of the material's
   * contact region.
   * \param normal_damping The damping coefficient in normal direction of the
   * material's contact region.
   * \param tangential_damping The damping coefficient in tangential direction
   * of the material's contact region. \param forward_csf The coefficient of
   * static friction (CSF) of the material \f$ [0..\infty) \f$. \param
   * backward_csf The coefficient of backward static friction (CSF) of the
   * material \f$ [0..\infty) \f$. \param lateral_csf The coefficient of lateral
   * static friction (CSF) of the material \f$ [0..\infty) \f$. \param
   * forward_cdf The coefficient of dynamic friction (CDF) of the material \f$
   * [0..\infty) \f$. \param backward_cdf The coefficient of backward dynamic
   * friction (CSF) of the material \f$ [0..\infty) \f$. \param lateral_cdf The
   * coefficient of lateral dynamic friction (CSF) of the material \f$
   * [0..\infty) \f$.
   */
  inline Material::Material(std::string name, real_t density, real_t cor,
                            real_t poisson, real_t young, real_t shear,
                            real_t stiffness, real_t normal_damping,
                            real_t tangential_damping, real_t forward_csf,
                            real_t backward_csf, real_t lateral_csf,
                            real_t forward_cdf, real_t backward_cdf,
                            real_t lateral_cdf)
      : name_(std::move(name)),  // The name of the material
        density_(density),       // The density of the material
        restitution_(cor),     // The coefficient of restitution of the material
        static_(forward_csf),  // The coefficient of forward static friction
        backward_static_(backward_csf),
        lateral_static_(lateral_csf),
        dynamic_(forward_cdf),  // The coefficient of dynamic friction of the
                                // material
        backward_dynamic_(backward_cdf),
        lateral_dynamic_(lateral_cdf),
        poisson_(poisson),      // The Poisson's ratio for the material
        young_(young),          // The Young's modulus for the material
        shear_(shear),          // The shear modulus for the material
        stiffness_(stiffness),  // The stiffness in normal direction of the
                                // material's contact region.
        dampingN_(
            normal_damping),  // The damping coefficient in normal direction of
                              // the material's contact region.
        dampingT_(
            tangential_damping)  // The damping coefficient in tangential
                                 // direction of the material's contact region.
  {}
  //****************************************************************************

  //****************************************************************************
  /*!\brief The constructor of the Material class.
   *
   * \param name The name of the material.
   * \param density The density of the material \f$ (0..\infty) \f$.
   * \param cor The coefficient of restitution (COR) of the material
   * \f$ [0..1] \f$.
   * \param poisson The Poisson's ratio of the material \f$ [-1..0.5] \f$.
   * \param young The Young's modulus of the material \f$ (0..\infty) \f$.
   * \param stiffness The stiffness in normal direction of the material's
   * contact region.
   * \param normal_damping The damping coefficient in normal direction of the
   * material's contact region.
   * \param tangential_damping The damping coefficient in tangential direction
   * of the material's contact region. \param csf The coefficient of static
   * friction (CSF) of the material \f$ [0..\infty) \f$. \param cdf The
   * coefficient of dynamic friction (CDF) of the material \f$ [0..\infty) \f$.
   */
  inline Material::Material(std::string name, real_t density, real_t cor,
                            real_t poisson, real_t young, real_t shear,
                            real_t stiffness, real_t normal_damping,
                            real_t tangential_damping, real_t csf, real_t cdf)
      : Material(std::move(name), density, cor, poisson, young, shear,
                 stiffness, normal_damping, tangential_damping, csf, csf, csf,
                 cdf, cdf, cdf) {}
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the name of the material.
   *
   * \return The name of the material.
   */
  inline const std::string& Material::get_name() const noexcept {
    return name_;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the density of the material.
   *
   * \return The density of the material.
   */
  inline real_t Material::get_density() const noexcept { return density_; }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of restitution of the material.
   *
   * \return The coefficient of restitution of the material.
   */
  inline real_t Material::get_restitution() const noexcept {
    return restitution_;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of static friction of the material.
   *
   * \return The coefficient of static friction of the material.
   */
  inline real_t Material::get_static_friction() const noexcept {
    return static_;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of forward static friction of the material.
   *
   * \return The coefficient of static friction of the material.
   */
  inline real_t Material::get_forward_static_friction() const noexcept {
    return get_static_friction();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of backward static friction of the material.
   *
   * \return The coefficient of backward static friction of the material.
   */
  inline real_t Material::get_backward_static_friction() const noexcept {
    return backward_static_;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of lateral static friction of the material.
   *
   * \return The coefficient of lateral static friction of the material.
   */
  inline real_t Material::get_lateral_static_friction() const noexcept {
    return lateral_static_;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of dynamic friction of the material.
   *
   * \return The coefficient of dynamic friction of the material.
   */
  inline real_t Material::get_dynamic_friction() const noexcept {
    return dynamic_;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of forward dynamic friction of the material.
   *
   * \return The coefficient of dynamic friction of the material.
   */
  inline real_t Material::get_forward_dynamic_friction() const noexcept {
    return get_dynamic_friction();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of backward dynamic friction of the
   * material.
   *
   * \return The coefficient of backward dynamic friction of the material.
   */
  inline real_t Material::get_backward_dynamic_friction() const noexcept {
    return backward_dynamic_;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of lateral dynamic friction of the material.
   *
   * \return The coefficient of lateral dynamic friction of the material.
   */
  inline real_t Material::get_lateral_dynamic_friction() const noexcept {
    return lateral_dynamic_;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the Poisson's ratio of the material.
   *
   * \return The Poisson's ratio of the material.
   */
  inline real_t Material::get_poisson_ratio() const noexcept {
    return poisson_;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the Young's modulus of the material.
   *
   * \return The Young's modulus of the material.
   */
  inline real_t Material::get_youngs_modulus() const noexcept { return young_; }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the shear modulus of the material.
   *
   * \return The Shear modulus of the material.
   */
  inline real_t Material::get_shear_modulus() const noexcept { return shear_; }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the stiffness in normal direction of the material's contact
   * region.
   *
   * \return The stiffness in normal direction of the material's contact region.
   */
  inline real_t Material::get_stiffness() const noexcept { return stiffness_; }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the damping coefficient in normal direction of the
   * material's contact region.
   *
   * \return The damping coefficient in normal direction of the material's
   * contact region.
   */
  inline real_t Material::get_normal_damping() const noexcept {
    return dampingN_;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the damping coefficient in tangential direction of the
   * material's contact region.
   *
   * \return The damping coefficient in tangential direction of the material's
   * contact region.
   */
  inline real_t Material::get_tangential_damping() const noexcept {
    return dampingT_;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the name of the given material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The name of the given material.
   */
  inline const std::string& Material::get_name(MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_name();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the density of the given material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The density of the given material.
   */
  inline real_t Material::get_density(MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_density();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of restitution of the given material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The coefficient of restitution of the given material.
   */
  inline real_t Material::get_restitution(MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_restitution();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the composite coefficient of restitution for a collision
   * between two rigid bodies. \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The resulting composite coefficient of restitution of the
   * collision.
   */
  inline real_t Material::get_restitution(MaterialID material1,
                                          MaterialID material2) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    return corTable_(material1, material2);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of static friction of the given material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The coefficient of static friction of the given material.
   */
  inline real_t Material::get_static_friction(MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_static_friction();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of static friction for a collision between
   * two rigid bodies.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The resulting coefficient of static friction of the collision.
   */
  inline real_t Material::get_static_friction(MaterialID material1,
                                              MaterialID material2) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    return csfTable_(material1, material2);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of forward static friction of the given
   * material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The coefficient of forward static friction of the given material.
   */
  inline real_t Material::get_forward_static_friction(
      MaterialID material) noexcept {
    return Material::get_static_friction(material);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of forward static friction for a collision
   * between two rigid bodies.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The resulting coefficient of forward static friction of the
   * collision.
   */
  inline real_t Material::get_forward_static_friction(
      MaterialID material1, MaterialID material2) noexcept {
    return Material::get_static_friction(material1, material2);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of backward static friction of the given
   * material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The coefficient of backward static friction of the given material.
   */
  inline real_t Material::get_backward_static_friction(
      MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_backward_static_friction();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of backward static friction for a collision
   * between two rigid bodies.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The resulting coefficient of backward static friction of the
   * collision.
   */
  inline real_t Material::get_backward_static_friction(
      MaterialID material1, MaterialID material2) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    return backward_csfTable_(material1, material2);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of lateral static friction of the given
   * material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The coefficient of lateral static friction of the given material.
   */
  inline real_t Material::get_lateral_static_friction(
      MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_lateral_static_friction();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of lateral static friction for a collision
   * between two rigid bodies.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The resulting coefficient of lateral static friction of the
   * collision.
   */
  inline real_t Material::get_lateral_static_friction(
      MaterialID material1, MaterialID material2) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    return lateral_csfTable_(material1, material2);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of dynamic friction of the given material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The coefficient of dynamic friction of the given material.
   */
  inline real_t Material::get_dynamic_friction(MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_dynamic_friction();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of dynamic friction for a collision between
   * two rigid bodies.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The resulting coefficient of dynamic friction of the collision.
   */
  inline real_t Material::get_dynamic_friction(MaterialID material1,
                                               MaterialID material2) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    return cdfTable_(material1, material2);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of forward dynamic friction of the given
   * material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The coefficient of forward dynamic friction of the given material.
   */
  inline real_t Material::get_forward_dynamic_friction(
      MaterialID material) noexcept {
    return Material::get_dynamic_friction(material);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of forward dynamic friction for a collision
   * between two rigid bodies.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The resulting coefficient of forward dynamic friction of the
   * collision.
   */
  inline real_t Material::get_forward_dynamic_friction(
      MaterialID material1, MaterialID material2) noexcept {
    return Material::get_dynamic_friction(material1, material2);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of backward dynamic friction of the given
   * material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The coefficient of backward dynamic friction of the given material.
   */
  inline real_t Material::get_backward_dynamic_friction(
      MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_backward_dynamic_friction();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of backward dynamic friction for a collision
   * between two rigid bodies.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The resulting coefficient of backward dynamic friction of the
   * collision.
   */
  inline real_t Material::get_backward_dynamic_friction(
      MaterialID material1, MaterialID material2) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    return backward_cdfTable_(material1, material2);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of lateral dynamic friction of the given
   * material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The coefficient of lateral dynamic friction of the given material.
   */
  inline real_t Material::get_lateral_dynamic_friction(
      MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_lateral_dynamic_friction();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the coefficient of lateral dynamic friction for a collision
   * between two rigid bodies.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The resulting coefficient of lateral dynamic friction of the
   * collision.
   */
  inline real_t Material::get_lateral_dynamic_friction(
      MaterialID material1, MaterialID material2) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    return lateral_cdfTable_(material1, material2);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the Poisson's ratio of the given material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The Poisson's ratio of the given material.
   */
  inline real_t Material::get_poisson_ratio(MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_poisson_ratio();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the Young's modulus of the given material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The Young's modulus of the given material.
   */
  inline real_t Material::get_youngs_modulus(MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_youngs_modulus();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the (effective) Young's modulus for a collision between two
   * rigid bodies.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The resulting (effective) Young's modulus of the collision.
   *
   * This function returns the effective Young's modulus for a collision between
   * two rigid bodies. The effective Young's modulus is calculated as
   *
   * \f[\frac{1}{E_{eff}} = \frac{1 - \nu_1^2}{E_1}+\frac{1 - \nu_2^2}{E_2} \f]
   *
   * where \f$ E_1 \f$ and \f$ E_2 \f$ are the Young's modulus for the first and
   * second material, respectively, and \f$ \nu_1 \f$ and \f$ \nu_2 \f$ are the
   * Poisson's ratio for the materials.
   */
  inline real_t Material::get_youngs_modulus(MaterialID material1,
                                             MaterialID material2) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");

    real_t const nu1(get_poisson_ratio(material1));
    real_t const nu2(get_poisson_ratio(material2));
    real_t const y1(get_youngs_modulus(material1));
    real_t const y2(get_youngs_modulus(material2));

    real_t const tmp1(y2 * (real_t(1) - nu1 * nu1));
    real_t const tmp2(y1 * (real_t(1) - nu2 * nu2));

    return ((y1 * y2) / (tmp1 + tmp2));
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the Shear modulus of the given material.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The Shear modulus of the given material.
   */
  inline real_t Material::get_shear_modulus(MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_shear_modulus();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the stiffness in normal direction of the material's contact
   * region.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The stiffness in normal direction of the contact region of the
   * given material.
   */
  inline real_t Material::get_stiffness(MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_stiffness();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the stiffness in normal direction of the contact between two
   * materials.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The stiffness in normal direction of the contact between two
   * materials.
   *
   * Rigid body theory assumes that deformation during contact is localized to
   * the contact region. Therefore the contact region is often modelled
   * simplified as a spring-damper. When two bodies are in contact the
   * spring-dampers are serially connected and thus the contact stiffness can be
   * expressed as the series connection of two springs: \f$ k_*^{-1} = k_1^{-1}
   * + k_2^{-1}\f$.
   */
  inline real_t Material::get_stiffness(MaterialID material1,
                                        MaterialID material2) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");

    return inv(inv(get_stiffness(material1)) + inv(get_stiffness(material2)));
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the damping coefficient in normal direction of the
   * material's contact region.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The damping in normal direction of the contact region of the given
   * material.
   */
  inline real_t Material::get_normal_damping(MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_normal_damping();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the damping in normal direction of the contact between two
   * materials.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The damping in normal direction of the contact between two
   * materials.
   *
   * Rigid body theory assumes that deformation during contact is localized to
   * the contact region. Therefore the contact region is often modelled
   * simplified as a spring-damper. When two bodies are in contact the
   * spring-dampers are serially connected and thus the contact damping can be
   * expressed as the series connection of two viscous dampers: \f$ c_*^{-1} =
   * c_1^{-1} + c_2^{-1}\f$.
   */
  inline real_t Material::get_normal_damping(MaterialID material1,
                                             MaterialID material2) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");

    return inv(inv(get_normal_damping(material1)) +
               inv(get_normal_damping(material2)));
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the damping coefficient in tangential direction of the
   * material's contact region.
   * \ingroup materials
   *
   * \param material The material to be queried.
   * \return The damping in tangential direction of the contact region of the
   * given material.
   */
  inline real_t Material::get_tangential_damping(MaterialID material) noexcept {
    ELASTICA_ASSERT(material < materials_.size(), "Invalid material ID");
    return materials_[material].get_tangential_damping();
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Returns the damping in tangential direction of the contact between
   * two materials.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \return The damping in tangential direction of the contact between two
   * materials.
   *
   * Rigid body theory assumes that deformation during contact is localized to
   * the contact region. Therefore the contact region is often modelled
   * simplified as a spring-damper. When two bodies are in contact the
   * spring-dampers are serially connected and thus the contact damping can be
   * expressed as the series connection of two viscous dampers: \f$ c_*^{-1} =
   * c_1^{-1} + c_2^{-1}\f$.
   */
  inline real_t Material::get_tangential_damping(
      MaterialID material1, MaterialID material2) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");

    return inv(inv(get_tangential_damping(material1)) +
               inv(get_tangential_damping(material2)));
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Setting the coefficient of restitution between material \a material1
   * and \a material2.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \param cor The coefficient of restitution between \a material1 and \a
   * material2.
   * \return void
   */
  inline void Material::set_restitution(MaterialID material1,
                                        MaterialID material2,
                                        real_t cor) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    corTable_(material1, material2) = cor;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Setting the coefficient of static friction between material \a
   * material1 and \a material2.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \param csf The coefficient of static friction between \a material1 and \a
   * material2.
   * \return void
   */
  inline void Material::set_static_friction(MaterialID material1,
                                            MaterialID material2,
                                            real_t csf) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    csfTable_(material1, material2) = csf;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Setting the coefficient of forward static friction between material
   * \a material1 and \a material2.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \param forward_csf The coefficient of forward static friction between \a
   * material1 and \a material2. \return void
   */
  inline void Material::set_forward_static_friction(
      MaterialID material1, MaterialID material2, real_t forward_csf) noexcept {
    Material::set_static_friction(material1, material2, forward_csf);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Setting the coefficient of backward static friction between material
   * \a material1 and \a material2.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \param backward_csf The coefficient of backward static friction between \a
   * material1 and \a material2. \return void
   */
  inline void Material::set_backward_static_friction(
      MaterialID material1, MaterialID material2,
      real_t backward_csf) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    backward_csfTable_(material1, material2) = backward_csf;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Setting the coefficient of lateral static friction between material
   * \a material1 and \a material2.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \param lateral_csf The coefficient of lateral_static friction between \a
   * material1 and \a material2. \return void
   */
  inline void Material::set_lateral_static_friction(
      MaterialID material1, MaterialID material2, real_t lateral_csf) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    lateral_csfTable_(material1, material2) = lateral_csf;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Setting the coefficient of dynamic friction between material \a
   * material1 and \a material2.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \param cdf The coefficient of dynamic friction between \a material1 and \a
   * material2.
   * \return void
   */
  inline void Material::set_dynamic_friction(MaterialID material1,
                                             MaterialID material2,
                                             real_t cdf) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    cdfTable_(material1, material2) = cdf;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Setting the coefficient of forward dynamic friction between material
   * \a material1 and \a material2.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \param csf The coefficient of forward dynamic friction between \a material1
   * and \a material2.
   * \return void
   */
  inline void Material::set_forward_dynamic_friction(
      MaterialID material1, MaterialID material2, real_t forward_cdf) noexcept {
    Material::set_dynamic_friction(material1, material2, forward_cdf);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Setting the coefficient of backward dynamic friction between
   * material \a material1 and \a material2. \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \param backward_cdf The coefficient of backward dynamic friction between \a
   * material1 and \a material2. \return void
   */
  inline void Material::set_backward_dynamic_friction(
      MaterialID material1, MaterialID material2,
      real_t backward_cdf) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    backward_cdfTable_(material1, material2) = backward_cdf;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Setting the coefficient of lateral dynamic friction between material
   * \a material1 and \a material2.
   * \ingroup materials
   *
   * \param material1 The material of the first colliding rigid body.
   * \param material2 The material of the second colliding rigid body.
   * \param lateral_cdf The coefficient of lateral_dynamic friction between \a
   * material1 and \a material2. \return void
   */
  inline void Material::set_lateral_dynamic_friction(
      MaterialID material1, MaterialID material2, real_t lateral_cdf) noexcept {
    ELASTICA_ASSERT(material1 < materials_.size(), "Invalid material ID");
    ELASTICA_ASSERT(material2 < materials_.size(), "Invalid material ID");
    lateral_cdfTable_(material1, material2) = lateral_cdf;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Computes the shear modulus given the youngs modulus and the poisson
   * ratio
   * \ingroup materials
   *
   */
  inline real_t Material::calculate_shear_modulus(real_t young,
                                                  real_t poisson) noexcept {
    return static_cast<real_t>(0.5) * young /
           (static_cast<real_t>(1.0) + poisson);
  }
  //****************************************************************************

  //============================================================================
  //
  //  MATERIAL IRON
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Specification of the material iron.
   * \ingroup materials
   *
   * The Iron class represents the material iron. It is implemented as a veneer
   class for the
   * Material base class to set the properties of iron:
   *
   * - Name: "iron"
   * - Density: \f$ 7.874 \frac{kg}{dm^3} \f$
   * - Coefficient of restitution: 0.5
   * - Coefficient of static friction: 0.1
   * - Coefficient of dynamic friction: 0.1
   * - Poisson's ratio: 0.24
   * - Young's modulus: 200 GPa
   * - Stiffness: \f$ ~200 \frac{N}{m} \f$
   * - Normal Damping: \f$ 0 \frac{Ns}{m} \f$
   * - Tangential Damping: \f$ 0 \frac{Ns}{m} \f$
   *
   * Since several parameters are not unitless they might not match the scaling
   of the simulation.
   * In that case custom materials must be created. Also even though the
   stiffness is proportional
   * to Young's modulus the proportionality constant depends on other parameters
   such as the shape of
   * the contact region or the radii of the objects. Thus if the simulation does
   rely on the value of
   * the stiffness the user must supply an appropriate stiffness coefficient.
   Since no published
   * values were available for the damping coefficients they are deactivated.
   *
   * The iron material is automatically registered and can be directly used by
   the predefined
   * constant specifier \a iron:

     \code
     // Creating an iron sphere
     SphereID sphere = createSphere( 1, 0.0, 0.0, 0.0, iron );
     \endcode
   */
  class Iron : public Material {
   public:
    //**Constructor*************************************************************
    /*!\name Constructor */
    //@{
    explicit inline Iron();
    //@}
    //**************************************************************************
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Default constructor for the Iron class.
   */
  inline Iron::Iron()
      : Material("iron", static_cast<real_t>(7.874), static_cast<real_t>(0.5),
                 static_cast<real_t>(0.24), static_cast<real_t>(200),
                 Material::calculate_shear_modulus(200, 0.24),
                 static_cast<real_t>(200), static_cast<real_t>(1e-10),
                 static_cast<real_t>(1e-10), static_cast<real_t>(0.1),
                 static_cast<real_t>(0.1)) {
    // Added small dampingN and damping T to prevent division by 0 in inv.
    static_assert(sizeof(Material) == sizeof(Iron), "Not same size");
  }
  //****************************************************************************

  //=================================================================================================
  //
  //  MATERIAL COPPER
  //
  //=================================================================================================

  //*************************************************************************************************
  /*!\brief Specification of the material copper.
   * \ingroup materials
   *
   * The Copper class represents the material copper. It is implemented as a
   veneer class for
   * the Material base class to set the properties of iron:
   *
   * - Name: "copper"
   * - Density: \f$ 8.92 \frac{kg}{dm^3} \f$
   * - Coefficient of restitution: 0.5
   * - Coefficient of static friction: 0.1
   * - Coefficient of dynamic friction: 0.1
   * - Poisson's ratio: 0.33
   * - Young's modulus: 117 GPa
   * - Stiffness: \f$ ~117 \frac{N}{m} \f$
   * - Normal Damping: \f$ 0 \frac{Ns}{m} \f$
   * - Tangential Damping: \f$ 0 \frac{Ns}{m} \f$
   *
   * Since several parameters are not unitless they might not match the scaling
   of the simulation.
   * In that case custom materials must be created. Also even though the
   stiffness is proportional
   * to Young's modulus the proportionality constant depends on other parameters
   such as the shape of
   * the contact region or the radii of the objects. Thus if the simulation does
   rely on the value of
   * the stiffness the user must supply an appropriate stiffness coefficient.
   Since no published
   * values were available for the damping coefficients they are deactivated.
   *
   * The copper material is automatically registered and can be directly used by
   the predefined
   * constant specifier \a copper:

     \code
     // Creating a copper sphere
     SphereID sphere = createSphere( 1, 0.0, 0.0, 0.0, copper );
     \endcode
   */
  class Copper : public Material {
   public:
    //**Constructor*********************************************************************************
    /*!\name Constructor */
    //@{
    explicit inline Copper();
    //@}
    //**********************************************************************************************
  };
  //*************************************************************************************************

  //*************************************************************************************************
  /*!\brief Default constructor for the Copper class.
   */
  inline Copper::Copper()
      : Material("copper", static_cast<real_t>(8.92), static_cast<real_t>(0.5),
                 static_cast<real_t>(0.33), static_cast<real_t>(117),
                 Material::calculate_shear_modulus(117, 0.33),
                 static_cast<real_t>(117), static_cast<real_t>(1e-10),
                 static_cast<real_t>(1e-10), static_cast<real_t>(0.1),
                 static_cast<real_t>(0.1)) {
    static_assert(sizeof(Material) == sizeof(Copper), "Not same size");
  }
  //*************************************************************************************************

  //=================================================================================================
  //
  //  MATERIAL GRANITE
  //
  //=================================================================================================

  //*************************************************************************************************
  /*!\brief Specification of the material granite.
   * \ingroup materials
   *
   * The Granite class represents the material granite. It is implemented as a
   veneer class for
   * the Material base class to set the properties of granite:
   *
   * - Name: "granite"
   * - Density: \f$ 2.80 \frac{kg}{dm^3} \f$
   * - Coefficient of restitution: 0.5
   * - Coefficient of static friction: 0.1
   * - Coefficient of dynamic friction: 0.1
   * - Poisson's ratio: 0.25
   * - Young's modulus: 55 GPa
   * - Stiffness: \f$ ~55 \frac{N}{m} \f$
   * - Normal Damping: \f$ 0 \frac{Ns}{m} \f$
   * - Tangential Damping: \f$ 0 \frac{Ns}{m} \f$
   *
   * Since several parameters are not unitless they might not match the scaling
   of the simulation.
   * In that case custom materials must be created. Also even though the
   stiffness is proportional
   * to Young's modulus the proportionality constant depends on other parameters
   such as the shape of
   * the contact region or the radii of the objects. Thus if the simulation does
   rely on the value of
   * the stiffness the user must supply an appropriate stiffness coefficient.
   Since no published
   * values were available for the damping coefficients they are deactivated.
   *
   * The granite material is automatically registered and can be directly used
   by the predefined
   * constant specifier \a granite:

     \code
     // Creating a granite sphere
     SphereID sphere = createSphere( 1, 0.0, 0.0, 0.0, granite );
     \endcode
   */
  class Granite : public Material {
   public:
    //**Constructor*********************************************************************************
    /*!\name Constructor */
    //@{
    explicit inline Granite();
    //@}
    //**********************************************************************************************
  };
  //*************************************************************************************************

  //*************************************************************************************************
  /*!\brief Default constructor for the Granite class.
   */
  inline Granite::Granite()
      : Material("granite", static_cast<real_t>(2.8), static_cast<real_t>(0.5),
                 static_cast<real_t>(0.25), static_cast<real_t>(55),
                 Material::calculate_shear_modulus(55, 0.25),
                 static_cast<real_t>(55), static_cast<real_t>(1e-10),
                 static_cast<real_t>(1e-10), static_cast<real_t>(0.1),
                 static_cast<real_t>(0.1)) {
    static_assert(sizeof(Material) == sizeof(Granite), "Not same size");
  }
  //*************************************************************************************************

  //=================================================================================================
  //
  //  MATERIAL OAK
  //
  //=================================================================================================

  //*************************************************************************************************
  /*!\brief Specification of the material oak.
   * \ingroup materials
   *
   * The Oak class represents the material oak wood. It is implemented as a
   veneer class for the
   * Material base class to set the properties of oak wood:
   *
   * - Name: "oak"
   * - Density: \f$ 0.8 \frac{kg}{dm^3} \f$
   * - Coefficient of restitution: 0.5
   * - Coefficient of static friction: 0.1
   * - Coefficient of dynamic friction: 0.1
   * - Poisson's ratio: 0.35
   * - Young's modulus: 11 GPa
   * - Stiffness: \f$ ~11 \frac{N}{m} \f$
   * - Normal Damping: \f$ 0 \frac{Ns}{m} \f$
   * - Tangential Damping: \f$ 0 \frac{Ns}{m} \f$
   *
   * Since several parameters are not unitless they might not match the scaling
   of the simulation.
   * In that case custom materials must be created. Also even though the
   stiffness is proportional
   * to Young's modulus the proportionality constant depends on other parameters
   such as the shape of
   * the contact region or the radii of the objects. Thus if the simulation does
   rely on the value of
   * the stiffness the user must supply an appropriate stiffness coefficient.
   Since no published
   * values were available for the damping coefficients they are deactivated.
   *
   * The oak wood material is automatically registered and can be directly used
   by the predefined
   * constant specifier \a oak:

     \code
     // Creating an oak wood sphere
     SphereID sphere = createSphere( 1, 0.0, 0.0, 0.0, oak );
     \endcode
   */
  class Oak : public Material {
   public:
    //**Constructor*********************************************************************************
    /*!\name Constructor */
    //@{
    explicit inline Oak();
    //@}
    //**********************************************************************************************
  };
  //*************************************************************************************************

  //*************************************************************************************************
  /*!\brief Default constructor for the Oak class.
   */
  inline Oak::Oak()
      : Material("oak", static_cast<real_t>(0.8), static_cast<real_t>(0.5),
                 static_cast<real_t>(0.35), static_cast<real_t>(11),
                 Material::calculate_shear_modulus(11, 0.35),
                 static_cast<real_t>(11), static_cast<real_t>(1e-10),
                 static_cast<real_t>(1e-10), static_cast<real_t>(0.1),
                 static_cast<real_t>(0.1)) {
    static_assert(sizeof(Material) == sizeof(Oak), "Not same size");
  }
  //****************************************************************************

  //============================================================================
  //
  //  MATERIAL FIR
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Specification of the material fir.
   * \ingroup materials
   *
   * The Fir class represents the material fir wood. It is implemented as a
   veneer class for the
   * Material base class to set the properties of fir wood:
   *
   * - Name: "fir"
   * - Density: \f$ 0.5 \frac{kg}{dm^3} \f$
   * - Coefficient of restitution: 0.5
   * - Coefficient of static friction: 0.1
   * - Coefficient of dynamic friction: 0.1
   * - Poisson's ratio: 0.34
   * - Young's modulus: 13 GPa
   * - Stiffness: \f$ ~13 \frac{N}{m} \f$
   * - Normal Damping: \f$ 0 \frac{Ns}{m} \f$
   * - Tangential Damping: \f$ 0 \frac{Ns}{m} \f$
   *
   * Since several parameters are not unitless they might not match the scaling
   of the simulation.
   * In that case custom materials must be created. Also even though the
   stiffness is proportional
   * to Young's modulus the proportionality constant depends on other parameters
   such as the shape of
   * the contact region or the radii of the objects. Thus if the simulation does
   rely on the value of
   * the stiffness the user must supply an appropriate stiffness coefficient.
   Since no published
   * values were available for the damping coefficients they are deactivated.
   *
   * The fir wood material is automatically registered and can be directly used
   by the predefined
   * constant specifier \a fir:

     \code
     // Creating a fir wood sphere
     SphereID sphere = createSphere( 1, 0.0, 0.0, 0.0, fir );
     \endcode
   */
  class Fir : public Material {
   public:
    //**Constructor*************************************************************
    /*!\name Constructor */
    //@{
    explicit inline Fir();
    //@}
    //**************************************************************************
  };
  //****************************************************************************

  //****************************************************************************
  /*!\brief Default constructor for the Fir class.
   */
  inline Fir::Fir()
      : Material("fir", static_cast<real_t>(0.5), static_cast<real_t>(0.5),
                 static_cast<real_t>(0.34), static_cast<real_t>(13),
                 Material::calculate_shear_modulus(13, 0.34),
                 static_cast<real_t>(13), static_cast<real_t>(1e-10),
                 static_cast<real_t>(1e-10), static_cast<real_t>(0.1),
                 static_cast<real_t>(0.1)) {
    static_assert(sizeof(Material) == sizeof(Fir), "Not same size");
  }
  //****************************************************************************

  //============================================================================
  //
  //  MATERIAL CONSTANTS
  //
  //============================================================================

  //****************************************************************************
  /*!\brief ID for the material iron.
   * \ingroup materials
   *
   * This material can be used to create iron rigid bodies. Iron has the
   * following material properties:
   *
   * - Name: "iron"
   * - Density: \f$ 7.874 \frac{kg}{dm^3} \f$
   * - Restitution factor: 0.5
   */
  MaterialID const iron = 0;
  //****************************************************************************

  //****************************************************************************
  /*!\brief ID for the material copper.
   * \ingroup materials
   *
   * This material can be used to create copper rigid bodies. Copper has the
   * following material properties:
   *
   * - Name: "copper"
   * - Density: \f$ 8.92 \frac{kg}{dm^3} \f$
   * - Coefficient of restitution: 0.5
   * - Coefficient of static friction: 0.1
   * - Coefficient of dynamic friction: 0.1
   */
  MaterialID const copper = 1;
  //****************************************************************************

  //****************************************************************************
  /*!\brief ID for the material granite.
   * \ingroup materials
   *
   * This material can be used to create granite rigid bodies.
   */
  MaterialID const granite = 2;
  //****************************************************************************

  //****************************************************************************
  /*!\brief ID for the material oak wood.
   * \ingroup materials
   *
   * This material can be used to create rigid bodies made from oak wood.
   */
  MaterialID const oak = 3;
  //****************************************************************************

  //****************************************************************************
  /*!\brief ID for the material fir wood.
   * \ingroup materials
   *
   * This material can be used to create rigid bodies made from fir wood.
   */
  MaterialID const fir = 4;
  //****************************************************************************

  //****************************************************************************
  /*!\brief ID for an invalid material.
   * \ingroup materials
   *
   * This MaterialID is returned by the getMaterial() function in case no
   * material with the specified name is returned. This value should not be used
   * to create rigid bodies or in any other function!
   */
  MaterialID const invalid_material = static_cast<MaterialID>(-1);
  //****************************************************************************

  //============================================================================
  //
  //  MATERIAL FUNCTIONS
  //
  //============================================================================

  //****************************************************************************
  /*!\name Material functions */
  //@{
  MaterialID create_material(std::string const& name, real_t density,
                             real_t cor, real_t young, real_t shear,
                             real_t stiffness, real_t dampingN, real_t dampingT,
                             real_t forward_csf, real_t backward_csf,
                             real_t lateral_csf, real_t forward_cdf,
                             real_t backward_cdf, real_t lateral_cdf);
  MaterialID create_material(real_t density, real_t cor, real_t young,
                             real_t shear, real_t stiffness, real_t dampingN,
                             real_t dampingT, real_t forward_csf,
                             real_t backward_csf, real_t lateral_csf,
                             real_t forward_cdf, real_t backward_cdf,
                             real_t lateral_cdf);
  MaterialID create_material(std::string const& name, real_t density,
                             real_t cor, real_t young, real_t shear,
                             real_t stiffness, real_t dampingN, real_t dampingT,
                             real_t csf, real_t cdf);
  MaterialID create_material(real_t density, real_t cor, real_t young,
                             real_t shear, real_t stiffness, real_t dampingN,
                             real_t dampingT, real_t csf, real_t cdf);
  //@}
  //****************************************************************************

}  // namespace elastica

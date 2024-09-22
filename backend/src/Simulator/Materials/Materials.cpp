//==============================================================================
/*!
//  \file
//  \brief Source file for materials
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

//******************************************************************************
// Includes
//******************************************************************************

#include "Materials.hpp"

#include <sstream>
#include <stdexcept>

namespace elastica {

  //============================================================================
  //
  //  DEFINITION AND INITIALIZATION OF THE STATIC MEMBER VARIABLES
  //
  //============================================================================

  Materials Material::materials_;
  Material::MatrixType Material::corTable_;
  Material::MatrixType Material::csfTable_;
  Material::MatrixType Material::backward_csfTable_;
  Material::MatrixType Material::lateral_csfTable_;
  Material::MatrixType Material::cdfTable_;
  Material::MatrixType Material::backward_cdfTable_;
  Material::MatrixType Material::lateral_cdfTable_;

  // clang-format off
  // clang-tidy (UnusedGlobalDeclarationInspection)
  bool Material::materialsActivated_(activate_materials());  // NOLINT
  // clang-format on
  unsigned int Material::anonymousMaterials_ = 0;

  //============================================================================
  //
  //  CLASS MATERIALS
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Automatic registration of the default materials.
   *
   * \return \a true after the materials have been registered.
   */
  bool Material::activate_materials() noexcept {
    // Registering the default materials
    materials_.push_back(Iron());
    materials_.push_back(Copper());
    materials_.push_back(Granite());
    materials_.push_back(Oak());
    materials_.push_back(Fir());

    // Initializing the coefficients of restitution
    //                       | Iron                   | Copper                 |
    //                       Granite                | Oak                    |
    //                       Fir
    //                       |
    real_t const cor[5][5] = {
        {static_cast<real_t>(0.25), static_cast<real_t>(0.25),
         static_cast<real_t>(0.25), static_cast<real_t>(0.25),
         static_cast<real_t>(0.25)},  // Iron
        {static_cast<real_t>(0.25), static_cast<real_t>(0.25),
         static_cast<real_t>(0.25), static_cast<real_t>(0.25),
         static_cast<real_t>(0.25)},  // Copper
        {static_cast<real_t>(0.25), static_cast<real_t>(0.25),
         static_cast<real_t>(0.25), static_cast<real_t>(0.25),
         static_cast<real_t>(0.25)},  // Granite
        {static_cast<real_t>(0.25), static_cast<real_t>(0.25),
         static_cast<real_t>(0.25), static_cast<real_t>(0.25),
         static_cast<real_t>(0.25)},  // Oak
        {static_cast<real_t>(0.25), static_cast<real_t>(0.25),
         static_cast<real_t>(0.25), static_cast<real_t>(0.25),
         static_cast<real_t>(0.25)}  // Fir
    };
    corTable_ = cor;

    // Initializing the coefficients of static friction
    //                       | Iron                   | Copper                 |
    //                       Granite                | Oak                    |
    //                       Fir
    //                       |
    real_t const csf[5][5] = {
        {static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20)},  // Iron
        {static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20)},  // Copper
        {static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20)},  // Granite
        {static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20)},  // Oak
        {static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20)}  // Fir
    };
    csfTable_ = csf;
    backward_csfTable_ = csf;
    lateral_csfTable_ = csf;

    // Initializing the coefficients of dynamic friction
    //                       | Iron                   | Copper                 |
    //                       Granite                | Oak                    |
    //                       Fir
    //                       |
    real_t const cdf[5][5] = {
        {static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20)},  // Iron
        {static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20)},  // Copper
        {static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20)},  // Granite
        {static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20)},  // Oak
        {static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20), static_cast<real_t>(0.20),
         static_cast<real_t>(0.20)}  // Fir
    };
    cdfTable_ = cdf;
    lateral_cdfTable_ = cdf;
    backward_cdfTable_ = cdf;

    return true;
  }
  //****************************************************************************

  //============================================================================
  //
  //  MATERIAL FUNCTIONS
  //
  //============================================================================

  //****************************************************************************
  /*!\brief Creating a new custom material.
   * \ingroup materials
   *
   * \details
   * This function creates the new, custom material \a name with the given
   * properties. The following example illustrates the use of this function:
   * \example
   * \code
   * // Creates the material "myMaterial" with the following material
   * // properties:
   * //  - material density               : 2.54
   * //  - coefficient of restitution     : 0.8
   * //  - Young's modulus                : 80
   * //  - Shear modulus                  : 120
   * //  - Contact stiffness              : 100
   * //  - dampingN                       : 10
   * //  - dampingT                       : 11
   * //  - coefficient of forward static friction : 0.1
   * //  - coefficient of backward static friction : 0.2
   * //  - coefficient of lateral static friction : 0.15
   * //  - coefficient of forward dynamic friction: 0.05
   * //  - coefficient of backward dynamic friction: 0.1
   * //  - coefficient of lateral dynamic friction: 0.075
   * MaterialID myMaterial = create_material("myMaterial", 2.54, 0.8, 80,
   *                                          120, 100, 10, 11, 0.1, 0.2, 0.15,
   *                                          0.05, 0.1, 0.075);
   * \endcode
   *
   * In case the name of the material is already in use or if any of the
   * coefficients is not in its allowed range, a std::invalid_argument exception
   * is thrown.
   *
   * The coefficient of restitution is given for self-similar collisions that is
   * collision of bodies made of similar material. The composite coefficient of
   * restitution \f$e_*\f$ is estimated as proposed by Stronge:
   * \f$\frac{e_*^2}{k_*} = \frac{e_1^2}{k_1} + \frac{e_2^2}{k_2}\f$.
   *
   * \param name The name of the material.
   * \param density The density of the material \f$ (0..\infty) \f$.
   * \param cor The coefficient of restitution (COR) of the material
   * \f$ [0..1] \f$.
   * \param young The Young's modulus of the material \f$ (0..\infty) \f$.
   * \param shear The Shear modulus of the material \f$ (0..\infty) \f$.
   * \param stiffness The stiffness in normal direction of the material's
   * contact region.
   * \param dampingN The damping coefficient in normal direction of the
   * material's contact region.
   * \param dampingT The damping coefficient in tangential direction of the
   * material's contact region.
   * \param forward_csf The coefficient of static friction (CSF) of the material
   * in the forward direction \f$ [0..\infty) \f$.
   * \param backward_csf The coefficient of static friction (CSF) of the
   * material in the backward direction \f$ [0..\infty) \f$.
   * \param lateral_csf The coefficient of static friction (CSF) of the material
   * in the lateral direction \f$ [0..\infty) \f$.
   * \param forward_cdf The coefficient of dynamic friction (CDF) of the
   * material in the forward direction \f$ [0..\infty) \f$
   * \param backward_cdf The coefficient of dynamic friction (CDF) of the
   * material in the backward direction \f$ [0..\infty) \f$
   * \param lateral_cdf The coefficient of dynamic friction (CDF) of the
   * material in the lateral direction \f$ [0..\infty) \f$
   *
   * \return The MaterialID for the new material.
   * \exception std::invalid_argument Invalid material parameter.
   */
  MaterialID create_material(std::string const& name, real_t density,
                             real_t cor, real_t young, real_t shear,
                             real_t stiffness, real_t dampingN, real_t dampingT,
                             real_t forward_csf, real_t backward_csf,
                             real_t lateral_csf, real_t forward_cdf,
                             real_t backward_cdf, real_t lateral_cdf) {
    typedef Material M;

    // Checking the material name
    auto curr(M::materials_.cbegin());
    auto end(M::materials_.cend());
    for (; curr != end; ++curr) {
      if (curr->get_name() == name)
        throw std::invalid_argument("Material of that name already exists!");
    }

    // Checking the density
    if (density <= real_t(0))
      throw std::invalid_argument("Invalid material density!");

    // Checking the coefficient of restitution
    if (cor < real_t(0) or cor > real_t(1))
      throw std::invalid_argument("Invalid coefficient of restitution!");

    // Checking the coefficients of static friction
    if (forward_csf < real_t(0))
      throw std::invalid_argument(
          "Invalid coefficient of forward static friction!");

    // Checking the coefficient of static friction
    if (backward_csf < real_t(0))
      throw std::invalid_argument(
          "Invalid coefficient of backward static friction!");

    // Checking the coefficient of static friction
    if (lateral_csf < real_t(0))
      throw std::invalid_argument(
          "Invalid coefficient of lateral static friction!");

    // Checking the coefficient of dynamic friction
    if (forward_cdf < real_t(0))
      throw std::invalid_argument(
          "Invalid coefficient of forward dynamic friction!");

    // Checking the coefficient of dynamic friction
    if (backward_cdf < real_t(0))
      throw std::invalid_argument(
          "Invalid coefficient of backward dynamic friction!");

    // Checking the coefficient of dynamic friction
    if (lateral_cdf < real_t(0))
      throw std::invalid_argument(
          "Invalid coefficient of lateral dynamic friction!");

    //    // Checking the Poisson's ratio
    //    if (poisson < real_t(-1) or poisson > real_t(0.5))
    //      throw std::invalid_argument("Invalid Poisson's ratio");
    //
    // Checking the Young's modulus
    if (young <= real_t(0))
      throw std::invalid_argument("Invalid Young's modulus");

    if (shear <= real_t(0))
      throw std::invalid_argument("Invalid shear modulus");

    // Checking the stiffness
    if (stiffness <= real_t(0))
      throw std::invalid_argument("Invalid stiffness");

    // Checking the damping coefficients
    if (dampingN < real_t(0) or dampingT < real_t(0))
      throw std::invalid_argument("Invalid damping coefficients");

    // Registering the new material
    real_t const poisson(0.5);
    M::materials_.push_back(Material(name, density, cor, poisson, young, shear,
                                     stiffness, dampingN, dampingT, forward_csf,
                                     backward_csf, lateral_csf, forward_cdf,
                                     backward_cdf, lateral_cdf));
    MaterialID const mat(M::materials_.size() - 1);

    // Updating the restitution table, the static friction table and the dynamic
    // friction table
    M::corTable_.extend(1, 1, true);
    M::csfTable_.extend(1, 1, true);
    M::backward_csfTable_.extend(1, 1, true);
    M::lateral_csfTable_.extend(1, 1, true);
    M::cdfTable_.extend(1, 1, true);
    M::backward_cdfTable_.extend(1, 1, true);
    M::lateral_cdfTable_.extend(1, 1, true);

    ELASTICA_ASSERT(M::corTable_.rows() == M::corTable_.columns(),
                    "Invalid matrix size");

    ELASTICA_ASSERT(M::csfTable_.rows() == M::csfTable_.columns(),
                    "Invalid matrix size");
    ELASTICA_ASSERT(
        M::backward_csfTable_.rows() == M::backward_csfTable_.columns(),
        "Invalid matrix size");
    ELASTICA_ASSERT(
        M::lateral_csfTable_.rows() == M::lateral_csfTable_.columns(),
        "Invalid matrix size");

    ELASTICA_ASSERT(M::cdfTable_.rows() == M::cdfTable_.columns(),
                    "Invalid matrix size");
    ELASTICA_ASSERT(
        M::backward_cdfTable_.rows() == M::backward_cdfTable_.columns(),
        "Invalid matrix size");
    ELASTICA_ASSERT(
        M::lateral_cdfTable_.rows() == M::lateral_cdfTable_.columns(),
        "Invalid matrix size");

    M::corTable_(mat, mat) = cor;

    M::csfTable_(mat, mat) = forward_csf + forward_csf;
    M::backward_csfTable_(mat, mat) = backward_csf + backward_csf;
    M::lateral_csfTable_(mat, mat) = lateral_csf + lateral_csf;

    M::cdfTable_(mat, mat) = forward_cdf + forward_cdf;
    M::backward_cdfTable_(mat, mat) = backward_cdf + backward_cdf;
    M::lateral_cdfTable_(mat, mat) = lateral_cdf + lateral_cdf;

    for (Materials::size_type i = 0; i < mat; ++i) {
      M::corTable_(mat, i) = M::corTable_(i, mat) =
          std::sqrt((sq(M::materials_[i].get_restitution()) /
                         M::materials_[i].get_stiffness() +
                     sq(cor) / stiffness) *
                    M::get_stiffness(mat, i));

      M::csfTable_(mat, i) = M::csfTable_(i, mat) =
          M::materials_[i].get_static_friction() + forward_csf;
      M::backward_csfTable_(mat, i) = M::backward_csfTable_(i, mat) =
          M::materials_[i].get_backward_static_friction() + backward_csf;
      M::lateral_csfTable_(mat, i) = M::lateral_csfTable_(i, mat) =
          M::materials_[i].get_lateral_static_friction() + lateral_csf;

      M::cdfTable_(mat, i) = M::cdfTable_(i, mat) =
          M::materials_[i].get_dynamic_friction() + forward_cdf;
      M::backward_cdfTable_(mat, i) = M::backward_cdfTable_(i, mat) =
          M::materials_[i].get_backward_dynamic_friction() + backward_cdf;
      M::lateral_cdfTable_(mat, i) = M::lateral_cdfTable_(i, mat) =
          M::materials_[i].get_lateral_dynamic_friction() + lateral_cdf;
    }

    return mat;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Creating a new anonymous custom material.
   * \ingroup materials
   *
   * \details
   * This function creates a new, custom material with the given properties. It
   * will be named 'Material${X}', where X is an incrementing number.
   *
   * \param density The density of the material \f$ (0..\infty) \f$.
   * \param cor The coefficient of restitution (COR) of the material
   * \f$ [0..1] \f$.
   * \param young The Young's modulus of the material \f$ (0..\infty) \f$.
   * \param shear The Shear modulus of the material \f$ (0..\infty) \f$.
   * \param stiffness The stiffness in normal direction of the material's
   * contact region.
   * \param dampingN The damping coefficient in normal direction of the
   * material's contact region.
   * \param dampingT The damping coefficient in tangential direction of the
   * material's contact region.
   * \param forward_csf The coefficient of static friction (CSF) of the material
   * in the forward direction \f$ [0..\infty) \f$.
   * \param backward_csf The coefficient of static friction (CSF) of the
   * material in the backward direction \f$ [0..\infty) \f$.
   * \param lateral_csf The coefficient of static friction (CSF) of the material
   * in the lateral direction \f$ [0..\infty) \f$.
   * \param forward_cdf The coefficient of dynamic friction (CDF) of the
   * material in the forward direction \f$ [0..\infty) \f$
   * \param backward_cdf The coefficient of dynamic friction (CDF) of the
   * material in the backward direction \f$ [0..\infty) \f$
   * \param lateral_cdf The coefficient of dynamic friction (CDF) of the
   * material in the lateral direction \f$ [0..\infty) \f$
   *
   * \return The MaterialID for the new material.
   * \exception std::invalid_argument Invalid material parameter.
   */
  MaterialID create_material(real_t density, real_t cor, real_t young,
                             real_t shear, real_t stiffness, real_t dampingN,
                             real_t dampingT, real_t forward_csf,
                             real_t backward_csf, real_t lateral_csf,
                             real_t forward_cdf, real_t backward_cdf,
                             real_t lateral_cdf) {
    std::ostringstream sstr;

    do {
      if (Material::anonymousMaterials_ + 1 == 0)
        throw std::runtime_error("Index overflow for anonymous materials");
      sstr.str("");
      sstr << "Material" << ++Material::anonymousMaterials_;
    } while (Material::find(sstr.str()) != invalid_material);

    return create_material(sstr.str(), density, cor, young, shear, stiffness,
                           dampingN, dampingT, forward_csf, backward_csf,
                           lateral_csf, forward_cdf, backward_cdf, lateral_cdf);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Creating a new custom material.
   * \ingroup materials
   *
   * \details
   * This function creates the new, custom material \a name with the given
   * properties. The following example illustrates the use of this function:
   * \example
   * \code
   * // Creates the material "myMaterial" with the following material
   * // properties:
   * //  - material density               : 2.54
   * //  - coefficient of restitution     : 0.8
   * //  - Young's modulus                : 80
   * //  - Shear modulus                  : 120
   * //  - Contact stiffness              : 100
   * //  - dampingN                       : 10
   * //  - dampingT                       : 11
   * //  - coefficient of static friction : 0.1
   * //  - coefficient of dynamic friction: 0.05
   * MaterialID myMaterial = create_material("myMaterial", 2.54, 0.8,
   *                                          80, 120, 100, 10, 11, 0.1, 0.05);
   * \endcode
   *
   * In case the name of the material is already in use or if any of the
   * coefficients is not in its allowed range, a std::invalid_argument exception
   * is thrown.
   *
   * The coefficient of restitution is given for self-similar collisions that is
   * collision of bodies made of similar material. The composite coefficient of
   * restitution \f$e_*\f$ is estimated as proposed by Stronge:
   * \f$\frac{e_*^2}{k_*} = \frac{e_1^2}{k_1} + \frac{e_2^2}{k_2}\f$.
   *
   * \param name The name of the material.
   * \param density The density of the material \f$ (0..\infty) \f$.
   * \param cor The coefficient of restitution (COR) of the material
   * \f$ [0..1] \f$.
   * \param young The Young's modulus of the material \f$ (0..\infty) \f$.
   * \param shear The Shear modulus of the material \f$ (0..\infty) \f$.
   * \param stiffness The stiffness in normal direction of the material's
   * contact region.
   * \param dampingN The damping coefficient in normal direction of the
   * material's contact region.
   * \param dampingT The damping coefficient in tangential direction of the
   * material's contact region.
   * \param csf The coefficient of static friction (CSF) of the material
   * \f$ [0..\infty) \f$.
   * \param cdf The coefficient of dynamic friction (CDF) of the material \f$
   * [0..\infty) \f$.
   *
   * \return The MaterialID for the new material.
   * \exception std::invalid_argument Invalid material parameter.
   */
  MaterialID create_material(std::string const& name, real_t density,
                             real_t cor, real_t young, real_t shear,
                             real_t stiffness, real_t dampingN, real_t dampingT,
                             real_t csf, real_t cdf) {
    return create_material(name, density, cor, young, shear, stiffness,
                           dampingN, dampingT, csf, csf, csf, cdf, cdf, cdf);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Creating a new anonymous custom material.
   * \ingroup materials
   *
   * \details
   * This function creates a new, custom material with the given properties. It
   * will be named 'Material${X}', where X is an incrementing number.
   *
   * \param density The density of the material \f$ (0..\infty) \f$.
   * \param cor The coefficient of restitution (COR) of the material
   * \f$ [0..1] \f$.
   * \param poisson The Poisson's ratio of the material \f$ [-1..0.5] \f$.
   * \param young The Young's modulus of the material \f$ (0..\infty) \f$.
   * \param shear The Shear modulus of the material \f$ (0..\infty) \f$.
   * \param stiffness The stiffness in normal direction of the material's
   * contact region.
   * \param dampingN The damping coefficient in normal direction of the
   * material's contact region.
   * \param dampingT The damping coefficient in tangential direction of the
   * material's contact region.
   * \param csf The coefficient of static friction (CSF) of the material
   * \f$ [0..\infty) \f$.
   * \param cdf The coefficient of dynamic friction (CDF) of the
   * material \f$ [0..\infty) \f$
   *
   * \return The MaterialID for the new material.
   * \exception std::invalid_argument Invalid material parameter.
   */
  MaterialID create_material(real_t density, real_t cor, real_t young,
                             real_t shear, real_t stiffness, real_t dampingN,
                             real_t dampingT, real_t csf, real_t cdf) {
    return create_material(density, cor, young, shear, stiffness, dampingN,
                           dampingT, csf, csf, csf, cdf, cdf, cdf);
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Searching for a registered material.
   * \ingroup materials
   *
   * \param name The name of the material.
   * \return The MaterialID of the material if the material is found, \a
   * invalid_material otherwise.
   * \exception std::runtime_error, in case an invalid name is specified
   *
   * The function searches for a registered material with the given name. If the
   * material is found, the corresponding MaterialID is returned. Otherwise, \a
   * invalid_material is returned.
   */
  MaterialID Material::find(std::string const& name) noexcept {
    for (Material::SizeType i = 0; i < Material::materials_.size(); ++i) {
      if (Material::materials_[i].get_name() == name) {
        return i;
      }
    }
    // We return invalid material rather than raise an exception here for
    // having anonymous materials
    return invalid_material;
  }
  //****************************************************************************

  //****************************************************************************
  /*!\brief Searching for registered materials with a prefix.
  // \ingroup materials
  //
  // \param prefix The prefix common to the names of the materials.
  // \return A std::vector object containing the MaterialIDs of all materials
  // found.
  //
  // The function collects all registered materials with names beginning with
  // the given string. Their IDs are assembled in an std::vector object. If no
  // Materials are found, the container is empty.
  */
  std::vector<MaterialID> Material::find_prefix(std::string const& prefix) {
    std::vector<MaterialID> results;
    for (Material::SizeType i = 0; i < Material::materials_.size(); ++i) {
      if (Material::materials_[i].get_name().compare(0, prefix.size(),
                                                     prefix) == 0) {
        results.push_back(i);
      }
    }
    return results;
  }
  //****************************************************************************

}  // namespace elastica

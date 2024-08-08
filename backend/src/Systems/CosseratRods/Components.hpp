#pragma once

//******************************************************************************
// Includes
//******************************************************************************
/// Types first
#include "Systems/CosseratRods/Components/Types.hpp"
///
#include "Systems/CosseratRods/Components/Elasticity.hpp"
#include "Systems/CosseratRods/Components/Geometry.hpp"
#include "Systems/CosseratRods/Components/Kinematics.hpp"
#include "Systems/CosseratRods/Components/Names.hpp"
#include "Systems/CosseratRods/Components/Tags.hpp"
#include "Systems/CosseratRods/Components/helpers.hpp"

namespace elastica {

  namespace cosserat_rod {

    //**************************************************************************
    /*!\defgroup cosserat_rod_component Cosserat Rod Components
     * \ingroup cosserat_rod
     * \brief Data-structures to customize orthogonal aspects of Cosserat rods
     *
     * \details
     * `Components` are structures customizing orthogonal aspects of a Cosserat
     * rod within \elastica. Indeed, a typical rod has features broadly
     * encompassing geometry, elasticity etc. which can be decoupled from one
     * another and whose communication happens by a common interface/protocol.
     * These features are implemented as `Components`. Within
     * the parlance of `c++`, `Components` are implemented as curious mixins
     * intended as policy classes that control the behavior of the downstream
     * Cosserat-rod blocks and its slices.
     */
    //**************************************************************************

    //**************************************************************************
    /*!\defgroup cosserat_rod_custom_entries Cosserat Rod customization
     * \ingroup cosserat_rod_component
     * \brief Entry points to customize core-rod functionality
     *
     * \details
     * To customize the behavior of the library-provided CosseratRods, users are
     * encouraged to overload these set of functions with the following
     * signature
     * TODO
     */
    //**************************************************************************

    //**************************************************************************
    /*!\brief Components comprising Cosserat rods
    // \ingroup cosserat_rod_component
    */
    namespace component {}
    //**************************************************************************

  }  // namespace cosserat_rod

}  // namespace elastica

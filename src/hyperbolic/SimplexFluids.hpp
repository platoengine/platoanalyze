/*
 * SimplexFluids.hpp
 *
 *  Created on: Apr 5, 2021
 */

#pragma once

#include "Simplex.hpp"

namespace Plato
{

/***************************************************************************//**
 *  \tparam SpaceDim    (integer) spatial dimensions
 *  \tparam NumControls (integer) number of design variable fields (default = 1)
 *
 *  \brief Base class for simplex fluid mechanics problems
 ******************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 Plato::OrdinalType NumControls = 1>
class SimplexFluids : public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;  /*!< number of spatial dimensions */
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per simplex cell */

    // optimization quantities of interest
    static constexpr Plato::OrdinalType mNumConfigDofsPerNode  = mNumSpatialDims; /*!< number of configuration degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumControlDofsPerNode = NumControls;     /*!< number of controls per node */
    static constexpr Plato::OrdinalType mNumConfigDofsPerCell  = mNumConfigDofsPerNode * mNumNodesPerCell;  /*!< number of configuration degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumControlDofsPerCell = mNumControlDofsPerNode * mNumNodesPerCell; /*!< number of controls per cell */

    // physical quantities of interest
    static constexpr Plato::OrdinalType mNumMassDofsPerNode     = 1; /*!< number of continuity degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumMassDofsPerCell     = mNumMassDofsPerNode * mNumNodesPerCell; /*!< number of continuity degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumEnergyDofsPerNode   = 1; /*!< number energy degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumEnergyDofsPerCell   = mNumEnergyDofsPerNode * mNumNodesPerCell; /*!< number of energy degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumMomentumDofsPerNode = mNumSpatialDims; /*!< number of momentum degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumMomentumDofsPerCell = mNumMomentumDofsPerNode * mNumNodesPerCell; /*!< number of momentum degrees of freedom per cell */

};
// class SimplexFluids

}
// namespace Plato

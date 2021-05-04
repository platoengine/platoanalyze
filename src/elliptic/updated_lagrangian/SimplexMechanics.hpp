#pragma once

#include "Simplex.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

/******************************************************************************/
/*! Base class for simplex-based updated lagrangian mechanics
 *
 *  The global state consists of nodal displacements, and the local state
 *  consists of the reference strain.
 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexMechanics : public Plato::Simplex<SpaceDim>
{
  public:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNumVoigtTerms   = (SpaceDim == 3) ? 6 :
                                             ((SpaceDim == 2) ? 3 :
                                            (((SpaceDim == 1) ? 1 : 0)));

    // obligatory (used by SimplexFadTypes to define the static AD types)
    static constexpr Plato::OrdinalType mNumNodeStatePerNode = 0;
    static constexpr Plato::OrdinalType mNumDofsPerNode      = SpaceDim;
    static constexpr Plato::OrdinalType mNumDofsPerCell      = mNumDofsPerNode*mNumNodesPerCell;
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = mNumVoigtTerms; // local state: reference strain
    // end obligatory

    static constexpr Plato::OrdinalType mNumControl = NumControls;

};
} // namespace UpdatedLagrangian

} // namespace Elliptic

} // namespace Plato

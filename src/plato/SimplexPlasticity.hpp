#pragma once

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexPlasticity : public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;  /*!< number of spatial dimensions */
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */

    static constexpr Plato::OrdinalType mNumVoigtTerms =
            (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 3 : (((SpaceDim == 1) ? 1 : 0))); /*!< number of Voigt terms */

    // degree-of-freedom attributes
    static constexpr auto mNumControl = NumControls;                            /*!< number of controls */
    static constexpr auto mNumDofsPerNode = SpaceDim + 1;                       /*!< number of degrees of freedom per node { disp_x, disp_y, disp_z, pressure} */
    static constexpr auto mPressureDofOffset = SpaceDim;                        /*!< number of pressure degrees of freedom offset */
    static constexpr auto mNumDofsPerCell = mNumDofsPerNode * mNumNodesPerCell; /*!< number of degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell =
            (SpaceDim == 3) ? 14 : ((SpaceDim == 2) ? 8 : (((SpaceDim == 1) ? 4 : 0))); /*!< number of local degrees of freedom per cell for J2-plasticity*/

    // This physics can be used with Variational Multi-scale (VMS) functionality
    // in PA. The following defines the nodal state attributes required by VMS.
    static constexpr auto mNumNodeStatePerNode = SpaceDim;                                /*!< number of node states, i.e. pressure gradient, dofs per node */
    static constexpr auto mNumNodeStatePerCell = mNumNodeStatePerNode * mNumNodesPerCell; /*!< number of node states, i.e. pressure gradient, dofs  per cell */
};
// class SimplexPlasticity

} // namespace Plato

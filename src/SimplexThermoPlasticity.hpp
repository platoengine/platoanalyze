#pragma once

#include "Simplex.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexThermoPlasticity : public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;  /*!< number of spatial dimensions */
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */

    /*!< number of rows and columns for second order stress and strain tensors */
    static constexpr Plato::OrdinalType mNumStressTerms =
            (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 4 : (((SpaceDim == 1) ? 1: 0)));

    static constexpr Plato::OrdinalType mNumVoigtTerms =
            (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 3 : (((SpaceDim == 1) ? 1 : 0))); /*!< number of Voigt terms */

    // degree-of-freedom attributes
    static constexpr auto mNumControl = NumControls;                            /*!< number of controls */
    static constexpr auto mNumDofsPerNode = mNumSpatialDims + 2;                /*!< number of global degrees of freedom per node { disp_x, disp_y, disp_z, temperature, pressure} */
    static constexpr Plato::OrdinalType mDisplacementDofOffset = 0;             /*!< displacement degrees of freedom offset */
    static constexpr auto mTemperatureDofOffset = mNumSpatialDims;              /*!< temperature degrees of freedom offset */
    static constexpr auto mPressureDofOffset = mNumSpatialDims + 1;             /*!< pressure degrees of freedom offset */
    static constexpr auto mNumDofsPerCell = mNumDofsPerNode * mNumNodesPerCell; /*!< number of global degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell =
            (SpaceDim == 3) ? 14 : ((SpaceDim == 2) ? 10 : (((SpaceDim == 1) ? 4 : 0))); /*!< number of local degrees of freedom per cell for J2-plasticity*/

    // This physics can be used with Variational Multi-scale (VMS) functionality
    // in PA. The following defines the nodal state attributes required by VMS.
    static constexpr auto mNumNodeStatePerNode = mNumSpatialDims;                         /*!< number of node states, i.e. pressure gradient, dofs per node */
    static constexpr auto mNumNodeStatePerCell = mNumNodeStatePerNode * mNumNodesPerCell; /*!< number of node states, i.e. pressure gradient, dofs per cell */
};
// class SimplexThermoPlasticity

}// namespace Plato

#pragma once

#include "Simplex.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexThermoPlasticity : public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;

    /*!< number of rows and columns for second order stress and strain tensors */
    static constexpr Plato::OrdinalType mNumStressTerms =
            (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 4 : (((SpaceDim == 1) ? 1 : 0)));

    static constexpr Plato::OrdinalType mNumVoigtTerms =
            (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 3 : (((SpaceDim == 1) ? 1 : 0)));
    static constexpr Plato::OrdinalType mNumDofsPerNode = SpaceDim + 2; // displacement + pressure + temperature
    static constexpr Plato::OrdinalType mNumDofsPerCell = mNumDofsPerNode * mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell =
            (SpaceDim == 3) ? 14 : ((SpaceDim == 2) ? 8 : (((SpaceDim == 1) ? 4 : 0)));

    static constexpr Plato::OrdinalType mNumControl = NumControls;

    static constexpr Plato::OrdinalType mNumNodeStatePerNode = SpaceDim;
    static constexpr Plato::OrdinalType mNumNodeStatePerCell = mNumNodeStatePerNode * mNumNodesPerCell;
};
// class SimplexPlasticity

}// namespace Plato

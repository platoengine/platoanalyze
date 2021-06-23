#pragma once

#include "PlatoStaticsTypes.hpp"
#include "Simplex.hpp"

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************/
/*! Add mass term functor.

 Given filtered density field, compute the "mass" term.
 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumDofsPerNode = 1, Plato::OrdinalType DofOffset = 0>
class AddMassTerm : public Plato::Simplex<SpaceDim>
{
private:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */

public:
    /******************************************************************************//**
     * \brief Add mass term
     * \param [in] aCellOrdinal cell (i.e. element) ordinal
     * \param [in/out] aOutput output, i.e. Helmholtz residual
     * \param [in] aFilteredDensity input filtered density workset
     * \param [in] aUnfilteredDensity input unfiltered density workset
     * \param [in] aBasisFunctions basis functions
     * \param [in] aCellVolume cell volume
    **********************************************************************************/
    template<typename ResultScalarType, typename StateScalarType, typename ControlScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarMultiVectorT<ResultScalarType> & aOutput,
                                       const Plato::ScalarVectorT<StateScalarType> & aFilteredDensity,
                                       const Plato::ScalarVectorT<ControlScalarType> & aUnfilteredDensity,
                                       const Plato::ScalarVectorT<Plato::Scalar> & aBasisFunctions,
                                       const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tLocalOrdinal = tNodeIndex * NumDofsPerNode + DofOffset;
            aOutput(aCellOrdinal, tLocalOrdinal) += aBasisFunctions(tNodeIndex) * ( aFilteredDensity(aCellOrdinal) - aUnfilteredDensity(aCellOrdinal) ) * aCellVolume(aCellOrdinal);
        }
    }
};
// class AddMassTerm

} // namespace Helmholtz

} // namespace Plato

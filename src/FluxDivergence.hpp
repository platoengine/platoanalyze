#pragma once

#include "PlatoStaticsTypes.hpp"
#include "Simplex.hpp"

namespace Plato
{

/******************************************************************************/
/*! Flux divergence functor.

 Given a thermal flux, compute the flux divergence.
 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumDofsPerNode = 1, Plato::OrdinalType DofOffset = 0>
class FluxDivergence : public Plato::Simplex<SpaceDim>
{
private:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */

public:
    /******************************************************************************//**
     * \brief Compute flux divergence
     * \param [in] aCellOrdinal cell (i.e. element) ordinal
     * \param [in/out] aOutput output, i.e. flux divergence
     * \param [in] aFlux input flux workset
     * \param [in] aGradient configuration gradients
     * \param [in] aStateValues 2D state values workset
     * \param [in] aScale scale parameter (default = 1.0)
    **********************************************************************************/
    template<typename ForcingScalarType, typename FluxScalarType, typename GradientScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void operator()(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarMultiVectorT<ForcingScalarType> aOutput,
                                       Plato::ScalarMultiVectorT<FluxScalarType> aFlux,
                                       Plato::ScalarArray3DT<GradientScalarType> aGradient,
                                       Plato::ScalarVectorT<VolumeScalarType> aCellVolume,
                                       Plato::Scalar aScale = 1.0) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tLocalOrdinal = tNodeIndex * NumDofsPerNode + DofOffset;
            for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
            {
                aOutput(aCellOrdinal, tLocalOrdinal) += aScale * aFlux(aCellOrdinal, tDimIndex)
                        * aGradient(aCellOrdinal, tNodeIndex, tDimIndex) * aCellVolume(aCellOrdinal);
            }
        }
    }
};
// class FluxDivergence

}
// namespace Plato

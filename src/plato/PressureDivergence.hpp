#ifndef PRESSURE_DIVERGENCE
#define PRESSURE_DIVERGENCE

#include "plato/SimplexMechanics.hpp"

namespace Plato
{

/******************************************************************************/
/*! Pressure Divergence functor.

 Given a pressure, p, and divergence matrix, b, compute the pressure divergence.

 f_{e,j(I,i)} = s v p_{e} b_{e,I,i}

 e:  element index
 I:  local node index
 i:  dimension index
 s:  scale factor.  1.0 by default.
 v:  cell volume.  Single value per cell, so single point integration is assumed.
 p_{e}:  pressure in element, e
 b_{e,I,i}:  basis derivative of local node I of element e with respect to dimension i
 j(I,i):  strided 1D index of local dof i for local node I.
 j(I,i) = I*NumDofsPerNode + i + DofOffset

 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumDofsPerNode = SpaceDim, Plato::OrdinalType DofOffset = 0>
class PressureDivergence : public Plato::SimplexMechanics<SpaceDim>
{
private:
    using Plato::SimplexMechanics<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */

public:
    /******************************************************************************//**
     * \brief Compute the divergence of the pressure field
     * \param [in] aCellOrdinal cell (i.e. element ordinal)
     * \param [in/out] aOutput pressure divergence workset
     * \param [in] aPressure pressure workset
     * \param [in] aGradient configuration gradients
     * \param [in] aCellVolume cell (i.e. element) volume
     * \param [in] aScale scalar parameter (default = 1.0)
    **********************************************************************************/
    template<typename ForcingScalarType, typename PressureScalarType, typename GradientScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline
    void operator()(Plato::OrdinalType aCellOrdinal,
                    Plato::ScalarMultiVectorT<ForcingScalarType> aOutput,
                    Plato::ScalarVectorT<PressureScalarType> aPressure,
                    Plato::ScalarArray3DT<GradientScalarType> aGradient,
                    Plato::ScalarVectorT<VolumeScalarType> aCellVolume,
                    Plato::Scalar aScale = 1.0) const
    {
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * NumDofsPerNode + tDimIndex + DofOffset;
                aOutput(aCellOrdinal, tLocalOrdinal) += aScale * aCellVolume(aCellOrdinal) * aPressure(aCellOrdinal)
                        * aGradient(aCellOrdinal, tNodeIndex, tDimIndex);
            }
        }
    }
};
// class PressureDivergence

}
// namespace Plato

#endif

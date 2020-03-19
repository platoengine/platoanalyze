#ifndef STRESS_DIVERGENCE
#define STRESS_DIVERGENCE

#include "SimplexMechanics.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Stress Divergence functor. Given the stress tensor, apply the divergence
 *  operator to the stress tensor.
 *
 * \tparam SpaceDim       spatial dimensions
 * \tparam NumDofsPerNode number of degrees of freedom per node
 * \tparam DofOffset      offset apply to degree of freedom indexing
*******************************************************************************/
template<Plato::OrdinalType SpaceDim,
         Plato::OrdinalType NumDofsPerNode = SpaceDim,
         Plato::OrdinalType DofOffset = 0>
class StressDivergence : public Plato::SimplexMechanics<SpaceDim>
{
private:
    using Plato::SimplexMechanics<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell, i.e. element */

    // 2-D Example: mVoigt[0][0] = 0, mVoigt[0][1] = 2, mVoigt[1][0] = 2, mVoigt[1][1] = 1,
    // where the stress tensor in Voigt notation is given s = {s_11, s_22, s_12, s_33} (plane strain)
    Plato::OrdinalType mVoigt[SpaceDim][SpaceDim]; /*!< matrix with indices to stress tensor entries in Voigt notation */

public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    StressDivergence()
    {
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < SpaceDim; tDofIndex++)
        {
            mVoigt[tDofIndex][tDofIndex] = tVoigtTerm++;
        }
        for(Plato::OrdinalType tDofIndexJ = SpaceDim - 1; tDofIndexJ >= 1; tDofIndexJ--)
        {
            for(Plato::OrdinalType tDofIndexI = tDofIndexJ - 1; tDofIndexI >= 0; tDofIndexI--)
            {
                mVoigt[tDofIndexI][tDofIndexJ] = tVoigtTerm;
                mVoigt[tDofIndexJ][tDofIndexI] = tVoigtTerm++;
            }
        }
    }

    /***************************************************************************//**
     * \brief Apply stress divergence operator to stress tensor
     *
     * \tparam ForcingScalarType   Kokkos::View POD type
     * \tparam StressScalarType    Kokkos::View POD type
     * \tparam GradientScalarType  Kokkos::View POD type
     * \tparam VolumeScalarType    Kokkos::View POD type
     *
     * \param aCellOrdinal cell index
     * \param aOutput      stress divergence
     * \param aStress      stress tensor
     * \param aGradient    spatial gradient tensor
     * \param aCellVolume  cell volume
     * \param aScale       multiplier (default = 1.0)
     *
    *******************************************************************************/
    template<typename ForcingScalarType,
             typename StressScalarType,
             typename GradientScalarType,
             typename VolumeScalarType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarMultiVectorT<ForcingScalarType> & aOutput,
                                       const Plato::ScalarMultiVectorT<StressScalarType> & aStress,
                                       const Plato::ScalarArray3DT<GradientScalarType> & aGradient,
                                       const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                                       const Plato::Scalar aScale = 1.0) const
    {

        for(Plato::OrdinalType tDimIndexI = 0; tDimIndexI < SpaceDim; tDimIndexI++)
        {
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * NumDofsPerNode + tDimIndexI + DofOffset;
                for(Plato::OrdinalType tDimIndexJ = 0; tDimIndexJ < SpaceDim; tDimIndexJ++)
                {
                    aOutput(aCellOrdinal, tLocalOrdinal) +=
                        aScale * aCellVolume(aCellOrdinal) * aStress(aCellOrdinal, mVoigt[tDimIndexI][tDimIndexJ]) * aGradient(aCellOrdinal, tNodeIndex, tDimIndexJ);
                }
            }
        }
    }
};

} // namespace Plato

#endif

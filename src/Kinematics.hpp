#ifndef KINEMATICS_HPP
#define KINEMATICS_HPP

#include "SimplexMechanics.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Two-field Mechanical kinematics functor.

 Given a gradient matrix and state array, compute the pressure gradient
 and symmetric gradient of the displacement.
 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class StabilizedKinematics : public Plato::SimplexStabilizedMechanics<SpaceDim>
{
private:
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumVoigtTerms;     /*!< number of Voigt terms */
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumNodesPerCell;   /*!< number of nodes per cell */
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumDofsPerNode;    /*!< number of nodes per node */
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mPressureDofOffset; /*!< number of pressure dofs offset */

public:
    /***********************************************************************************
     * \brief Compute deviatoric stress, volume flux, cell stabilization
     * \param [in] aCellOrdinal cell ordinal
     * \param [in/out] aStrain displacement strains workset on H^{1}(\Omega)
     * \param [in/out] aPressureGrad pressure gradient workset on L^2(\Omega)
     * \param [in] aState state workset
     * \param [in] aGradient configuration gradient workset
     **********************************************************************************/
    template<typename StrainScalarType, typename StateScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void operator()(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarMultiVectorT<StrainScalarType> const& aStrain,
                                       Plato::ScalarMultiVectorT<StrainScalarType> const& aPressureGrad,
                                       Plato::ScalarMultiVectorT<StateScalarType> const& aState,
                                       Plato::ScalarArray3DT<GradientScalarType> const& aGradient) const
    {
        // compute strain
        //
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < SpaceDim; tDofIndex++)
        {
            aStrain(aCellOrdinal, tVoigtTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + tDofIndex;
                aStrain(aCellOrdinal, tVoigtTerm) += aState(aCellOrdinal, tLocalOrdinal)
                        * aGradient(aCellOrdinal, tNodeIndex, tDofIndex);
            }
            tVoigtTerm++;
        }

        for(Plato::OrdinalType tDofJ = SpaceDim - 1; tDofJ >= 1; tDofJ--)
        {
            for(Plato::OrdinalType tDofI = tDofJ - 1; tDofI >= 0; tDofI--)
            {
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
                {
                    Plato::OrdinalType tLocalOrdinalI = tNodeIndex * mNumDofsPerNode + tDofI;
                    Plato::OrdinalType tLocalOrdinalJ = tNodeIndex * mNumDofsPerNode + tDofJ;
                    aStrain(aCellOrdinal, tVoigtTerm) +=
                            ( aState(aCellOrdinal, tLocalOrdinalJ) * aGradient(aCellOrdinal, tNodeIndex, tDofI)
                            + aState(aCellOrdinal, tLocalOrdinalI) * aGradient(aCellOrdinal, tNodeIndex, tDofJ) );
                }
                tVoigtTerm++;
            }
        }

        // compute pressure gradient
        //
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < SpaceDim; tDofIndex++)
        {
            aPressureGrad(aCellOrdinal, tDofIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * mNumDofsPerNode + mPressureDofOffset;
                aPressureGrad(aCellOrdinal, tDofIndex) += aState(aCellOrdinal, tLocalOrdinal)
                        * aGradient(aCellOrdinal, tNodeIndex, tDofIndex);
            }
        }
    }
};
// class StabilizedKinematics

}
// namespace Plato

#endif

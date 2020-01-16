#ifndef LGR_PLATO_KINETICS_HPP
#define LGR_PLATO_KINETICS_HPP

#include "SimplexStabilizedMechanics.hpp"
#include "LinearElasticMaterial.hpp"
#include <Omega_h_matrix.hpp>

namespace Plato
{

/******************************************************************************/
/*! Two-field Elasticity functor.

 given: strain, pressure gradient, fine scale displacement, pressure

 compute: deviatoric stress, volume flux, cell stabilization
 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class StabilizedKinetics : public Plato::SimplexStabilizedMechanics<SpaceDim>
{
private:
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumVoigtTerms; /*!< number of Voigt terms */
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumDofsPerNode; /*!< number of nodes per node */
    using Plato::SimplexStabilizedMechanics<SpaceDim>::mNumDofsPerCell; /*!< number of nodes per cell */

    const Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness; /*!< material matrix with lame constants */
    Plato::Scalar mBulkModulus, mShearModulus; /*!< shear and bulk moduli */

    const Plato::Scalar mPressureScaling; /*!< pressure scaling term */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model interface
    **********************************************************************************/
    StabilizedKinetics(const Teuchos::RCP<Plato::LinearElasticMaterial<SpaceDim>> aMaterialModel) :
            mCellStiffness(aMaterialModel->getStiffnessMatrix()),
            mBulkModulus(0.0),
            mShearModulus(0.0),
            mPressureScaling(aMaterialModel->getPressureScaling())
    {
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            mBulkModulus += mCellStiffness(0, tDimIndex);
        }
        mBulkModulus /= SpaceDim;

        Plato::OrdinalType tNumShear = mNumVoigtTerms - SpaceDim;
        for(Plato::OrdinalType tShearIndex = 0; tShearIndex < tNumShear; tShearIndex++)
        {
            mShearModulus += mCellStiffness(tShearIndex + SpaceDim, tShearIndex + SpaceDim);
        }
        mShearModulus /= tNumShear;
    }

    /***********************************************************************************
     * \brief Compute deviatoric stress, volume flux, cell stabilization
     * \param [in] aCellOrdinal cell ordinal
     * \param [in] aCellVolume cell volume workset
     * \param [in] aProjectedPGrad projected pressure gradient workset on H^{1}(\Omega)
     * \param [in] aPressure pressure workset on L^2(\Omega)
     * \param [in] aStrain displacement strains workset on H^{1}(\Omega)
     * \param [in] aPressureGrad pressure gradient workset on L^2(\Omega)
     * \param [out] aDevStress deviatoric stress workset
     * \param [out] aVolumeFlux volume flux workset
     * \param [out] aCellStabilization stabilization term workset
     **********************************************************************************/
    template<typename KineticsScalarType, typename KinematicsScalarType, typename NodeStateScalarType,
            typename VolumeScalarType>
    DEVICE_TYPE inline void operator()(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarVectorT<VolumeScalarType> const& aCellVolume,
                                       Plato::ScalarMultiVectorT<NodeStateScalarType> const& aProjectedPGrad,
                                       Plato::ScalarVectorT<KineticsScalarType> const& aPressure,
                                       Plato::ScalarMultiVectorT<KinematicsScalarType> const& aStrain,
                                       Plato::ScalarMultiVectorT<KinematicsScalarType> const& aPressureGrad,
                                       Plato::ScalarMultiVectorT<KineticsScalarType> const& aDevStress,
                                       Plato::ScalarVectorT<KineticsScalarType> const& aVolumeFlux,
                                       Plato::ScalarMultiVectorT<KineticsScalarType> const& aCellStabilization) const
    {
        // compute deviatoric stress, i.e. \sigma - \frac{1}{3}trace(\sigma)
        //
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            aDevStress(aCellOrdinal, tVoigtIndex_I) = static_cast<Plato::Scalar>(0.0);
            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
                aDevStress(aCellOrdinal, tVoigtIndex_I) += aStrain(aCellOrdinal, tVoigtIndex_J)
                        * mCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
        KineticsScalarType tTraceStress(0.0);
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            tTraceStress += aDevStress(aCellOrdinal, tDimIndex);
        }
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            aDevStress(aCellOrdinal, tDimIndex) -= tTraceStress / static_cast<Plato::Scalar>(3.0);
        }
        //printf("DevStress(%d,1) = %.10f\n", aCellOrdinal, aDevStress(aCellOrdinal,0));
        //printf("DevStress(%d,2) = %.10f\n", aCellOrdinal, aDevStress(aCellOrdinal,1));
        //printf("DevStress(%d,3) = %.10f\n", aCellOrdinal, aDevStress(aCellOrdinal,2));

        // compute volume strain, i.e. \div(u)
        //
        KinematicsScalarType tVolumetricStrain = 0.0;
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            tVolumetricStrain += aStrain(aCellOrdinal, tDimIndex);
        }

        // compute volume difference
        //
        aPressure(aCellOrdinal) *= mPressureScaling;
        aVolumeFlux(aCellOrdinal) = mPressureScaling * (tVolumetricStrain - aPressure(aCellOrdinal) / mBulkModulus);

        // compute cell stabilization
        //
        KinematicsScalarType tTau = pow(aCellVolume(aCellOrdinal), static_cast<Plato::Scalar>(2.0 / 3.0))
                / (static_cast<Plato::Scalar>(2.0) * mShearModulus);
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            aCellStabilization(aCellOrdinal, tDimIndex) = mPressureScaling
                    * tTau * (mPressureScaling * aPressureGrad(aCellOrdinal, tDimIndex) - aProjectedPGrad(aCellOrdinal, tDimIndex));
        }
    }
};
// class StabilizedKinetics

}
// namespace Plato

#endif

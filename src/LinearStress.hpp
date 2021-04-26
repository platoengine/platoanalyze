#ifndef PLATO_LINEAR_STRESS_HPP
#define PLATO_LINEAR_STRESS_HPP

#include "SimplexMechanics.hpp"
#include "LinearElasticMaterial.hpp"

#include <Omega_h_matrix.hpp>

namespace Plato
{

/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}
 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class LinearStress : public Plato::SimplexMechanics<SpaceDim>
{
private:
    using Plato::SimplexMechanics<SpaceDim>::mNumVoigtTerms;               /*!< number of stress/strain terms */

    const Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;  /*!< material stiffness matrix */
    Omega_h::Vector<mNumVoigtTerms> mReferenceStrain;                      /*!< reference strain tensor */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
    **********************************************************************************/
    LinearStress(const Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> aCellStiffness) :
            mCellStiffness(aCellStiffness)
    {
        for(Plato::OrdinalType tIndex = 0; tIndex < mNumVoigtTerms; tIndex++)
        {
            mReferenceStrain(tIndex) = 0.0;
        }
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMaterialModel material model interface
    **********************************************************************************/
    LinearStress(const Teuchos::RCP<Plato::LinearElasticMaterial<SpaceDim>> aMaterialModel) :
            mCellStiffness(aMaterialModel->getStiffnessMatrix()),
            mReferenceStrain(aMaterialModel->getReferenceStrain())
    {
    }

    /******************************************************************************//**
     * \brief Compute Cauchy stress tensor
     * \param [in]  aCellOrdinal element ordinal
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
    **********************************************************************************/
    template<typename StressScalarType, typename StrainScalarType>
    DEVICE_TYPE inline void operator()(Plato::OrdinalType aCellOrdinal,
                                       Kokkos::View<StressScalarType**, Plato::Layout, Plato::MemSpace> const& aCauchyStress,
                                       Kokkos::View<StrainScalarType**, Plato::Layout, Plato::MemSpace> const& aSmallStrain) const
    {

        // compute stress
        //
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            aCauchyStress(aCellOrdinal, tVoigtIndex_I) = 0.0;
            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
                aCauchyStress(aCellOrdinal, tVoigtIndex_I) += (aSmallStrain(aCellOrdinal, tVoigtIndex_J)
                        - mReferenceStrain(tVoigtIndex_J)) * mCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
    }

    /******************************************************************************//**
     * \brief Compute Cauchy stress tensor
     * \param [in]  aCellOrdinal element ordinal
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aStrainInc Incremental strain tensor
     * \param [in]  aPrevStrain Reference strain tensor
    **********************************************************************************/
    template<typename StressScalarType, typename StrainScalarType, typename PrevStrainScalarType>
    DEVICE_TYPE inline void
    operator()(
        Plato::OrdinalType aCellOrdinal,
        Kokkos::View<StressScalarType**,     Plato::Layout, Plato::MemSpace> const& aCauchyStress,
        Kokkos::View<StrainScalarType**,     Plato::Layout, Plato::MemSpace> const& aStrainInc,
        Kokkos::View<PrevStrainScalarType**, Plato::Layout, Plato::MemSpace> const& aPrevStrain
    ) const
    {

        // compute stress
        //
        for(Plato::OrdinalType tVoigtIndex_I = 0; tVoigtIndex_I < mNumVoigtTerms; tVoigtIndex_I++)
        {
            aCauchyStress(aCellOrdinal, tVoigtIndex_I) = 0.0;
            for(Plato::OrdinalType tVoigtIndex_J = 0; tVoigtIndex_J < mNumVoigtTerms; tVoigtIndex_J++)
            {
                aCauchyStress(aCellOrdinal, tVoigtIndex_I) +=
                    (aStrainInc(aCellOrdinal, tVoigtIndex_J) - mReferenceStrain(tVoigtIndex_J) + aPrevStrain(aCellOrdinal, tVoigtIndex_J))
                  * mCellStiffness(tVoigtIndex_I, tVoigtIndex_J);
            }
        }
    }

};
// class LinearStress

}// namespace Plato
#endif

#ifndef PLATO_RAYLEIGH_STRESS_HPP
#define PLATO_RAYLEIGH_STRESS_HPP

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
class RayleighStress : public Plato::SimplexMechanics<SpaceDim>
{
private:
    using Plato::SimplexMechanics<SpaceDim>::mNumVoigtTerms;               /*!< number of stress/strain terms */

    const Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;  /*!< material stiffness matrix */
    Omega_h::Vector<mNumVoigtTerms> mReferenceStrain;                      /*!< reference strain tensor */
    Plato::Scalar mRayleighB;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
    **********************************************************************************/
    RayleighStress(const Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> aCellStiffness) :
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
    RayleighStress(const Teuchos::RCP<Plato::LinearElasticMaterial<SpaceDim>> aMaterialModel) :
            mCellStiffness(aMaterialModel->getStiffnessMatrix()),
            mReferenceStrain(aMaterialModel->getReferenceStrain()),
            mRayleighB(aMaterialModel->getRayleighB())
    {
    }

    /******************************************************************************//**
     * \brief Compute Cauchy stress tensor
     * \param [in]  aCellOrdinal element ordinal
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
     * \param [in]  aVelGrad Velocity gradient tensor
    **********************************************************************************/
    template<typename StressScalarType, typename StrainScalarType, typename VelGradScalarType>
    DEVICE_TYPE inline void
    operator()( int cellOrdinal,
                Kokkos::View<StressScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& aCauchyStress,
                Kokkos::View<StrainScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& aSmallStrain,
                Kokkos::View<VelGradScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& aVelGrad) const {

      // compute stress
      //
      for( int iVoigt=0; iVoigt<mNumVoigtTerms; iVoigt++){
        aCauchyStress(cellOrdinal,iVoigt) = 0.0;
        for( int jVoigt=0; jVoigt<mNumVoigtTerms; jVoigt++){
          aCauchyStress(cellOrdinal,iVoigt) += (aSmallStrain(cellOrdinal,jVoigt)-mReferenceStrain(jVoigt))*mCellStiffness(iVoigt, jVoigt)
                                     +  aVelGrad(cellOrdinal,jVoigt)*mCellStiffness(iVoigt, jVoigt)*mRayleighB;
        }
      }
    }
};
// class RayleighStress

}// namespace Plato
#endif

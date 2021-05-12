#ifndef PLATO_ABSTRACT_LINEAR_STRESS_HPP
#define PLATO_ABSTRACT_LINEAR_STRESS_HPP

#include "ExpInstMacros.hpp"
#include "LinearElasticMaterial.hpp"
#include "SimplexFadTypes.hpp"
#include "SimplexMechanics.hpp"

#include <Omega_h_matrix.hpp>

namespace Plato
{

/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}
 */
/******************************************************************************/
template< typename EvaluationType, typename SimplexPhysics >
class AbstractLinearStress :
    public Plato::SimplexMechanics<EvaluationType::SpatialDim>
{
protected:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using StateT  = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    using StrainT = typename Plato::fad_type_t<SimplexPhysics, StateT, ConfigT>; /*!< strain variables automatic differentiation type */

    using Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of stress/strain terms */

    const Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness; /*!< material stiffness matrix */

    Omega_h::Vector<mNumVoigtTerms> mReferenceStrain; /*!< reference strain tensor */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
    **********************************************************************************/
    AbstractLinearStress(const Omega_h::Matrix<mNumVoigtTerms,
                                               mNumVoigtTerms> aCellStiffness) :
      mCellStiffness(aCellStiffness)
    {
        for(Plato::OrdinalType tIndex = 0; tIndex < mNumVoigtTerms; tIndex++)
        {
            mReferenceStrain(tIndex) = 0.0;
        }
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] const aMaterialModel material model interface
    **********************************************************************************/
    AbstractLinearStress(const Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> aMaterialModel) :
      mCellStiffness  (aMaterialModel->getStiffnessMatrix()),
      mReferenceStrain(aMaterialModel->getReferenceStrain())
    {
    }

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
    **********************************************************************************/
    virtual void
    operator()(Plato::ScalarMultiVectorT<ResultT> const& aCauchyStress,
               Plato::ScalarMultiVectorT<StrainT> const& aSmallStrain) const = 0;

};
// class AbstractLinearStress

}// namespace Plato
#endif

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC2(Plato::AbstractLinearStress  , Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC2(Plato::AbstractLinearStress  , Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC2(Plato::AbstractLinearStress  , Plato::SimplexMechanics, 3)
#endif


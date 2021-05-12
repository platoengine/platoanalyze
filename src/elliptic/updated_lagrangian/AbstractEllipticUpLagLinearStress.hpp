#ifndef PLATO_ABSTRACT_ELLIPTIC_UPDATED_LAGRANGIAN_LINEAR_STRESS_HPP
#define PLATO_ABSTRACT_ELLIPTIC_UPDATED_LAGRANGIAN_LINEAR_STRESS_HPP

#include "elliptic/updated_lagrangian/ExpInstMacros.hpp"
#include "LinearElasticMaterial.hpp"
#include "EllipticUpLagSimplexFadTypes.hpp"
#include "SimplexMechanics.hpp"

#include <Omega_h_matrix.hpp>

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}
 */
/******************************************************************************/
template< typename EvaluationType, typename SimplexPhysics >
class AbstractEllipticUpLagLinearStress :
    public Plato::Elliptic::UpdatedLagrangian::SimplexMechanics<EvaluationType::SpatialDim>
{
protected:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using GlobalStateT = typename EvaluationType::GlobalStateScalarType; /*!< state variables automatic differentiation type */
    using LocalStateT  = typename EvaluationType::LocalStateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigT      = typename EvaluationType::ConfigScalarType;      /*!< configuration variables automatic differentiation type */
    using ResultT      = typename EvaluationType::ResultScalarType;      /*!< result variables automatic differentiation type */

    using StrainT = typename Plato::fad_type_t<SimplexPhysics, GlobalStateT, ConfigT>; /*!< strain variables automatic differentiation type */

    using Plato::Elliptic::UpdatedLagrangian::SimplexMechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of stress/strain terms */

    const Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;  /*!< material stiffness matrix */

    Omega_h::Vector<mNumVoigtTerms> mReferenceStrain;                      /*!< reference strain tensor */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
    **********************************************************************************/
    AbstractEllipticUpLagLinearStress(const Omega_h::Matrix<mNumVoigtTerms,
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
    AbstractEllipticUpLagLinearStress(const Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> aMaterialModel) :
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
    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aStrainInc Incremental strain tensor
     * \param [in]  aPrevStrain Reference strain tensor
    **********************************************************************************/
    virtual void
    operator()(Plato::ScalarMultiVectorT<ResultT>     const& aCauchyStress,
               Plato::ScalarMultiVectorT<StrainT>     const& aStrainInc,
               Plato::ScalarMultiVectorT<LocalStateT> const& aPrevStrain) const = 0;
};
// class AbstractEllipticUpLagLinearStress

}// namespace UpdatedLagrangian

}// namespace Elliptic

}// namespace Plato
#endif

#ifdef PLATOANALYZE_1D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC2(Plato::Elliptic::UpdatedLagrangian::AbstractEllipticUpLagLinearStress, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC2(Plato::Elliptic::UpdatedLagrangian::AbstractEllipticUpLagLinearStress, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC2(Plato::Elliptic::UpdatedLagrangian::AbstractEllipticUpLagLinearStress, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 3)
#endif


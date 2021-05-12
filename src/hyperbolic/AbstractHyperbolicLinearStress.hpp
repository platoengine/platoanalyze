#ifndef PLATO_HYPERBOLIC_ABSTRACT_LINEAR_STRESS_HPP
#define PLATO_HYPERBOLIC_ABSTRACT_LINEAR_STRESS_HPP

#include "LinearElasticMaterial.hpp"
#include "SimplexMechanics.hpp"

#include "hyperbolic/HyperbolicExpInstMacros.hpp"
#include "hyperbolic/HyperbolicSimplexFadTypes.hpp"

#include <Omega_h_matrix.hpp>

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************/
/*! Stress functor.

 given a strain, compute the stress.
 stress tensor in Voigt notation = {s_xx, s_yy, s_zz, s_yz, s_xz, s_xy}
 */
/******************************************************************************/
template< typename EvaluationType, typename SimplexPhysics >
class AbstractHyperbolicLinearStress :
    public Plato::SimplexMechanics<EvaluationType::SpatialDim>
{
protected:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using StateT    = typename EvaluationType::StateScalarType;    /*!< state variables automatic differentiation type */
    using StateDotT = typename EvaluationType::StateDotScalarType; /*!< state dot variables automatic differentiation type */
    using ConfigT   = typename EvaluationType::ConfigScalarType;   /*!< configuration variables automatic differentiation type */
    using ResultT   = typename EvaluationType::ResultScalarType;   /*!< result variables automatic differentiation type */

    using StrainT  = typename Plato::fad_type_t<SimplexPhysics, StateT,    ConfigT>; /*!<   strain variables automatic differentiation type */
    using VelGradT = typename Plato::fad_type_t<SimplexPhysics, StateDotT, ConfigT>; /*!< vel grad variables automatic differentiation type */

    using Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of stress/strain terms */

    Plato::Scalar mRayleighB;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aCellStiffness material element stiffness matrix
    **********************************************************************************/
    AbstractHyperbolicLinearStress(const Omega_h::Matrix<mNumVoigtTerms,
                                   mNumVoigtTerms> aCellStiffness)
    {
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] const aMaterialModel material model interface
    **********************************************************************************/
    AbstractHyperbolicLinearStress(const Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> aMaterialModel) :
        mRayleighB(aMaterialModel->getRayleighB())
    {
    }

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor
     * \param [out] aCauchyStress Cauchy stress tensor
     * \param [in]  aSmallStrain Infinitesimal strain tensor
     * \param [in]  aVelGrad Velocity gradient tensor
    **********************************************************************************/
    virtual void
    operator()(Plato::ScalarMultiVectorT<ResultT > const& aCauchyStress,
               Plato::ScalarMultiVectorT<StrainT > const& aSmallStrain,
               Plato::ScalarMultiVectorT<VelGradT> const& aVelGrad) const = 0;
};
// class AbstractHyperbolicLinearStress

}// namespace Hyperbolic

}// namespace Plato

#endif

#ifdef PLATOANALYZE_1D
PLATO_HYPERBOLIC_EXPL_DEC2(Plato::Hyperbolic::AbstractHyperbolicLinearStress  , Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_HYPERBOLIC_EXPL_DEC2(Plato::Hyperbolic::AbstractHyperbolicLinearStress  , Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_HYPERBOLIC_EXPL_DEC2(Plato::Hyperbolic::AbstractHyperbolicLinearStress  , Plato::SimplexMechanics, 3)
#endif


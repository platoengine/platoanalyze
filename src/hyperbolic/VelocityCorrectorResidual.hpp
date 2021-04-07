/*
 * VelocityCorrectorResidual.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include "BLAS2.hpp"
#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "NaturalBCs.hpp"
#include "FluidsUtils.hpp"
#include "UtilsTeuchos.hpp"
#include "SpatialModel.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/BrinkmanForces.hpp"
#include "hyperbolic/ThermalBuoyancy.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/MassConservationUtils.hpp"
#include "hyperbolic/AbstractVectorFunction.hpp"
#include "hyperbolic/MomentumConservationUtils.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \class VelocityCorrectorResidual
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT forward automatic differentiation evaluation type
 *
 * \brief Class responsible for the evaluation of the momentum corrector residual,
 *   defined as
 *
 * \f[
 *   \int_{\Omega_e} w_i u_i^{n+1} d\Omega_e = \int_{\Omega_e} w_i u_i^{\ast} d\Omega_e
 *     - \Delta{t}\int_{\Omega_e} w_i\frac{\partial p^{n+\theta_p}}{\partial x_i}
 *     + \frac{\Delta{t}^2}{2}\int_{\Omega_e}(\frac{\partial w_i}{\partial x_k} u_k^n)
 *       \frac{\partial p^n}{\partial x_i} d\Omega_e.
 * \f]
 *
 * where \f$ u_i \f$ is the i-th component of the velocity field, \f$ u_i^{\ast} \f$
 * is the i-th component of the velocity predictor field, \f$ p^n \f$ is the
 * pressure field, \f$ x_i \f$ is the i-th coordinate and \f$ \theta_p \f$ is
 * the pressure artificial damping parameter.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class VelocityCorrectorResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode    = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell    = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */
    static constexpr auto mNumNodesPerCell   = PhysicsT::SimplexT::mNumNodesPerCell; /*!< number of nodes per cell */
    static constexpr auto mNumSpatialDims    = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */

    // set local ad types
    using ResultT    = typename EvaluationT::ResultScalarType; /*!< result/output FAD type */
    using ConfigT    = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using CurVelT    = typename EvaluationT::CurrentMomentumScalarType; /*!< current velocity FAD type */
    using PrevVelT   = typename EvaluationT::PreviousMomentumScalarType; /*!< previous velocity FAD type */
    using PredVelT   = typename EvaluationT::MomentumPredictorScalarType; /*!< predicted velocity FAD type */
    using CurPressT  = typename EvaluationT::CurrentMassScalarType; /*!< current pressure FAD type */
    using PrevPressT = typename EvaluationT::PreviousMassScalarType; /*!< previous pressure FAD type */

    /*!< pressure gradient FAD type */
    using PressGradT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurPressT, PrevPressT, ConfigT>;

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< integration rule */

    Plato::Scalar mPressureTheta = 1.0; /*!< artificial pressure damping */
    Plato::Scalar mViscosityTheta = 1.0; /*!< artificial viscosity damping */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output data metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    VelocityCorrectorResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setAritificalPressureDamping(aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~VelocityCorrectorResidual(){}

    /***************************************************************************//**
     * \brief Evaluate Brinkman forces, including stabilized forces if enabled.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set local data structures
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarMultiVectorT<PressGradT> tPressGradGP("pressure gradient", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CurVelT> tCurVelGP("current velocity at Gauss points", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity at Gauss points", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PredVelT> tPredVelGP("predicted velocity at Gauss points", tNumCells, mNumSpatialDims);

        // set input state worksets
        auto tConfigWS    = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurVelWS    = Plato::metadata<Plato::ScalarMultiVectorT<CurVelT>>(aWorkSets.get("current velocity"));
        auto tPrevVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tPredVelWS   = Plato::metadata<Plato::ScalarMultiVectorT<PredVelT>>(aWorkSets.get("current predictor"));
        auto tCurPressWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurPressT>>(aWorkSets.get("current pressure"));
        auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevPressT>>(aWorkSets.get("previous pressure"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // transfer member data to device
        auto tPressureTheta = mPressureTheta;
        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. add previous pressure gradient to residual, i.e. R += Delta{t} G(p_n + \theta\Delta{p})
            Plato::Fluids::calculate_pressure_gradient<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tPressureTheta, tGradient, tCurPressWS, tPrevPressWS, tPressGradGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPressGradGP, aResultWS);
            Plato::blas2::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);

            // 2. add current delta inertial force to residual, i.e. R += M(u_{n+1} - u_n)
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tCurVelGP, aResultWS);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPrevVelGP, aResultWS, -1.0);

            // 3. add delta predicted inertial force to residual, i.e. R -= M(\bar{u} - u_n)
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredVelWS, tPredVelGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPredVelGP, aResultWS, -1.0);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPrevVelGP, aResultWS);
        }, "calculate corrected velocity residual");
    }

    /***************************************************************************//**
     * \brief Evaluate non-prescribed boundary conditions.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* boundary integral equals zero */ }

    /***************************************************************************//**
     * \brief Evaluate prescribed boundary conditions.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return; /* prescribed force integral equals zero */ }

private:
    /***************************************************************************//**
     * \brief Set artificial pressure damping parameter.
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    void setAritificalPressureDamping(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mPressureTheta = tTimeIntegration.get<Plato::Scalar>("Pressure Damping", 1.0);
        }
    }
};
// class VelocityCorrectorResidual

}
// namespace Fluids

}
// namespace Plato

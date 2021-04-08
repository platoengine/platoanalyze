/*
 * VelocityPredictorResidual.hpp
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
#include "ExpInstMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/BrinkmanForces.hpp"
#include "hyperbolic/ThermalBuoyancy.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/AbstractVectorFunction.hpp"
#include "hyperbolic/MomentumConservationUtils.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \class VelocityPredictorResidual
 *
 * \tparam PhysicsT    fluid flow physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \brief Evaluate momentum predictor residual, defined as
 *
 * \f[
 *   \mathcal{R}^n_i(w^h_i) = I^n_i(w^h_i) - F^n_i(w^h_i) - B^n_i(w^h_i) - S_i^n(w^h_i) - E_i^n(w^h_i) = 0.
 * \f]
 *
 * Inertial Forces:
 *
 * \f[
 *   I^n_i(w^h_i) =
 *     \int_{\Omega}w_i^h\left(\frac{\bar{u}^{\ast}_i - \bar{u}_i^{n}}{\Delta\bar{t}}\right)d\Omega
 * \f]
 *
 * Internal Forces:
 *
 * \f[
 *   F^n_i(w^h_i) =
 *     - \int_{\Omega} w_i^h\left( \bar{u}_j^n\frac{\partial \bar{u}_i^n}{\partial\bar{x}_j} \right) d\Omega
 *     - \int_{\Omega} \frac{\partial w_i^h}{\partial\bar{x}_j}\bar\tau_{ij}^n\,d\Omega
 *     + \int_\Omega w_i^h\left(Gr_i Pr^2\bar{T}^n\right)\,d\Omega
 * \f]
 *
 * Stabilizing Forces:
 *
 * \f[
 *   S_i^n(w^h_i) =
 *     \frac{\Delta\bar{t}}{2}\left[ \int_{\Omega} \frac{\partial w_i^h}{\partial\bar{x}_k}
 *     \left( \bar{u}^n_k \hat{F}^n_{\bar{u}_i} \right) d\Omega \right]
 * \f]
 *
 * where
 *
 * \f[
 *   \hat{F}^n_{\bar{u}_i} = -\bar{u}_j^n \frac{\partial\bar{u}_i^n}{\partial \bar{x}_j} + Gr_i Pr^2\bar{T}^n
 * \f]
 *
 * External Forces:
 *
 * \f[
 *   E_i^n(w^h_i) = \int_{\Gamma-\Gamma_t}w_i^h\bar{\tau}^n_{ij}n_j\,d\Gamma
 * \f]
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class VelocityPredictorResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerCell    = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumTempDofsPerCell   = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell;   /*!< number of configuration degrees of freedom per cell */

    // set local ad types
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using ControlT  = typename EvaluationT::ControlScalarType; /*!< control FAD type */
    using PrevVelT  = typename EvaluationT::PreviousMomentumScalarType; /*!< previous velocity FAD type */
    using PredVelT  = typename EvaluationT::MomentumPredictorScalarType; /*!< predicted velocity FAD type */
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType; /*!< previous temperature FAD type */

    using AdvectionT  = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>; /*!< advection force FAD type */
    using PredStrainT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PredVelT, ConfigT>; /*!< predicted strain rate FAD type */
    using PrevStrainT = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, PrevVelT, ConfigT>; /*!< previous strain rate FAD type */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature rule evaluator */

    // set right hand side force evaluators
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mPrescribedBCs; /*!< prescribed boundary conditions, e.g. tractions */
    std::shared_ptr<Plato::Fluids::BrinkmanForces<PhysicsT,EvaluationT>> mBrinkmanForces; /*!< Brinkman force evaluator */
    std::shared_ptr<Plato::Fluids::ThermalBuoyancy<PhysicsT,EvaluationT>> mThermalBuoyancy; /*!< thermal buoyancy force evaluator */

    // set member scalar data
    Plato::Scalar mTheta = 1.0; /*!< artificial viscous damping */
    Plato::Scalar mViscocity = 1.0; /*!< dimensionless viscocity constant */
    Plato::Scalar mStabilization = 0.0; /*!< stabilization scalar multiplier */

    bool mCalculateBrinkmanForces = false; /*!< indicator to determine if Brinkman forces will be considered in calculations */
    bool mCalculateThermalBuoyancyForces = false; /*!< indicator to determine if thermal buoyancy forces will be considered in calculations */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  holds mesh and entity sets (e.g. node and side sets)
     *   metadata for this spatial domain (e.g. element block)
     * \param [in] aDataMap holds output metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    VelocityPredictorResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->initialize(aDomain, aDataMap, aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~VelocityPredictorResidual(){}

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate predictor residual.
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

        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);

        Plato::ScalarArray3DT<PredStrainT> tPredStrainRate("cell strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);
        Plato::ScalarArray3DT<PrevStrainT> tPrevStrainRate("cell strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);

        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<PredVelT> tPredVelGP("predicted velocity", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<AdvectionT> tAdvection("advected force", tNumCells, mNumSpatialDims);

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set input state worksets
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPredVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PredVelT>>(aWorkSets.get("current predictor"));
        auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member data to device
        auto tTheta = mTheta;
        auto tViscocity = mViscocity;
        auto tStabilization = mStabilization;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. add predicted viscous force to residual, i.e. R += \theta K \bar{u}
            Plato::Fluids::strain_rate<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tPredVelWS, tGradient, tPredStrainRate);
            Plato::Fluids::integrate_viscous_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tViscocity, tCellVolume, tGradient, tPredStrainRate, aResultWS, tTheta);

            // 2. add previous viscous force to residual, i.e. R -= (\theta-1)K u_n
            auto tMultiplier = (tTheta - static_cast<Plato::Scalar>(1.0));
            Plato::Fluids::strain_rate<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tPrevVelWS, tGradient, tPrevStrainRate);
            Plato::Fluids::integrate_viscous_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tViscocity, tCellVolume, tGradient, tPrevStrainRate, aResultWS, -tMultiplier);

            // 3. add advection force to residual, i.e. R += C u_n
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            Plato::Fluids::calculate_advected_momentum_forces<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tGradient, tPrevVelWS, tPrevVelGP, tAdvection);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tAdvection, aResultWS);

            // 4. apply time step, i.e. \Delta{t}( \theta K\bar{u} + C u_n - (\theta-1)K u_n )
            Plato::blas2::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), aResultWS);

            // 5. add predicted inertial force to residual, i.e. R += M\bar{u}
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPredVelWS, tPredVelGP);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPredVelGP, aResultWS);

            // 6. add previous inertial force to residual, i.e. R -= M u_n
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tPrevVelGP, aResultWS, -1.0);

            // 7. add stabilizing convective term to residual. i.e. R += \frac{\Delta{t}^2}{2}K_{u}u^{n}
            tMultiplier = tStabilization * static_cast<Plato::Scalar>(0.5) * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::Fluids::integrate_stabilizing_vector_force<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tPrevVelGP, tAdvection, aResultWS, tMultiplier);
        }, "quasi-implicit predicted velocity residual");

        if(mCalculateThermalBuoyancyForces)
        {
            mThermalBuoyancy->evaluate(aWorkSets, aResultWS);
        }

        if(mCalculateBrinkmanForces)
        {
            mBrinkmanForces->evaluate(aWorkSets, aResultWS);
        }
    }

   /***************************************************************************//**
    * \fn void evaluateBoundary
    * \brief Evaluate non-prescribed boundary forces.
    * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
    * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
    * \param [in/out] aResultWS result/output workset
    ******************************************************************************/
   void evaluateBoundary
   (const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResult)
   const override
   { return; }

   /***************************************************************************//**
    * \fn void evaluatePrescribed
    * \brief Evaluate prescribed boundary forces.
    * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
    * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
    * \param [in/out] aResultWS result/output workset
    ******************************************************************************/
   void evaluatePrescribed
   (const Plato::SpatialModel & aSpatialModel,
    const Plato::WorkSets & aWorkSets,
    Plato::ScalarMultiVectorT<ResultT> & aResultWS)
   const override
   {
       if( mPrescribedBCs != nullptr )
       {
           // set input worksets
           auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
           auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
           auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));

           // 1. add prescribed traction force to residual
           auto tNumCells = aResultWS.extent(0);
           Plato::ScalarMultiVectorT<ResultT> tTractionWS("traction forces", tNumCells, mNumDofsPerCell);
           mPrescribedBCs->get( aSpatialModel, tPrevVelWS, tControlWS, tConfigWS, tTractionWS);

           // 2. apply time step to traction force
           auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));
           Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
           {
               Plato::blas2::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), tTractionWS);
               Plato::blas2::update<mNumDofsPerCell>(aCellOrdinal, -1.0, tTractionWS, 1.0, aResultWS);
           }, "traction force");
       }
   }

private:
   /***************************************************************************//**
    * \fn void initialize
    * \brief Initialize member data.
    * \param [in] aDomain  spatial domain metadata
    * \param [in] aDataMap output data metadata
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
   void initialize
   (const Plato::SpatialDomain & aDomain,
    Plato::DataMap             & aDataMap,
    Teuchos::ParameterList     & aInputs)
   {
       this->setAritificalDamping(aInputs);
       this->setNaturalBoundaryConditions(aInputs);
       mViscocity = Plato::Fluids::calculate_viscosity_constant(aInputs);
       mStabilization = Plato::Fluids::stabilization_constant("Momentum Conservation", aInputs);

       this->setBrinkmanForces(aDomain, aDataMap, aInputs);
       this->setThermalBuoyancyForces(aDomain, aDataMap, aInputs);
   }

   /***************************************************************************//**
    * \fn void setBrinkmanForces
    * \brief Set Brinkman forces if enabled. The Brinkman forces are used to model
    *   a fictitious solid material within a fluid domain. These froces are only used
    *   in density-based topology optimization problems.
    * \param [in] aDomain  spatial domain metadata
    * \param [in] aDataMap output data metadata
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
   void setBrinkmanForces
   (const Plato::SpatialDomain & aDomain,
    Plato::DataMap             & aDataMap,
    Teuchos::ParameterList     & aInputs)
   {
       mCalculateBrinkmanForces = Plato::Fluids::calculate_brinkman_forces(aInputs);
       if(mCalculateBrinkmanForces)
       {
           mBrinkmanForces =
               std::make_shared<Plato::Fluids::BrinkmanForces<PhysicsT,EvaluationT>>(aDomain, aDataMap, aInputs);
       }
   }

   /***************************************************************************//**
    * \fn void setThermalBuoyancyForces
    * \brief Set thermal buoyancy forces if enabled.
    * \param [in] aDomain  spatial domain metadata
    * \param [in] aDataMap output data metadata
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
   void setThermalBuoyancyForces
   (const Plato::SpatialDomain & aDomain,
    Plato::DataMap             & aDataMap,
    Teuchos::ParameterList     & aInputs)
   {
       mCalculateThermalBuoyancyForces = Plato::Fluids::calculate_heat_transfer(aInputs);
       if(mCalculateThermalBuoyancyForces)
       {
           mThermalBuoyancy =
               std::make_shared<Plato::Fluids::ThermalBuoyancy<PhysicsT,EvaluationT>>(aDomain, aDataMap, aInputs);
       }
   }

   /***************************************************************************//**
    * \fn void setAritificalDamping
    * \brief Set artificial viscous damping. This parameter is related to the time
    *   integration scheme.
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
   void setAritificalDamping(Teuchos::ParameterList& aInputs)
   {
       if(aInputs.isSublist("Time Integration"))
       {
           auto tTimeIntegration = aInputs.sublist("Time Integration");
           mTheta = tTimeIntegration.get<Plato::Scalar>("Viscosity Damping", 1.0);
       }
   }

   /***************************************************************************//**
    * \fn void setNaturalBoundaryConditions
    * \brief Set natural boundary conditions if defined by the user.
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
   void setNaturalBoundaryConditions(Teuchos::ParameterList& aInputs)
   {
       if(aInputs.isSublist("Momentum Natural Boundary Conditions"))
       {
           auto tInputsNaturalBCs = aInputs.sublist("Momentum Natural Boundary Conditions");
           mPrescribedBCs = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tInputsNaturalBCs);
       }
   }
};
// class VelocityPredictorResidual

}
// namespace Fluids

}
// namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::VelocityPredictorResidual, Plato::MomentumConservation, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::VelocityPredictorResidual, Plato::MomentumConservation, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::VelocityPredictorResidual, Plato::MomentumConservation, Plato::SimplexFluids, 3, 1)
#endif

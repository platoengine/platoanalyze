/*
 * ThermalBuoyancy.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "FluidsUtils.hpp"
#include "SpatialModel.hpp"
#include "ExpInstMacros.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/MomentumConservationUtils.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam PhysicsT    Physics Type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) Evaluation Type
 *
 * \class ThermalBuoyancy
 *
 * \brief Class responsible for evaluating thermal buoyancy forces, including
 *   stabilization forces associated with the thermal buoyancy forces.
 *
 * Thermal Buoyancy Force:
 *   \[ \Delta{t}*Bu*Gr_i \int_{\Omega_e} w T^n d\Omega_e \]
 *
 * Stabilized Buoyancy Force:
 *   \[ \frac{\Delta{t}^2}{2}*Bu*Gr_i \int_{\Omega_e} (\frac{\partial w}{\partial x_k} u_k^n ) T^n d\Omega_e \]
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class ThermalBuoyancy
{
private:
    static constexpr auto mNumSpatialDims       = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell      = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerNode    = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode   = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */

    // set local ad types
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD type */
    using PrevVelT  = typename EvaluationT::PreviousMomentumScalarType; /*!< previous velocity FAD type */
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType; /*!< previous temperature FAD type */

    Plato::Scalar mStabilization = 0.0; /*!< stabilization constant */
    Plato::Scalar mBuoyancyConst = 0.0; /*!< dimensionless buoyancy constant */
    Plato::Scalar mBuoyancyDamping = 1.0; /*!< artificial buoyancy damping */
    Plato::ScalarVector mNaturalConvectionNum; /*!< dimensionless natural convection number (either Rayleigh or Grashof - depends on user's input) */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialDomain& mSpatialDomain; /*!< Plato spatial model */
    Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims> mCubatureRule; /*!< cubature rule evaluator */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output data map
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    ThermalBuoyancy
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mDataMap(aDataMap),
         mSpatialDomain(aDomain),
         mCubatureRule(Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>())
    {
        this->setAritificalDamping(aInputs);
        mBuoyancyConst = Plato::Fluids::calculate_buoyancy_constant(aInputs);
        mStabilization = Plato::Fluids::stabilization_constant("Momentum Conservation", aInputs);
        mNaturalConvectionNum = Plato::Fluids::parse_natural_convection_number<mNumSpatialDims>(aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    ~ThermalBuoyancy(){}

    /***************************************************************************//**
     * \brief Evaluate thermal buoyancy forces, including stabilized forces if enabled.
     * \param [in] aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [in/out] aResultWS result/output workset
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS) const
    {
        auto tNumCells = mSpatialDomain.numCells();
        if( tNumCells != static_cast<Plato::OrdinalType>(aResultWS.extent(0)) )
        {
            THROWERR(std::string("Number of elements mismatch. Spatial domain and output/result workset ")
                + "have different number of cells. " + "Spatial domain has '" + std::to_string(tNumCells)
                + "' elements and output workset has '" + std::to_string(aResultWS.extent(0)) + "' elements.")
        }

        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumTempDofsPerNode> tIntrplScalarField;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0/*offset*/, mNumSpatialDims> tIntrplVectorField;

        // set temporary worksets
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("cell gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarVectorT<PrevTempT> tPrevTempGP("previous temperature at Gauss point", tNumCells);
        Plato::ScalarMultiVectorT<PrevVelT> tPrevVelGP("previous velocity", tNumCells, mNumSpatialDims);
        Plato::ScalarMultiVectorT<ResultT> tThermalBuoyancy("thermal buoyancy", tNumCells, mNumSpatialDims);

        // set input worksets
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevVelT>>(aWorkSets.get("previous velocity"));
        auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));
        auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));

        // transfer member data to device
        auto tStabilization = mStabilization;
        auto tBuoyancyConst = mBuoyancyConst;
        auto tBuoyancyDamping = mBuoyancyDamping;
        auto tNaturalConvectionNum = mNaturalConvectionNum;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tCubWeight;

            // 1. add previous buoyancy force to residual, i.e. R += (\Delta{t}*Bu*Gr_i) M T_n, where Bu is the buoyancy constant
            auto tMultiplier = tBuoyancyDamping * tCriticalTimeStep(0);
            tIntrplScalarField(aCellOrdinal, tBasisFunctions, tPrevTempWS, tPrevTempGP);
            Plato::Fluids::calculate_natural_convective_forces<mNumSpatialDims>
                (aCellOrdinal, tBuoyancyConst, tNaturalConvectionNum, tPrevTempGP, tThermalBuoyancy);
            Plato::Fluids::integrate_vector_field<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tBasisFunctions, tCellVolume, tThermalBuoyancy, aResultWS, -tMultiplier);

            // 2. add stabilizing buoyancy force to residual. i.e. R += \frac{\Delta{t}^2}{2} Bu*Gr_i) M T_n
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
            tMultiplier = tStabilization * static_cast<Plato::Scalar>(0.5) * tBuoyancyDamping * tCriticalTimeStep(0) * tCriticalTimeStep(0);
            Plato::Fluids::integrate_stabilizing_vector_force<mNumNodesPerCell, mNumSpatialDims>
                (aCellOrdinal, tCellVolume, tGradient, tPrevVelGP, tThermalBuoyancy, aResultWS, -tMultiplier);
        }, "add contribution from thermal buoyancy forces to residual");
    }

private:
    /***************************************************************************//**
     * \fn void setAritificalDamping
     * \brief Set artificial buoyancy damping parameter.
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    void setAritificalDamping(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Hyperbolic") == false)
        {
            THROWERR("'Hyperbolic' Parameter List is not defined.")
        }
        auto tHyperbolic = aInputs.sublist("Hyperbolic");
        if(tHyperbolic.isSublist("Momentum Conservation"))
        {
            auto tMomentumConservation = tHyperbolic.sublist("Momentum Conservation");
            mBuoyancyDamping = tMomentumConservation.get<Plato::Scalar>("Buoyancy Damping", 1.0);
        }
    }
};
// class ThermalBuoyancy

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::ThermalBuoyancy, Plato::MomentumConservation, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::ThermalBuoyancy, Plato::MomentumConservation, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::ThermalBuoyancy, Plato::MomentumConservation, Plato::SimplexFluids, 3, 1)
#endif

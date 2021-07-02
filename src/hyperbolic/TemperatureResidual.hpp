/*
 * TemperatureResidual.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include "BLAS2.hpp"
#include "MetaData.hpp"
#include "WorkSets.hpp"
#include "NaturalBCs.hpp"
#include "SpatialModel.hpp"
#include "ExpInstMacros.hpp"
#include "AbstractVolumeIntegrand.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/FluidsThermalSources.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/AbstractVectorFunction.hpp"
#include "hyperbolic/FluidsVolumeIntegrandFactory.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \class TemperatureResidual
 *
 * \tparam PhysicsT    Physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \brief Class responsible for the evaluation of the energy residual.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class TemperatureResidual : public Plato::Fluids::AbstractVectorFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */
    static constexpr auto mNumSpatialDims = PhysicsT::SimplexT::mNumSpatialDims; /*!< number of spatial dimensions */

    // set local ad type
    using ResultT   = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */
    using ConfigT   = typename EvaluationT::ConfigScalarType; /*!< configuration FAD evaluation type */
    using ControlT  = typename EvaluationT::ControlScalarType; /*!< control FAD evaluation type */
    using PrevTempT = typename EvaluationT::PreviousEnergyScalarType;  /*!< previous energy FAD evaluation type */

    const Plato::SpatialDomain& mSpatialDomain; /*!< spatial domain metadata */
    Plato::Fluids::ThermalSources<PhysicsT, EvaluationT> mThermalSources; /*!< interface to thermal source evaluator */
    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mHeatFlux; /*!< natural boundary condition evaluator */
    std::shared_ptr<Plato::AbstractVolumeIntegrand<PhysicsT,EvaluationT>> mInternalForces; /*!< internal force evaluator */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aDomain  spatial domain metadata
     * \param [in] aDataMap output database metadata
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    TemperatureResidual
    (const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mSpatialDomain(aDomain)
    {
        // initialize neumann/natural force evaluator
        this->initializeNaturalBoundaryConditions(aInputs);

        // initializr internal force evaluator 
        Plato::Fluids::VolumeIntegrandFactory tFactory;
        mInternalForces = tFactory.createInternalThermalForces<PhysicsT,EvaluationT>(aDomain,aDataMap,aInputs);

        // initialize thermal source evaluator
        mThermalSources.initializeStabilizedThermalSources(aDomain, aDataMap, aInputs);
    }

    /***************************************************************************//**
     * \brief Destructor
     ******************************************************************************/
    virtual ~TemperatureResidual(){}

    /***************************************************************************//**
     * \fn void evaluate
     * \brief Evaluate predictor residual.
     * \param [in]  aWorkSets holds input worksets (e.g. states, control, etc)
     * \param [out] aResultWS result/output workset
     ******************************************************************************/
    void evaluate
    (const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        mInternalForces->evaluate(aWorkSets, aResultWS);
        mThermalSources.evaluate(aWorkSets, aResultWS);
    }

    /***************************************************************************//**
     * \fn void evaluateBoundary
     * \brief Evaluate boundary forces, not related to any prescribed boundary force, 
     *        resulting from applying integration by part to the residual equation.
     * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in]  aWorkSets     holds input worksets (e.g. states, control, etc)
     * \param [out] aResultWS     result/output workset
     ******************************************************************************/
    void evaluateBoundary
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResult)
    const override
    { return;  }

    /***************************************************************************//**
     * \fn void evaluatePrescribed
     * \brief Evaluate prescribed boundary forces.
     * \param [in]  aSpatialModel holds mesh and entity sets (e.g. node and side sets) metadata
     * \param [in]  aWorkSets     holds input worksets (e.g. states, control, etc)
     * \param [out] aResultWS     result/output workset
     ******************************************************************************/
    void evaluatePrescribed
    (const Plato::SpatialModel & aSpatialModel,
     const Plato::WorkSets & aWorkSets,
     Plato::ScalarMultiVectorT<ResultT> & aResultWS)
    const override
    {
        if( mHeatFlux != nullptr )
        {
            // set input state worksets
            auto tConfigWS   = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
            auto tControlWS  = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
            auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<PrevTempT>>(aWorkSets.get("previous temperature"));

            // evaluate prescribed flux
            auto tNumCells = aResultWS.extent(0);
            Plato::ScalarMultiVectorT<ResultT> tHeatFluxWS("heat flux", tNumCells, mNumDofsPerCell);
            mHeatFlux->get( aSpatialModel, tPrevTempWS, tControlWS, tConfigWS, tHeatFluxWS );

            auto tCriticalTimeStep = Plato::metadata<Plato::ScalarVector>(aWorkSets.get("critical time step"));
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                Plato::blas2::scale<mNumDofsPerCell>(aCellOrdinal, tCriticalTimeStep(0), tHeatFluxWS);
                Plato::blas2::update<mNumDofsPerCell>(aCellOrdinal, -1.0, tHeatFluxWS, 1.0, aResultWS);
            }, "heat flux contribution");
        }
    }

private:
    /***************************************************************************//**
     * \fn void initializeNaturalBoundaryConditions
     * \brief Set natural boundary conditions. The boundary conditions are set based
     *   on the information available in the input file.
     * \param [in] aInputs  input file metadata
     ******************************************************************************/
    void initializeNaturalBoundaryConditions
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Thermal Natural Boundary Conditions"))
        {
            auto tSublist = aInputs.sublist("Thermal Natural Boundary Conditions");
            mHeatFlux = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(tSublist);
        }
    }
};
// class TemperatureResidual

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::TemperatureResidual, Plato::EnergyConservation, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::TemperatureResidual, Plato::EnergyConservation, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::Fluids::TemperatureResidual, Plato::EnergyConservation, Plato::SimplexFluids, 3, 1)
#endif

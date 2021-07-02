/*
 * FluidsWorkSetsUtils.hpp
 *
 *  Created on: Apr 6, 2021
 */

#pragma once

#include <memory>

#include "BLAS1.hpp"
#include "WorkSets.hpp"
#include "Variables.hpp"
#include "SpatialModel.hpp"

#include "hyperbolic/FluidsWorkSetBuilders.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 * \tparam PhysicsT    fluid flow physics type
 *
 * \fn inline void build_scalar_function_worksets
 *
 * \brief Build metadata work sets used for the evaluation of a scalar function for fluid flow applications.
 *
 * \param [in] aDomain    computational domain metadata ( e.g. mesh and entity sets)
 * \param [in] aControls  1D array of control field
 * \param [in] aVariables state metadata (e.g. pressure, velocity, temperature, etc.)
 * \param [in] aMaps      holds maps from element to local field degrees of freedom
 *
 * \param [in/out] aWorkSets state work sets initialize with the correct FAD type
 ******************************************************************************/
template
<typename EvaluationT,
 typename PhysicsT>
inline void
build_scalar_function_worksets
(const Plato::SpatialDomain              & aDomain,
 const Plato::ScalarVector               & aControls,
 const Plato::Variables                  & aVariables,
 const Plato::LocalOrdinalMaps<PhysicsT> & aMaps,
       Plato::WorkSets                   & aWorkSets)
{
    auto tNumCells = aDomain.numCells();
    Plato::Fluids::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", tNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aDomain, aMaps.mControlOrdinalsMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aDomain, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);

    if(aVariables.defined("current temperature"))
    {
        using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
        auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
            ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", tNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
        tWorkSetBuilder.buildEnergyWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current temperature"), tCurTempWS->mData);
        aWorkSets.set("current temperature", tCurTempWS);
    }

    auto tCriticalTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("critical time step", 1) );
    Plato::blas1::copy(aVariables.vector("critical time step"), tCriticalTimeStep->mData);
    aWorkSets.set("critical time step", tCriticalTimeStep);
}
// function build_scalar_function_worksets


/***************************************************************************//**
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 * \tparam PhysicsT    fluid flow physics type
 *
 * \fn inline void build_scalar_function_worksets
 *
 * \brief Build metadata work sets used for the evaluation of a scalar function for fluid flow applications.
 *
 * \param [in] aNumCells  total number of cells
 * \param [in] aControls  1D array of control field
 * \param [in] aVariables state metadata (e.g. pressure, velocity, temperature, etc.)
 * \param [in] aMaps      holds maps from element to local field degrees of freedom
 *
 * \param [in/out] aWorkSets state work sets initialize with the correct FAD type
 ******************************************************************************/
template
<typename EvaluationT,
 typename PhysicsT>
inline void
build_scalar_function_worksets
(const Plato::OrdinalType                & aNumCells,
 const Plato::ScalarVector               & aControls,
 const Plato::Variables                  & aVariables,
 const Plato::LocalOrdinalMaps<PhysicsT> & aMaps,
       Plato::WorkSets                   & aWorkSets)
{
    Plato::Fluids::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", aNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aNumCells, aMaps.mControlOrdinalsMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aNumCells, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);

    if(aVariables.defined("current temperature"))
    {
        using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
        auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
            ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", aNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
        tWorkSetBuilder.buildEnergyWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current temperature"), tCurTempWS->mData);
        aWorkSets.set("current temperature", tCurTempWS);
    }

    auto tCriticalTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("critical time step", 1) );
    Plato::blas1::copy(aVariables.vector("critical time step"), tCriticalTimeStep->mData);
    aWorkSets.set("critical time step", tCriticalTimeStep);
}
// function build_scalar_function_worksets


/***************************************************************************//**
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 * \tparam PhysicsT    fluid flow physics type
 *
 * \fn inline void build_vector_function_worksets
 *
 * \brief Build metadata work sets used for the evaluation of a vector function for fluid flow applications.
 *
 * \param [in] aDomain    computational domain metadata ( e.g. mesh and entity sets)
 * \param [in] aControls  1D array of control field
 * \param [in] aVariables state metadata (e.g. pressure, velocity, temperature, etc.)
 * \param [in] aMaps      holds maps from element to local field degrees of freedom
 *
 * \param [in/out] aWorkSets state work sets initialize with the correct FAD type
 ******************************************************************************/
template
<typename EvaluationT,
 typename PhysicsT>
inline void
build_vector_function_worksets
(const Plato::SpatialDomain              & aDomain,
 const Plato::ScalarVector               & aControls,
 const Plato::Variables                  & aVariables,
 const Plato::LocalOrdinalMaps<PhysicsT> & aMaps,
       Plato::WorkSets                   & aWorkSets)

{
    auto tNumCells = aDomain.numCells();
    Plato::Fluids::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentPredictorT = typename EvaluationT::MomentumPredictorScalarType;
    auto tPredictorWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPredictorT> > >
        ( Plato::ScalarMultiVectorT<CurrentPredictorT>("current predictor", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current predictor"), tPredictorWS->mData);
    aWorkSets.set("current predictor", tPredictorWS);

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", tNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    if(aVariables.defined("current temperature"))
    {
        using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
        auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
            ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", tNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
        tWorkSetBuilder.buildEnergyWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current temperature"), tCurTempWS->mData);
        aWorkSets.set("current temperature", tCurTempWS);
    }

    using PreviousVelocityT = typename EvaluationT::PreviousMomentumScalarType;
    auto tPrevVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousVelocityT> > >
        ( Plato::ScalarMultiVectorT<PreviousVelocityT>("previous velocity", tNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aDomain, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("previous velocity"), tPrevVelWS->mData);
    aWorkSets.set("previous velocity", tPrevVelWS);

    using PreviousPressureT = typename EvaluationT::PreviousMassScalarType;
    auto tPrevPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousPressureT> > >
        ( Plato::ScalarMultiVectorT<PreviousPressureT>("previous pressure", tNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("previous pressure"), tPrevPressWS->mData);
    aWorkSets.set("previous pressure", tPrevPressWS);

    if(aVariables.defined("previous temperature"))
    {
        using PreviousTemperatureT = typename EvaluationT::PreviousEnergyScalarType;
        auto tPrevTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousTemperatureT> > >
            ( Plato::ScalarMultiVectorT<PreviousTemperatureT>("previous temperature", tNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
        tWorkSetBuilder.buildEnergyWorkSet(aDomain, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("previous temperature"), tPrevTempWS->mData);
        aWorkSets.set("previous temperature", tPrevTempWS);
    }

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aDomain, aMaps.mControlOrdinalsMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aDomain, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);

    auto tCriticalTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("critical time step", 1) );
    Plato::blas1::copy(aVariables.vector("critical time step"), tCriticalTimeStep->mData);
    aWorkSets.set("critical time step", tCriticalTimeStep);
}
// function build_vector_function_worksets


/***************************************************************************//**
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 * \tparam PhysicsT    fluid flow physics type
 *
 * \fn inline void build_vector_function_worksets
 *
 * \brief Build metadata work sets used for the evaluation of a vector function for fluid flow applications.
 *
 * \param [in] aNumCells  total number of cells/elements in the domain
 * \param [in] aControls  1D array of control field
 * \param [in] aVariables state metadata (e.g. pressure, velocity, temperature, etc.)
 * \param [in] aMaps      holds maps from element to local field degrees of freedom
 *
 * \param [in/out] aWorkSets state work sets initialize with the correct FAD type
 ******************************************************************************/
template
<typename EvaluationT,
 typename PhysicsT>
inline void
build_vector_function_worksets
(const Plato::OrdinalType                & aNumCells,
 const Plato::ScalarVector               & aControls,
 const Plato::Variables                  & aVariables,
 const Plato::LocalOrdinalMaps<PhysicsT> & aMaps,
       Plato::WorkSets                   & aWorkSets)
{
    Plato::Fluids::WorkSetBuilder<PhysicsT> tWorkSetBuilder;

    using CurrentPredictorT = typename EvaluationT::MomentumPredictorScalarType;
    auto tPredictorWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPredictorT> > >
        ( Plato::ScalarMultiVectorT<CurrentPredictorT>("current predictor", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current predictor"), tPredictorWS->mData);
    aWorkSets.set("current predictor", tPredictorWS);

    using CurrentVelocityT = typename EvaluationT::CurrentMomentumScalarType;
    auto tCurVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentVelocityT> > >
        ( Plato::ScalarMultiVectorT<CurrentVelocityT>("current velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("current velocity"), tCurVelWS->mData);
    aWorkSets.set("current velocity", tCurVelWS);

    using CurrentPressureT = typename EvaluationT::CurrentMassScalarType;
    auto tCurPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentPressureT> > >
        ( Plato::ScalarMultiVectorT<CurrentPressureT>("current pressure", aNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current pressure"), tCurPressWS->mData);
    aWorkSets.set("current pressure", tCurPressWS);

    if(aVariables.defined("current temperature"))
    {
        using CurrentTemperatureT = typename EvaluationT::CurrentEnergyScalarType;
        auto tCurTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurrentTemperatureT> > >
            ( Plato::ScalarMultiVectorT<CurrentTemperatureT>("current temperature", aNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
        tWorkSetBuilder.buildEnergyWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("current temperature"), tCurTempWS->mData);
        aWorkSets.set("current temperature", tCurTempWS);
    }

    using PreviousVelocityT = typename EvaluationT::PreviousMomentumScalarType;
    auto tPrevVelWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousVelocityT> > >
        ( Plato::ScalarMultiVectorT<PreviousVelocityT>("previous velocity", aNumCells, PhysicsT::SimplexT::mNumMomentumDofsPerCell) );
    tWorkSetBuilder.buildMomentumWorkSet(aNumCells, aMaps.mVectorFieldOrdinalsMap, aVariables.vector("previous velocity"), tPrevVelWS->mData);
    aWorkSets.set("previous velocity", tPrevVelWS);

    using PreviousPressureT = typename EvaluationT::PreviousMassScalarType;
    auto tPrevPressWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousPressureT> > >
        ( Plato::ScalarMultiVectorT<PreviousPressureT>("previous pressure", aNumCells, PhysicsT::SimplexT::mNumMassDofsPerCell) );
    tWorkSetBuilder.buildMassWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("previous pressure"), tPrevPressWS->mData);
    aWorkSets.set("previous pressure", tPrevPressWS);

    if(aVariables.defined("previous temperature"))
    {
        using PreviousTemperatureT = typename EvaluationT::PreviousEnergyScalarType;
        auto tPrevTempWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PreviousTemperatureT> > >
            ( Plato::ScalarMultiVectorT<PreviousTemperatureT>("previous temperature", aNumCells, PhysicsT::SimplexT::mNumEnergyDofsPerCell) );
        tWorkSetBuilder.buildEnergyWorkSet(aNumCells, aMaps.mScalarFieldOrdinalsMap, aVariables.vector("previous temperature"), tPrevTempWS->mData);
        aWorkSets.set("previous temperature", tPrevTempWS);
    }

    using ControlT = typename EvaluationT::ControlScalarType;
    auto tControlWS = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<ControlT> > >
        ( Plato::ScalarMultiVectorT<ControlT>("control", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell) );
    tWorkSetBuilder.buildControlWorkSet(aNumCells, aMaps.mControlOrdinalsMap, aControls, tControlWS->mData);
    aWorkSets.set("control", tControlWS);

    using ConfigT = typename EvaluationT::ConfigScalarType;
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", aNumCells, PhysicsT::SimplexT::mNumNodesPerCell, PhysicsT::SimplexT::mNumConfigDofsPerNode) );
    tWorkSetBuilder.buildConfigWorkSet(aNumCells, aMaps.mNodeCoordinate, tConfig->mData);
    aWorkSets.set("configuration", tConfig);

    auto tCriticalTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("critical time step", 1) );
    Plato::blas1::copy(aVariables.vector("critical time step"), tCriticalTimeStep->mData);
    aWorkSets.set("critical time step", tCriticalTimeStep);
}
// function build_vector_function_worksets

}
// namespace Fluids

}
// namespace Plato

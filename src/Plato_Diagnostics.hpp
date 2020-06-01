/*
 * Plato_Diagnostics.hpp
 *
 *  Created on: Feb 11, 2019
 */

#pragma once

#include <Omega_h_mesh.hpp>

#include "BLAS1.hpp"
#include "WorksetBase.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoMathHelpers.hpp"
#include "geometric/ScalarFunctionBase.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/ScalarFunctionBase.hpp"
#include "LocalVectorFunctionInc.hpp"
#include "ImplicitFunctors.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Test partial derivative of objective function with respect to the controls
 * \param [in] aProblem Plato problem interface
 * \param [in] aMesh    Omega_H mesh database
 * return minimum finite difference error
**********************************************************************************/
template<class PlatoProblem>
inline Plato::Scalar test_objective_grad_wrt_control(PlatoProblem & aProblem, Omega_h::Mesh & aMesh)
{
    // Allocate Data
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(0.5, tControls);

    Plato::ScalarVector tStep = Plato::ScalarVector("Step", tNumVerts);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.025, 0.05, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);

    // Compute gradient
    auto tGlobalStates = aProblem.solution(tControls);
    auto tObjGradZ = aProblem.objectiveGradient(tControls, tGlobalStates);
    auto tGradientDotStep = Plato::blas1::dot(tObjGradZ, tStep);

    std::ostringstream tOutput;
    tOutput << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step"
        << std::setw(18) << "FD Approx" << std::setw(20) << "abs(Error)" << "\n";

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 6;
    auto tTrialControl = Plato::ScalarVector("Trial Control", tNumVerts);

    std::vector<Plato::Scalar> tFiniteDiffApproxError;
    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        auto tEpsilon = static_cast<Plato::Scalar>(1) / std::pow(static_cast<Plato::Scalar>(10), tIndex);

        // four point finite difference approximation
        Plato::blas1::update(1.0, tControls, 0.0, tTrialControl);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialControl);
        tGlobalStates = aProblem.solution(tTrialControl);
        auto tValuePlus1Eps = aProblem.objectiveValue(tTrialControl, tGlobalStates);

        Plato::blas1::update(1.0, tControls, 0.0, tTrialControl);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialControl);
        tGlobalStates = aProblem.solution(tTrialControl);
        auto tValueMinus1Eps = aProblem.objectiveValue(tTrialControl, tGlobalStates);

        Plato::blas1::update(1.0, tControls, 0.0, tTrialControl);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        tGlobalStates = aProblem.solution(tTrialControl);
        auto tValuePlus2Eps = aProblem.objectiveValue(tTrialControl, tGlobalStates);

        Plato::blas1::update(1.0, tControls, 0.0, tTrialControl);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        tGlobalStates = aProblem.solution(tTrialControl);
        auto tValueMinus2Eps = aProblem.objectiveValue(tTrialControl, tGlobalStates);

        auto tNumerator = -tValuePlus2Eps + static_cast<Plato::Scalar>(8.) * tValuePlus1Eps
                - static_cast<Plato::Scalar>(8.) * tValueMinus1Eps + tValueMinus2Eps;
        auto tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        auto tFiniteDiffAppx = tNumerator / tDenominator;
        auto tAppxError = abs(tFiniteDiffAppx - tGradientDotStep);
        tFiniteDiffApproxError.push_back(tAppxError);

        tOutput << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
            << tGradientDotStep << std::setw(19) << tFiniteDiffAppx << std::setw(19) << tAppxError << "\n";
    }
    std::cout << tOutput.str().c_str();

    const auto tMinError = *std::min_element(tFiniteDiffApproxError.begin(), tFiniteDiffApproxError.end());
    return tMinError;
}
// function test_objective_grad_wrt_control


/******************************************************************************//**
 * \brief Test partial derivative of constraint function with respect to the controls
 * \param [in] aProblem Plato problem interface
 * \param [in] aMesh    Omega_H mesh database
 * return minimum finite difference error
**********************************************************************************/
template<class PlatoProblem>
inline Plato::Scalar test_constraint_grad_wrt_control(PlatoProblem & aProblem, Omega_h::Mesh & aMesh)
{
    // Allocate Data
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(0.5, tControls);

    Plato::ScalarVector tStep = Plato::ScalarVector("Step", tNumVerts);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.025, 0.05, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);

    // Compute gradient
    auto tConstraintGradZ = aProblem.constraintGradient(tControls);
    auto tGradientDotStep = Plato::blas1::dot(tConstraintGradZ, tStep);

    std::ostringstream tOutput;
    tOutput << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step"
        << std::setw(18) << "FD Approx" << std::setw(20) << "abs(Error)" << "\n";

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 6;
    auto tTrialControl = Plato::ScalarVector("Trial Control", tNumVerts);

    std::vector<Plato::Scalar> tFiniteDiffApproxError;
    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        auto tEpsilon = static_cast<Plato::Scalar>(1) / std::pow(static_cast<Plato::Scalar>(10), tIndex);

        // four point finite difference approximation
        Plato::blas1::update(1.0, tControls, 0.0, tTrialControl);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialControl);
        auto tValuePlus1Eps = aProblem.constraintValue(tTrialControl);

        Plato::blas1::update(1.0, tControls, 0.0, tTrialControl);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialControl);
        auto tValueMinus1Eps = aProblem.constraintValue(tTrialControl);

        Plato::blas1::update(1.0, tControls, 0.0, tTrialControl);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        auto tValuePlus2Eps = aProblem.constraintValue(tTrialControl);

        Plato::blas1::update(1.0, tControls, 0.0, tTrialControl);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        auto tValueMinus2Eps = aProblem.constraintValue(tTrialControl);

        auto tNumerator = -tValuePlus2Eps + static_cast<Plato::Scalar>(8.) * tValuePlus1Eps
                - static_cast<Plato::Scalar>(8.) * tValueMinus1Eps + tValueMinus2Eps;
        auto tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        auto tFiniteDiffAppx = tNumerator / tDenominator;
        auto tAppxError = abs(tFiniteDiffAppx - tGradientDotStep);
        tFiniteDiffApproxError.push_back(tAppxError);

        tOutput << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
            << tGradientDotStep << std::setw(19) << tFiniteDiffAppx << std::setw(19) << tAppxError << "\n";
    }
    std::cout << tOutput.str().c_str();

    const auto tMinError = *std::min_element(tFiniteDiffApproxError.begin(), tFiniteDiffApproxError.end());
    return tMinError;
}
// function test_constraint_grad_wrt_control


// function test_partial_control
/******************************************************************************//**
 * \brief Test partial derivative with respect to the control variables
 * \param [in] aMesh mesh database
 * \param [in] aCriterion scalar function (i.e. scalar criterion) interface
 * \param [in] aSuperscriptLowerBound lower bound on the superscript used to compute the step (e.g. \f$10^{lb}/$f
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
inline void test_partial_control(Omega_h::Mesh & aMesh,
                                 Plato::Elliptic::AbstractScalarFunction<EvaluationType> & aCriterion,
                                 Plato::OrdinalType aSuperscriptLowerBound = 1,
                                 Plato::OrdinalType aSuperscriptUpperBound = 10)
{
    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;
    constexpr Plato::OrdinalType tDofsPerCell = SimplexPhysics::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = SimplexPhysics::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<SimplexPhysics> tWorksetBase(aMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tState("State", tTotalNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::blas1::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // FINITE DIFFERENCE TEST
    aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
    constexpr Plato::OrdinalType tNumControlFields = 1;
    Plato::ScalarVector tPartialZ("objective partial control", tNumVerts);
    Plato::VectorEntryOrdinal<tSpaceDim, tNumControlFields> tControlEntryOrdinal(&aMesh);
    Plato::assemble_scalar_gradient_fad<tNodesPerCell>(tNumCells, tControlEntryOrdinal, tResultWS, tPartialZ);

    Plato::ScalarVector tStep("step", tNumVerts);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::blas1::dot(tPartialZ, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18) << "FD Approx"
               << std::setw(20) << "abs(Error)" << "\n";

    Plato::ScalarVector tTrialControl("trial control", tNumVerts);
    for(Plato::OrdinalType tIndex = aSuperscriptLowerBound; tIndex <= aSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueOne = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueTwo = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueThree = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        tWorksetBase.worksetControl(tTrialControl, tControlWS);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueFour = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne
                - static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
              << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}
// function test_partial_control

/******************************************************************************//**
 * \brief Test partial derivative with respect to the state variables
 * \param [in] aMesh mesh database
 * \param [in] aCriterion scalar function (i.e. scalar criterion) interface
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
inline void test_partial_state(Omega_h::Mesh & aMesh, Plato::Elliptic::AbstractScalarFunction<EvaluationType> & aCriterion)
{
    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;
    constexpr Plato::OrdinalType tDofsPerCell = SimplexPhysics::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = SimplexPhysics::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(aMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tState("State", tTotalNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::blas1::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // finite difference
    aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
    Plato::ScalarVector tPartialU("objective partial state", tTotalNumDofs);
    Plato::VectorEntryOrdinal<tSpaceDim, tDofsPerNode> tStateEntryOrdinal(&aMesh);
    Plato::assemble_vector_gradient_fad<tNodesPerCell, tDofsPerNode>(tNumCells, tStateEntryOrdinal, tResultWS, tPartialU);

    Plato::ScalarVector tStep("step", tTotalNumDofs);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::blas1::dot(tPartialU, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18) << "FD Approx"
               << std::setw(20) << "abs(Error)" << "\n";

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 10;
    Plato::ScalarVector tTrialState("trial state", tTotalNumDofs);
    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueOne = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::blas1::update(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueTwo = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::blas1::update(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueThree = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::blas1::update(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialState);
        tWorksetBase.worksetState(tTrialState, tStateWS);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tResultWS);
        aCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);
        Plato::Scalar tObjFuncValueFour = Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResultWS);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne
                - static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
              << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}
// function test_partial_state


template<typename EvaluationType, typename Simplex>
inline void test_partial_control(Omega_h::Mesh & aMesh,
                                 Plato::Geometric::ScalarFunctionBase & aScalarFuncBase,
                                 Plato::OrdinalType aSuperscriptLowerBound = 1,
                                 Plato::OrdinalType aSuperscriptUpperBound = 10)
{
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // FINITE DIFFERENCE TEST
    Plato::ScalarVector tPartialZ = aScalarFuncBase.gradient_z(tControl);

    Plato::ScalarVector tStep("step", tNumVerts);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::blas1::dot(tPartialZ, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18) << "FD Approx"
               << std::setw(20) << "abs(Error)" << "\n";

    Plato::ScalarVector tTrialControl("trial control", tNumVerts);
    for(Plato::OrdinalType tIndex = aSuperscriptLowerBound; tIndex <= aSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueOne = aScalarFuncBase.value(tTrialControl);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueTwo = aScalarFuncBase.value(tTrialControl);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueThree = aScalarFuncBase.value(tTrialControl);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueFour = aScalarFuncBase.value(tTrialControl);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne
                - static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
              << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}

// function test_partial_control
template<typename EvaluationType, typename SimplexPhysics>
inline void test_partial_control(Omega_h::Mesh & aMesh,
                                 Plato::Elliptic::ScalarFunctionBase & aScalarFuncBase,
                                 Plato::OrdinalType aSuperscriptLowerBound = 1,
                                 Plato::OrdinalType aSuperscriptUpperBound = 10)
{
    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarMultiVector tStates("State", /*numSteps=*/1, tTotalNumDofs);
    auto tState = Kokkos::subview(tStates, 0, Kokkos::ALL());
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::blas1::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);

    // FINITE DIFFERENCE TEST
    Plato::ScalarVector tPartialZ = aScalarFuncBase.gradient_z(Plato::Solution(tStates), tControl, 0.0);


    Plato::ScalarVector tStep("step", tNumVerts);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::blas1::dot(tPartialZ, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18) << "FD Approx"
               << std::setw(20) << "abs(Error)" << "\n";

    Plato::ScalarVector tTrialControl("trial control", tNumVerts);
    for(Plato::OrdinalType tIndex = aSuperscriptLowerBound; tIndex <= aSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueOne = aScalarFuncBase.value(Plato::Solution(tStates), tTrialControl, 0.0);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueTwo = aScalarFuncBase.value(Plato::Solution(tStates), tTrialControl, 0.0);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueThree = aScalarFuncBase.value(Plato::Solution(tStates), tTrialControl, 0.0);

        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        Plato::Scalar tObjFuncValueFour = aScalarFuncBase.value(Plato::Solution(tStates), tTrialControl, 0.0);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne
                - static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
              << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}
// function test_partial_control

/******************************************************************************//**
 * \brief Test partial derivative with respect to the state variables
 * \param [in] aMesh mesh database
 * \param [in] aCriterion scalar function (i.e. scalar criterion) interface
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
inline void test_partial_state(Omega_h::Mesh & aMesh, Plato::Elliptic::ScalarFunctionBase & aScalarFuncBase)
{
    using StateT = typename EvaluationType::StateScalarType;
    using ConfigT = typename EvaluationType::ConfigScalarType;
    using ResultT = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // Create state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarMultiVector tStates("State", /*numSteps=*/1, tTotalNumDofs);
    auto tState = Kokkos::subview(tStates, 0, Kokkos::ALL());
    auto tHostState = Kokkos::create_mirror(tState);
    Plato::blas1::random(1, 5, tHostState);
    Kokkos::deep_copy(tState, tHostState);

    Plato::ScalarVector tPartialU = aScalarFuncBase.gradient_u(Plato::Solution(tStates), tControl, 0.0);

    Plato::ScalarMultiVector tTrialStates("trial state", /*numSteps=*/1, tTotalNumDofs);
    auto tTrialState = Kokkos::subview(tTrialStates, 0, Kokkos::ALL());

    Plato::ScalarVector tStep("step", tTotalNumDofs);
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::Scalar tGradientDotStep = Plato::blas1::dot(tPartialU, tStep);

    std::cout << std::right << std::setw(18) << "\nStep Size" << std::setw(20) << "Grad'*Step" << std::setw(18) << "FD Approx"
               << std::setw(20) << "abs(Error)" << "\n";

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 10;
    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update<Plato::ScalarVector>(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update<Plato::ScalarVector>(tEpsilon, tStep, 1.0, tTrialState);
        Plato::Scalar tObjFuncValueOne = aScalarFuncBase.value(Plato::Solution(tTrialStates), tControl, 0.0);

        Plato::blas1::update<Plato::ScalarVector>(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update<Plato::ScalarVector>(-tEpsilon, tStep, 1.0, tTrialState);
        Plato::Scalar tObjFuncValueTwo = aScalarFuncBase.value(Plato::Solution(tTrialStates), tControl, 0.0);

        Plato::blas1::update<Plato::ScalarVector>(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update<Plato::ScalarVector>(2.0 * tEpsilon, tStep, 1.0, tTrialState);
        Plato::Scalar tObjFuncValueThree = aScalarFuncBase.value(Plato::Solution(tTrialStates), tControl, 0.0);

        Plato::blas1::update<Plato::ScalarVector>(1.0, tState, 0.0, tTrialState);
        Plato::blas1::update<Plato::ScalarVector>(-2.0 * tEpsilon, tStep, 1.0, tTrialState);
        Plato::Scalar tObjFuncValueFour = aScalarFuncBase.value(Plato::Solution(tTrialStates), tControl, 0.0);

        Plato::Scalar tNumerator = -tObjFuncValueThree + static_cast<Plato::Scalar>(8.) * tObjFuncValueOne
                - static_cast<Plato::Scalar>(8.) * tObjFuncValueTwo + tObjFuncValueFour;
        Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
        Plato::Scalar tFiniteDiffAppxError = tNumerator / tDenominator;
        Plato::Scalar tAppxError = std::abs(tFiniteDiffAppxError - tGradientDotStep);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) << tEpsilon << std::setw(19)
              << tGradientDotStep << std::setw(19) << tFiniteDiffAppxError << std::setw(19) << tAppxError << "\n";
    }
}
// function test_partial_state

inline Plato::ScalarVector
local_workset_matrix_vector_multiply(const Plato::ScalarArray3D & aWorkset,
                                     const Plato::ScalarVector & aVector)
{
    const Plato::OrdinalType tNumCells            = aWorkset.extent(0);
    const Plato::OrdinalType tNumLocalDofsPerCell = aWorkset.extent(1);
    const Plato::OrdinalType tVectorSize          = aVector.extent(0);

    Plato::ScalarVector tResult("result", tVectorSize);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), 
                         LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::OrdinalType tStartingRowIndex = aCellOrdinal * tNumLocalDofsPerCell;
        for (Plato::OrdinalType tRow = 0; tRow < tNumLocalDofsPerCell; ++tRow)
        {
            tResult(tStartingRowIndex + tRow) = 0.0;
            for (Plato::OrdinalType tColumn = 0; tColumn < tNumLocalDofsPerCell; ++tColumn)
            {
                Plato::Scalar tValue = 
                             aWorkset(aCellOrdinal, tRow, tColumn) * aVector(tStartingRowIndex + tColumn);
                tResult(tStartingRowIndex + tRow) += tValue;
            }
        }
    }, "matrix vector multiply");
    return tResult;
}

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType DofsPerNode>
inline Plato::ScalarVector
global_workset_matrix_vector_multiply(const Plato::ScalarArray3D & aWorkset,
                                      const Plato::ScalarVector & aVector,
                                      const Plato::VectorEntryOrdinal<SpaceDim,DofsPerNode> & aEntryOrdinal,
                                      const Plato::OrdinalType & aNumNodesPerCell,
                                      const Plato::OrdinalType & aNumMatrixRows)
{
    const Plato::OrdinalType tNumWorksetRows = aWorkset.extent(0);
    const Plato::OrdinalType tNumWorksetCols = aWorkset.extent(1);
    const Plato::OrdinalType tVectorSize     = aVector.extent(0);

    Plato::ScalarVector tResult("result", aNumMatrixRows);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumWorksetRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for (Plato::OrdinalType tWorksetCol = 0; tWorksetCol < tNumWorksetCols; ++tWorksetCol)
        {
            Plato::OrdinalType tMatrixRow = aCellOrdinal * tNumWorksetCols + tWorksetCol;
            tResult(tMatrixRow) = 0.0;

            Plato::OrdinalType tADVarIndex = 0;
            for (Plato::OrdinalType tNode = 0; tNode < aNumNodesPerCell; ++tNode)
            {
                for (Plato::OrdinalType tDof = 0; tDof < DofsPerNode; ++tDof)
                {
                    Plato::OrdinalType tMatrixCol = aEntryOrdinal(aCellOrdinal, tNode, tDof);
                    Plato::Scalar tValue = 
                        aWorkset(aCellOrdinal, tWorksetCol, tADVarIndex) * aVector(tMatrixCol);
                    tResult(tMatrixRow) += tValue;
                    ++tADVarIndex;
                }
            }
        }
    }, "matrix vector multiply");
    return tResult;
}

template<Plato::OrdinalType D1, Plato::OrdinalType D2>
inline Plato::ScalarVector
control_workset_matrix_vector_multiply(const Plato::ScalarArray3D & aWorkset,
                                       const Plato::ScalarVector & aVector,
                                       const Plato::VectorEntryOrdinal<D1,D2> & aEntryOrdinal,
                                       const Plato::OrdinalType & aNumMatrixRows)
{
    const Plato::OrdinalType tNumWorksetRows = aWorkset.extent(0);
    const Plato::OrdinalType tNumWorksetCols = aWorkset.extent(1);
    const Plato::OrdinalType tNumADvarsPerCell = aWorkset.extent(2);
    const Plato::OrdinalType tVectorSize     = aVector.extent(0);

    Plato::ScalarVector tResult("result", aNumMatrixRows);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumWorksetRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for (Plato::OrdinalType tWorksetCol = 0; tWorksetCol < tNumWorksetCols; ++tWorksetCol)
        {
            Plato::OrdinalType tMatrixRow = aCellOrdinal * tNumWorksetCols + tWorksetCol;
            tResult(tMatrixRow) = 0.0;
            for (Plato::OrdinalType tADVariableIndex = 0; tADVariableIndex < tNumADvarsPerCell; ++tADVariableIndex)
            {
                Plato::OrdinalType tMatrixCol = aEntryOrdinal(aCellOrdinal, tADVariableIndex);
                Plato::Scalar tValue = aWorkset(aCellOrdinal, tWorksetCol, tADVariableIndex) * aVector(tMatrixCol);
                tResult(tMatrixRow) += tValue;
            }
        }
    }, "matrix vector multiply");
    return tResult;
}

/******************************************************************************//**
 * \brief Test partial derivative with respect to the global state variables
 * \param [in] aMesh mesh database
 * \param [in] aLocalVectorFuncInc local vector function inc to evaluate derivative of
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
inline void
test_partial_global_state(Omega_h::Mesh & aMesh, Plato::LocalVectorFunctionInc<SimplexPhysics> & aLocalVectorFuncInc)
{
    using StateT   = typename EvaluationType::StateScalarType;
    using LocalStateT   = typename EvaluationType::LocalStateScalarType;
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;
    constexpr Plato::OrdinalType tLocalDofsPerCell = SimplexPhysics::mNumLocalDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = SimplexPhysics::mNumNodesPerCell;

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // Create global state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tGlobalState("Global State", tTotalNumDofs);
    auto tHostGlobalState = Kokkos::create_mirror(tGlobalState);
    Plato::blas1::random(1, 5, tHostGlobalState);
    Kokkos::deep_copy(tGlobalState, tHostGlobalState);

    // Create previous global state workset
    Plato::ScalarVector tPrevGlobalState("Previous Global State", tTotalNumDofs);
    auto tHostPrevGlobalState = Kokkos::create_mirror(tPrevGlobalState);
    Plato::blas1::random(1, 5, tHostPrevGlobalState);
    Kokkos::deep_copy(tPrevGlobalState, tHostPrevGlobalState);

    // Create local state workset
    const Plato::OrdinalType tTotalNumLocalDofs = tNumCells * tLocalDofsPerCell;
    Plato::ScalarVector tLocalState("Local State", tTotalNumLocalDofs);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Plato::blas1::random(1.0, 2.0, tHostLocalState);
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    // Create previous local state workset
    Plato::ScalarVector tPrevLocalState("Previous Local State", tTotalNumLocalDofs);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    Plato::blas1::random(0.1, 0.9, tHostPrevLocalState);
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarArray3D tPartialU =
            aLocalVectorFuncInc.gradient_u(tGlobalState, tPrevGlobalState,
                                           tLocalState, tPrevLocalState,
                                           tControl, 0.0);

    constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
    Plato::VectorEntryOrdinal<tSpaceDim, tDofsPerNode> tEntryOrdinal(&aMesh);

    Plato::ScalarVector tStep("global state step", tTotalNumDofs); 
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.5, 1.0, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::ScalarVector tGradientDotStep =
            Plato::global_workset_matrix_vector_multiply(tPartialU, tStep, tEntryOrdinal, tNumNodesPerCell, tTotalNumDofs);

    std::cout << std::right << std::setw(14) << "\nStep Size" 
              << std::setw(20) << "abs(Error)" << std::endl;

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 6;
    Plato::ScalarVector tTrialGlobalState("trial global state", tTotalNumDofs);

    Plato::ScalarVector tErrorVector("error vector", tTotalNumDofs);

    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tGlobalState, 0.0, tTrialGlobalState);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialGlobalState);
        Plato::ScalarVector tVectorValueOne = aLocalVectorFuncInc.value(tTrialGlobalState, tPrevGlobalState,
                                                                        tLocalState, tPrevLocalState,
                                                                        tControl, 0.0);
        Plato::blas1::update(1.0, tGlobalState, 0.0, tTrialGlobalState);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialGlobalState);
        Plato::ScalarVector tVectorValueTwo = aLocalVectorFuncInc.value(tTrialGlobalState, tPrevGlobalState,
                                                                        tLocalState, tPrevLocalState,
                                                                        tControl, 0.0);
        Plato::blas1::update(1.0, tGlobalState, 0.0, tTrialGlobalState);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialGlobalState);
        Plato::ScalarVector tVectorValueThree = aLocalVectorFuncInc.value(tTrialGlobalState, tPrevGlobalState,
                                                                          tLocalState, tPrevLocalState,
                                                                          tControl, 0.0);
        Plato::blas1::update(1.0, tGlobalState, 0.0, tTrialGlobalState);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialGlobalState);
        Plato::ScalarVector tVectorValueFour = aLocalVectorFuncInc.value(tTrialGlobalState, tPrevGlobalState,
                                                                         tLocalState, tPrevLocalState,
                                                                         tControl, 0.0);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tTotalNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aDofOrdinal)
        {
            Plato::Scalar tValuePlus1Eps  = tVectorValueOne(aDofOrdinal);
            Plato::Scalar tValueMinus1Eps = tVectorValueTwo(aDofOrdinal);
            Plato::Scalar tValuePlus2Eps  = tVectorValueThree(aDofOrdinal);
            Plato::Scalar tValueMinus2Eps = tVectorValueFour(aDofOrdinal);

            Plato::Scalar tNumerator = -tValuePlus2Eps + static_cast<Plato::Scalar>(8.) * tValuePlus1Eps
                                       - static_cast<Plato::Scalar>(8.) * tValueMinus1Eps + tValueMinus2Eps;
            Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
            Plato::Scalar tFiniteDiffAppx = tNumerator / tDenominator;

            Plato::Scalar tAppxError = abs(tFiniteDiffAppx - tGradientDotStep(aDofOrdinal));

            tErrorVector(aDofOrdinal) = tAppxError;

        }, "compute error");

        Plato::Scalar tL1Error = 0.0;
        Plato::blas1::local_sum(tErrorVector, tL1Error);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) 
                  << tEpsilon << std::setw(19)
                  << tL1Error << std::endl;
    }
}
// function test_partial_global_state


/******************************************************************************//**
 * \brief Test partial derivative with respect to the previous global state variables
 * \param [in] aMesh mesh database
 * \param [in] aLocalVectorFuncInc local vector function inc to evaluate derivative of
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
inline void
test_partial_prev_global_state(Omega_h::Mesh & aMesh, Plato::LocalVectorFunctionInc<SimplexPhysics> & aLocalVectorFuncInc)
{
    using StateT   = typename EvaluationType::StateScalarType;
    using LocalStateT = typename EvaluationType::LocalStateScalarType;
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;
    constexpr Plato::OrdinalType tLocalDofsPerCell = SimplexPhysics::mNumLocalDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = SimplexPhysics::mNumNodesPerCell;

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // Create global state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tGlobalState("Global State", tTotalNumDofs);
    auto tHostGlobalState = Kokkos::create_mirror(tGlobalState);
    Plato::blas1::random(1, 5, tHostGlobalState);
    Kokkos::deep_copy(tGlobalState, tHostGlobalState);

    // Create previous global state workset
    Plato::ScalarVector tPrevGlobalState("Previous Global State", tTotalNumDofs);
    auto tHostPrevGlobalState = Kokkos::create_mirror(tPrevGlobalState);
    Plato::blas1::random(1, 5, tHostPrevGlobalState);
    Kokkos::deep_copy(tPrevGlobalState, tHostPrevGlobalState);

    // Create local state workset
    const Plato::OrdinalType tTotalNumLocalDofs = tNumCells * tLocalDofsPerCell;
    Plato::ScalarVector tLocalState("Local State", tTotalNumLocalDofs);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Plato::blas1::random(1.0, 2.0, tHostLocalState);
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    // Create previous local state workset
    Plato::ScalarVector tPrevLocalState("Previous Local State", tTotalNumLocalDofs);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    Plato::blas1::random(0.1, 0.9, tHostPrevLocalState);
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarArray3D tPartialUP =
            aLocalVectorFuncInc.gradient_up(tGlobalState, tPrevGlobalState,
                                            tLocalState, tPrevLocalState,
                                            tControl, 0.0);

    constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
    Plato::VectorEntryOrdinal<tSpaceDim, tDofsPerNode> tEntryOrdinal(&aMesh);

    Plato::ScalarVector tStep("prev global state step", tTotalNumDofs); 
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.5, 1.0, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::ScalarVector tGradientDotStep =
            Plato::global_workset_matrix_vector_multiply(tPartialUP, tStep, tEntryOrdinal, tNumNodesPerCell, tTotalNumDofs);

    std::cout << std::right << std::setw(14) << "\nStep Size" 
              << std::setw(20) << "abs(Error)" << std::endl;

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 6;
    Plato::ScalarVector tTrialPrevGlobalState("trial previous global state", tTotalNumDofs);

    Plato::ScalarVector tErrorVector("error vector", tTotalNumDofs);

    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tPrevGlobalState, 0.0, tTrialPrevGlobalState);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialPrevGlobalState);
        Plato::ScalarVector tVectorValueOne = aLocalVectorFuncInc.value(tGlobalState, tTrialPrevGlobalState,
                                                                        tLocalState, tPrevLocalState,
                                                                        tControl, 0.0);
        Plato::blas1::update(1.0, tPrevGlobalState, 0.0, tTrialPrevGlobalState);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialPrevGlobalState);
        Plato::ScalarVector tVectorValueTwo = aLocalVectorFuncInc.value(tGlobalState, tTrialPrevGlobalState,
                                                                        tLocalState, tPrevLocalState,
                                                                        tControl, 0.0);
        Plato::blas1::update(1.0, tPrevGlobalState, 0.0, tTrialPrevGlobalState);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialPrevGlobalState);
        Plato::ScalarVector tVectorValueThree = aLocalVectorFuncInc.value(tGlobalState, tTrialPrevGlobalState,
                                                                          tLocalState, tPrevLocalState,
                                                                          tControl, 0.0);
        Plato::blas1::update(1.0, tPrevGlobalState, 0.0, tTrialPrevGlobalState);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialPrevGlobalState);
        Plato::ScalarVector tVectorValueFour = aLocalVectorFuncInc.value(tGlobalState, tTrialPrevGlobalState,
                                                                         tLocalState, tPrevLocalState,
                                                                         tControl, 0.0);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tTotalNumDofs), 
                                 LAMBDA_EXPRESSION(const Plato::OrdinalType & aDofOrdinal)
        {
            Plato::Scalar tValuePlus1Eps  = tVectorValueOne(aDofOrdinal);
            Plato::Scalar tValueMinus1Eps = tVectorValueTwo(aDofOrdinal);
            Plato::Scalar tValuePlus2Eps  = tVectorValueThree(aDofOrdinal);
            Plato::Scalar tValueMinus2Eps = tVectorValueFour(aDofOrdinal);

            Plato::Scalar tNumerator = -tValuePlus2Eps + static_cast<Plato::Scalar>(8.) * tValuePlus1Eps
                                       - static_cast<Plato::Scalar>(8.) * tValueMinus1Eps + tValueMinus2Eps;
            Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
            Plato::Scalar tFiniteDiffAppx = tNumerator / tDenominator;

            Plato::Scalar tAppxError = abs(tFiniteDiffAppx - tGradientDotStep(aDofOrdinal));

            tErrorVector(aDofOrdinal) = tAppxError;

        }, "compute error");

        Plato::Scalar tL1Error = 0.0;
        Plato::blas1::local_sum(tErrorVector, tL1Error);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) 
                  << tEpsilon << std::setw(19)
                  << tL1Error << std::endl;
    }
}
// function test_partial_prev_global_state


/******************************************************************************//**
 * \brief Test partial derivative with respect to the local state variables
 * \param [in] aMesh mesh database
 * \param [in] aLocalVectorFuncInc local vector function inc to evaluate derivative of
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
inline void
test_partial_local_state(Omega_h::Mesh & aMesh, Plato::LocalVectorFunctionInc<SimplexPhysics> & aLocalVectorFuncInc)
{
    using StateT   = typename EvaluationType::StateScalarType;
    using LocalStateT   = typename EvaluationType::LocalStateScalarType;
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;
    constexpr Plato::OrdinalType tLocalDofsPerCell = SimplexPhysics::mNumLocalDofsPerCell;

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // Create global state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tGlobalState("Global State", tTotalNumDofs);
    auto tHostGlobalState = Kokkos::create_mirror(tGlobalState);
    Plato::blas1::random(1, 5, tHostGlobalState);
    Kokkos::deep_copy(tGlobalState, tHostGlobalState);

    // Create previous global state workset
    Plato::ScalarVector tPrevGlobalState("Previous Global State", tTotalNumDofs);
    auto tHostPrevGlobalState = Kokkos::create_mirror(tPrevGlobalState);
    Plato::blas1::random(1, 5, tHostPrevGlobalState);
    Kokkos::deep_copy(tPrevGlobalState, tHostPrevGlobalState);

    // Create local state workset
    const Plato::OrdinalType tTotalNumLocalDofs = tNumCells * tLocalDofsPerCell;
    Plato::ScalarVector tLocalState("Local State", tTotalNumLocalDofs);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Plato::blas1::random(1.0, 2.0, tHostLocalState);
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    // Create previous local state workset
    Plato::ScalarVector tPrevLocalState("Previous Local State", tTotalNumLocalDofs);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    Plato::blas1::random(0.1, 0.9, tHostPrevLocalState);
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarArray3D tPartialC =
            aLocalVectorFuncInc.gradient_c(tGlobalState, tPrevGlobalState,
                                           tLocalState, tPrevLocalState,
                                           tControl, 0.0);

    Plato::ScalarVector tStep("local state step", tTotalNumLocalDofs); 
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.5, 1.0, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::ScalarVector tGradientDotStep = Plato::local_workset_matrix_vector_multiply(tPartialC, tStep);

    std::cout << std::right << std::setw(14) << "\nStep Size" 
              << std::setw(20) << "abs(Error)" << std::endl;

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 6;
    Plato::ScalarVector tTrialLocalState("trial local state", tTotalNumLocalDofs);

    Plato::ScalarVector tErrorVector("error vector", tTotalNumLocalDofs);

    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tLocalState, 0.0, tTrialLocalState);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialLocalState);
        Plato::ScalarVector tVectorValueOne = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                        tTrialLocalState, tPrevLocalState,
                                                                        tControl, 0.0);
        Plato::blas1::update(1.0, tLocalState, 0.0, tTrialLocalState);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialLocalState);
        Plato::ScalarVector tVectorValueTwo = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                        tTrialLocalState, tPrevLocalState,
                                                                        tControl, 0.0);
        Plato::blas1::update(1.0, tLocalState, 0.0, tTrialLocalState);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialLocalState);
        Plato::ScalarVector tVectorValueThree = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                          tTrialLocalState, tPrevLocalState,
                                                                          tControl, 0.0);
        Plato::blas1::update(1.0, tLocalState, 0.0, tTrialLocalState);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialLocalState);
        Plato::ScalarVector tVectorValueFour = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                         tTrialLocalState, tPrevLocalState,
                                                                         tControl, 0.0);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tTotalNumLocalDofs), 
                                 LAMBDA_EXPRESSION(const Plato::OrdinalType & aDofOrdinal)
        {
            Plato::Scalar tValuePlus1Eps  = tVectorValueOne(aDofOrdinal);
            Plato::Scalar tValueMinus1Eps = tVectorValueTwo(aDofOrdinal);
            Plato::Scalar tValuePlus2Eps  = tVectorValueThree(aDofOrdinal);
            Plato::Scalar tValueMinus2Eps = tVectorValueFour(aDofOrdinal);

            Plato::Scalar tNumerator = -tValuePlus2Eps + static_cast<Plato::Scalar>(8.) * tValuePlus1Eps
                                       - static_cast<Plato::Scalar>(8.) * tValueMinus1Eps + tValueMinus2Eps;
            Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
            Plato::Scalar tFiniteDiffAppx = tNumerator / tDenominator;

            Plato::Scalar tAppxError = abs(tFiniteDiffAppx - tGradientDotStep(aDofOrdinal));

            tErrorVector(aDofOrdinal) = tAppxError;

        }, "compute error");

        Plato::Scalar tL1Error = 0.0;
        Plato::blas1::local_sum(tErrorVector, tL1Error);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) 
                  << tEpsilon << std::setw(19)
                  << tL1Error << std::endl;
    }
}
// function test_partial_local_state


/******************************************************************************//**
 * \brief Test partial derivative with respect to the previous local state variables
 * \param [in] aMesh mesh database
 * \param [in] aLocalVectorFuncInc local vector function inc to evaluate derivative of
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
inline void
test_partial_prev_local_state(Omega_h::Mesh & aMesh, Plato::LocalVectorFunctionInc<SimplexPhysics> & aLocalVectorFuncInc)
{
    using StateT   = typename EvaluationType::StateScalarType;
    using LocalStateT   = typename EvaluationType::LocalStateScalarType;
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;
    constexpr Plato::OrdinalType tLocalDofsPerCell = SimplexPhysics::mNumLocalDofsPerCell;

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // Create global state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tGlobalState("Global State", tTotalNumDofs);
    auto tHostGlobalState = Kokkos::create_mirror(tGlobalState);
    Plato::blas1::random(1, 5, tHostGlobalState);
    Kokkos::deep_copy(tGlobalState, tHostGlobalState);

    // Create previous global state workset
    Plato::ScalarVector tPrevGlobalState("Previous Global State", tTotalNumDofs);
    auto tHostPrevGlobalState = Kokkos::create_mirror(tPrevGlobalState);
    Plato::blas1::random(1, 5, tHostPrevGlobalState);
    Kokkos::deep_copy(tPrevGlobalState, tHostPrevGlobalState);

    // Create local state workset
    const Plato::OrdinalType tTotalNumLocalDofs = tNumCells * tLocalDofsPerCell;
    Plato::ScalarVector tLocalState("Local State", tTotalNumLocalDofs);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Plato::blas1::random(1.0, 2.0, tHostLocalState);
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    // Create previous local state workset
    Plato::ScalarVector tPrevLocalState("Previous Local State", tTotalNumLocalDofs);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    Plato::blas1::random(0.1, 0.9, tHostPrevLocalState);
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarArray3D tPartialCP =
            aLocalVectorFuncInc.gradient_cp(tGlobalState, tPrevGlobalState,
                                            tLocalState,  tPrevLocalState,
                                            tControl, 0.0 /*time step*/);

    Plato::ScalarVector tStep("previous local state step", tTotalNumLocalDofs); 
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.5, 1.0, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::ScalarVector tGradientDotStep = Plato::local_workset_matrix_vector_multiply(tPartialCP, tStep);

    std::cout << std::right << std::setw(14) << "\nStep Size" 
              << std::setw(20) << "abs(Error)" << std::endl;

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 0;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 4;
    Plato::ScalarVector tTrialPrevLocalState("trial previous local state", tTotalNumLocalDofs);

    Plato::ScalarVector tErrorVector("error vector", tTotalNumLocalDofs);

    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tPrevLocalState, 0.0, tTrialPrevLocalState);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialPrevLocalState);
        Plato::ScalarVector tVectorValueOne = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                        tLocalState, tTrialPrevLocalState,
                                                                        tControl, 0.0);
        Plato::blas1::update(1.0, tPrevLocalState, 0.0, tTrialPrevLocalState);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialPrevLocalState);
        Plato::ScalarVector tVectorValueTwo = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                        tLocalState, tTrialPrevLocalState,
                                                                        tControl, 0.0);
        Plato::blas1::update(1.0, tPrevLocalState, 0.0, tTrialPrevLocalState);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialPrevLocalState);
        Plato::ScalarVector tVectorValueThree = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                          tLocalState, tTrialPrevLocalState,
                                                                          tControl, 0.0);
        Plato::blas1::update(1.0, tPrevLocalState, 0.0, tTrialPrevLocalState);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialPrevLocalState);
        Plato::ScalarVector tVectorValueFour = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                         tLocalState, tTrialPrevLocalState,
                                                                         tControl, 0.0);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tTotalNumLocalDofs), 
                                 LAMBDA_EXPRESSION(const Plato::OrdinalType & aDofOrdinal)
        {
            Plato::Scalar tValuePlus1Eps  = tVectorValueOne(aDofOrdinal);
            Plato::Scalar tValueMinus1Eps = tVectorValueTwo(aDofOrdinal);
            Plato::Scalar tValuePlus2Eps  = tVectorValueThree(aDofOrdinal);
            Plato::Scalar tValueMinus2Eps = tVectorValueFour(aDofOrdinal);

            Plato::Scalar tNumerator = -tValuePlus2Eps + static_cast<Plato::Scalar>(8.) * tValuePlus1Eps
                                       - static_cast<Plato::Scalar>(8.) * tValueMinus1Eps + tValueMinus2Eps;
            Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
            Plato::Scalar tFiniteDiffAppx = tNumerator / tDenominator;

            Plato::Scalar tAppxError = abs(tFiniteDiffAppx - tGradientDotStep(aDofOrdinal));

            tErrorVector(aDofOrdinal) = tAppxError;

        }, "compute error");

        Plato::Scalar tL1Error = 0.0;
        Plato::blas1::local_sum(tErrorVector, tL1Error);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) 
                  << tEpsilon << std::setw(19)
                  << tL1Error << std::endl;
    }
}
// function test_partial_prev_local_state


/******************************************************************************//**
 * \brief Test partial derivative of local vector func inc with respect to the control variables
 * \param [in] aMesh mesh database
 * \param [in] aLocalVectorFuncInc local vector function inc to evaluate derivative of
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
inline void
test_partial_local_vect_func_inc_wrt_control(Omega_h::Mesh & aMesh, Plato::LocalVectorFunctionInc<SimplexPhysics> & aLocalVectorFuncInc)
{
    using StateT   = typename EvaluationType::StateScalarType;
    using LocalStateT   = typename EvaluationType::LocalStateScalarType;
    using ConfigT  = typename EvaluationType::ConfigScalarType;
    using ResultT  = typename EvaluationType::ResultScalarType;
    using ControlT = typename EvaluationType::ControlScalarType;

    const Plato::OrdinalType tNumCells = aMesh.nelems();
    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysics::mNumDofsPerNode;
    constexpr Plato::OrdinalType tLocalDofsPerCell = SimplexPhysics::mNumLocalDofsPerCell;

    // Create control workset
    const Plato::OrdinalType tNumVerts = aMesh.nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    auto tHostControl = Kokkos::create_mirror(tControl);
    Plato::blas1::random(0.5, 0.75, tHostControl);
    Kokkos::deep_copy(tControl, tHostControl);

    // Create global state workset
    const Plato::OrdinalType tTotalNumDofs = tNumVerts * tDofsPerNode;
    Plato::ScalarVector tGlobalState("Global State", tTotalNumDofs);
    auto tHostGlobalState = Kokkos::create_mirror(tGlobalState);
    Plato::blas1::random(1, 5, tHostGlobalState);
    Kokkos::deep_copy(tGlobalState, tHostGlobalState);

    // Create previous global state workset
    Plato::ScalarVector tPrevGlobalState("Previous Global State", tTotalNumDofs);
    auto tHostPrevGlobalState = Kokkos::create_mirror(tPrevGlobalState);
    Plato::blas1::random(1, 5, tHostPrevGlobalState);
    Kokkos::deep_copy(tPrevGlobalState, tHostPrevGlobalState);

    // Create local state workset
    const Plato::OrdinalType tTotalNumLocalDofs = tNumCells * tLocalDofsPerCell;
    Plato::ScalarVector tLocalState("Local State", tTotalNumLocalDofs);
    auto tHostLocalState = Kokkos::create_mirror(tLocalState);
    Plato::blas1::random(1.0, 2.0, tHostLocalState);
    Kokkos::deep_copy(tLocalState, tHostLocalState);

    // Create previous local state workset
    Plato::ScalarVector tPrevLocalState("Previous Local State", tTotalNumLocalDofs);
    auto tHostPrevLocalState = Kokkos::create_mirror(tPrevLocalState);
    Plato::blas1::random(0.1, 0.9, tHostPrevLocalState);
    Kokkos::deep_copy(tPrevLocalState, tHostPrevLocalState);

    Plato::ScalarArray3D tPartialZ =
            aLocalVectorFuncInc.gradient_z(tGlobalState, tPrevGlobalState,
                                           tLocalState, tPrevLocalState,
                                           tControl, 0.0 /*time step*/);

    constexpr Plato::OrdinalType tSpaceDim   = EvaluationType::SpatialDim;
    constexpr Plato::OrdinalType tNumControl = EvaluationType::NumControls;;
    Plato::VectorEntryOrdinal<tSpaceDim, tNumControl> tEntryOrdinal(&aMesh);

    Plato::ScalarVector tStep("control step", tNumVerts); 
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.05, 0.1, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);
    Plato::ScalarVector tGradientDotStep =
        Plato::control_workset_matrix_vector_multiply(tPartialZ, tStep, tEntryOrdinal, tTotalNumLocalDofs);

    std::cout << std::right << std::setw(14) << "\nStep Size" 
              << std::setw(20) << "abs(Error)" << std::endl;

    constexpr Plato::OrdinalType tSuperscriptLowerBound = 1;
    constexpr Plato::OrdinalType tSuperscriptUpperBound = 6;
    Plato::ScalarVector tTrialControl("trial control", tNumVerts);

    Plato::ScalarVector tErrorVector("error vector", tTotalNumLocalDofs);

    for(Plato::OrdinalType tIndex = tSuperscriptLowerBound; tIndex <= tSuperscriptUpperBound; tIndex++)
    {
        Plato::Scalar tEpsilon = tEpsilon = static_cast<Plato::Scalar>(1) /
                std::pow(static_cast<Plato::Scalar>(10), tIndex);
        // four point finite difference approximation
        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(tEpsilon, tStep, 1.0, tTrialControl);
        Plato::ScalarVector tVectorValueOne = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                        tLocalState, tPrevLocalState,
                                                                        tTrialControl, 0.0);
        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-tEpsilon, tStep, 1.0, tTrialControl);
        Plato::ScalarVector tVectorValueTwo = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                        tLocalState, tPrevLocalState,
                                                                        tTrialControl, 0.0);
        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        Plato::ScalarVector tVectorValueThree = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                          tLocalState, tPrevLocalState,
                                                                          tTrialControl, 0.0);
        Plato::blas1::update(1.0, tControl, 0.0, tTrialControl);
        Plato::blas1::update(-2.0 * tEpsilon, tStep, 1.0, tTrialControl);
        Plato::ScalarVector tVectorValueFour = aLocalVectorFuncInc.value(tGlobalState, tPrevGlobalState,
                                                                         tLocalState, tPrevLocalState,
                                                                         tTrialControl, 0.0);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tTotalNumLocalDofs), 
                                 LAMBDA_EXPRESSION(const Plato::OrdinalType & aDofOrdinal)
        {
            Plato::Scalar tValuePlus1Eps  = tVectorValueOne(aDofOrdinal);
            Plato::Scalar tValueMinus1Eps = tVectorValueTwo(aDofOrdinal);
            Plato::Scalar tValuePlus2Eps  = tVectorValueThree(aDofOrdinal);
            Plato::Scalar tValueMinus2Eps = tVectorValueFour(aDofOrdinal);

            Plato::Scalar tNumerator = -tValuePlus2Eps + static_cast<Plato::Scalar>(8.) * tValuePlus1Eps
                                       - static_cast<Plato::Scalar>(8.) * tValueMinus1Eps + tValueMinus2Eps;
            Plato::Scalar tDenominator = static_cast<Plato::Scalar>(12.) * tEpsilon;
            Plato::Scalar tFiniteDiffAppx = tNumerator / tDenominator;

            Plato::Scalar tAppxError = abs(tFiniteDiffAppx - tGradientDotStep(aDofOrdinal));

            tErrorVector(aDofOrdinal) = tAppxError;

        }, "compute error");

        Plato::Scalar tL1Error = 0.0;
        Plato::blas1::local_sum(tErrorVector, tL1Error);

        std::cout << std::right << std::scientific << std::setprecision(8) << std::setw(14) 
                  << tEpsilon << std::setw(19)
                  << tL1Error << std::endl;
    }
}
// function test_partial_local_state

}

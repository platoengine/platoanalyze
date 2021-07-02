/*
 * ElastoPlasticityTest.cpp
 *
 *  Created on: Sep 30, 2019
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoUtilities.hpp"
#include "PlatoTestHelpers.hpp"
#include "Plato_Diagnostics.hpp"

#include "PlasticityProblem.hpp"
#include "SimplexStabilizedMechanics.hpp"
#include "StabilizedElastostaticResidual.hpp"
#include "ComputeDeviatoricStrain.hpp"
#include "TimeData.hpp"

#include <Teuchos_XMLParameterListCoreHelpers.hpp>


namespace ElastoPlasticityTests
{


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_NewtonRaphsonStoppingCriterion)
{
    auto tCriterion = Plato::newton_raphson_stopping_criterion("absolute residual norm");
    TEST_EQUALITY(tCriterion, Plato::NewtonRaphson::ABSOLUTE_RESIDUAL_NORM);

    tCriterion = Plato::newton_raphson_stopping_criterion("relative residual norm");
    TEST_EQUALITY(tCriterion, Plato::NewtonRaphson::RELATIVE_RESIDUAL_NORM);

    TEST_THROW(Plato::newton_raphson_stopping_criterion("absolute displacement norm"), std::runtime_error);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputePrincipalStresses2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // Set configuration workset
    auto tNumCells = tMesh->nelems();
    using PhysicsT = Plato::SimplexPlasticity<tSpaceDim>;
    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<EvalType::ConfigScalarType> tConfigWS("configuration", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Set global, local state, and control worksets
    auto tNumNodes = tMesh->nverts();
    decltype(tNumNodes) tNumState = 3;
    auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    Plato::ScalarVectorT<EvalType::StateScalarType> tGlobalState("state", tNumState * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVector tGlobalStateWS("current state", tNumCells, PhysicsT::mNumDofsPerCell);
    tWorksetBase.worksetState(tGlobalState, tGlobalStateWS);

    Plato::ScalarMultiVectorT<EvalType::LocalStateScalarType> tLocalStateWS("local state", tNumCells, PhysicsT::mNumLocalDofsPerCell);
    Plato::ScalarMultiVectorT<EvalType::ControlScalarType> tControlWS("control", tNumCells, PhysicsT::mNumNodesPerCell);
    Kokkos::deep_copy(tControlWS, 1.0);

    // Compute principal stresses
    Plato::ScalarMultiVectorT<EvalType::ResultScalarType> tPrincipalStressWS("principal stresses", tNumCells, tSpaceDim);
    Plato::ComputePrincipalStresses<EvalType, PhysicsT> tComputePrincipalStresses;
    tComputePrincipalStresses.setBulkModulus(4);
    tComputePrincipalStresses.setShearModulus(1);
    tComputePrincipalStresses(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tPrincipalStressWS);

    // Test results
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{1.140440e-06, -2.737734e-07}, {9.837722e-07, 1.616228e-06}};
    auto tHostPrincipalStressWS = Kokkos::create_mirror(tPrincipalStressWS);
    Kokkos::deep_copy(tHostPrincipalStressWS, tPrincipalStressWS);
    for (size_t tCell = 0; tCell < tNumCells; tCell++)
    {
        for (size_t tDim = 0; tDim < tSpaceDim; tDim++)
        {
            //printf("%e\n", tHostPrincipalStressWS(tCell, tDim));
            TEST_FLOATING_EQUALITY(tHostPrincipalStressWS(tCell, tDim), tGold[tCell][tDim], tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputePrincipalStresses3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // Set configuration workset
    auto tNumCells = tMesh->nelems();
    using PhysicsT = Plato::SimplexPlasticity<tSpaceDim>;
    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<EvalType::ConfigScalarType> tConfigWS("configuration", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Set global, local state, and control worksets
    auto tNumNodes = tMesh->nverts();
    decltype(tNumNodes) tNumState = 4;
    auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    Plato::ScalarVectorT<EvalType::StateScalarType> tGlobalState("state", tNumState * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // disp_z
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVector tGlobalStateWS("current state", tNumCells, PhysicsT::mNumDofsPerCell);
    tWorksetBase.worksetState(tGlobalState, tGlobalStateWS);

    Plato::ScalarMultiVectorT<EvalType::LocalStateScalarType> tLocalStateWS("local state", tNumCells, PhysicsT::mNumLocalDofsPerCell);
    Plato::ScalarMultiVectorT<EvalType::ControlScalarType> tControlWS("control", tNumCells, PhysicsT::mNumNodesPerCell);
    Kokkos::deep_copy(tControlWS, 1.0);

    // Compute principal stresses
    Plato::ScalarMultiVectorT<EvalType::ResultScalarType> tPrincipalStressWS("principal stresses", tNumCells, tSpaceDim);
    Plato::ComputePrincipalStresses<EvalType, PhysicsT> tComputePrincipalStresses;
    tComputePrincipalStresses.setBulkModulus(4);
    tComputePrincipalStresses.setShearModulus(1);
    tComputePrincipalStresses(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tPrincipalStressWS);

    // Test results
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {
         {6.000000e-06,8.240967e-06,5.759033e-06},
         {5.333333e-06,7.564284e-06,4.302383e-06},
         {2.892366e-06,3.333333e-06,5.374301e-06},
         {3.210889e-06,6.666667e-07,-6.775555e-07},
         {2.872078e-06,-3.705232e-19,-2.472078e-06},
         {1.274022e-06,-2.000000e-06,-4.474022e-06}
        };
    auto tHostPrincipalStressWS = Kokkos::create_mirror(tPrincipalStressWS);
    Kokkos::deep_copy(tHostPrincipalStressWS, tPrincipalStressWS);
    for (size_t tCell = 0; tCell < tNumCells; tCell++)
    {
        for (size_t tDim = 0; tDim < tSpaceDim; tDim++)
        {
            //printf("%e\n", tHostPrincipalStressWS(tCell, tDim));
            TEST_FLOATING_EQUALITY(tHostPrincipalStressWS(tCell, tDim), tGold[tCell][tDim], tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_DeviatoricStress1D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 1;
    constexpr Plato::OrdinalType tNumStressTerms = 1;

    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStressTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostElasticStrain(i, j) = (i + 1.0) * (j + 1.0);
            //printf("ElasticStrain(%d,%d) = %f\n", i,j,tHostElasticStrain(i, j));
        }
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);

    Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress", tNumCells, tNumStressTerms);

    Plato::ComputeDeviatoricStress<tSpaceDim> tComputeDeviatoricStress;

    Plato::Scalar tShearModulus = 3.5;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeDeviatoricStress(tCellOrdinal, tShearModulus, tElasticStrain, tDeviatoricStress);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {4.666666667};
    auto tHostDeviatoricStress = Kokkos::create_mirror(tDeviatoricStress);
    Kokkos::deep_copy(tHostDeviatoricStress, tDeviatoricStress);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostDeviatoricStress(tCellIndex, tDofIndex), tGold[tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_DeviatoricStress2D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumStressTerms = 4;

    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStressTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostElasticStrain(i, j) = (i + 1.0) * (j + 1.0);
            //printf("ElasticStrain(%d,%d) = %f\n", i,j,tHostElasticStrain(i, j));
        }
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);

    Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress", tNumCells, tNumStressTerms);

    Plato::ComputeDeviatoricStress<tSpaceDim> tComputeDeviatoricStress;

    Plato::Scalar tShearModulus = 3.5;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeDeviatoricStress(tCellOrdinal, tShearModulus, tElasticStrain, tDeviatoricStress);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {-9.33333,-2.33333,10.5,11.6667};
    auto tHostDeviatoricStress = Kokkos::create_mirror(tDeviatoricStress);
    Kokkos::deep_copy(tHostDeviatoricStress, tDeviatoricStress);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostDeviatoricStress(tCellIndex, tDofIndex), tGold[tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_DeviatoricStress3D)
{
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumStressTerms = 6;

    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStressTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    for (unsigned int i = 0; i < tNumCells; ++i)
        for (unsigned int j = 0; j < tNumStressTerms; ++j)
        {
            tHostElasticStrain(i, j) = (i + 1.0) * (j + 1.0);
            //printf("(%d,%d) = %f\n", i,j,tHostElasticStrain(i, j));
        }
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);

    Plato::ScalarMultiVector tDeviatoricStress("deviatoric stress", tNumCells, tNumStressTerms);

    Plato::ComputeDeviatoricStress<tSpaceDim> tComputeDeviatoricStress;

    Plato::Scalar tShearModulus = 3.5;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeDeviatoricStress(tCellOrdinal, tShearModulus, tElasticStrain, tDeviatoricStress);
    }, "Unit Test");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {-7.0,0.0,7.0,14.0,17.5,21.0};
    auto tHostDeviatoricStress = Kokkos::create_mirror(tDeviatoricStress);
    Kokkos::deep_copy(tHostDeviatoricStress, tDeviatoricStress);

    Plato::OrdinalType tCellIndex = 0;
    for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumStressTerms; ++tDofIndex)
        TEST_FLOATING_EQUALITY(tHostDeviatoricStress(tCellIndex, tDofIndex), tGold[tDofIndex], tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_computeCauchyStress3D)
{
    //1. SET DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumStressTerms = 6;

    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStressTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumStressTerms; tDofIndex++)
        {
            tHostElasticStrain(tCellIndex, tDofIndex) = (tCellIndex + 1.0) * (tDofIndex + 1.0);
            //printf("ElasticStrain(%d,%d) = %f\n", tCellIndex,tDofIndex, tHostElasticStrain(tCellIndex, tDofIndex));
        }
    }
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);

    // 2. CALL FUNCTION
    constexpr Plato::Scalar tBulkModulus = 25;
    constexpr Plato::Scalar tShearModulus = 3;
    Plato::ComputeCauchyStress<tSpaceDim> tComputeCauchyStress;
    Plato::ScalarMultiVector tCauchyStress("cauchy stress", tNumCells, tNumStressTerms);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeCauchyStress(tCellOrdinal, tBulkModulus, tShearModulus, tElasticStrain, tCauchyStress);
    }, "Unit Test");

    // 3. TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{167,173,179,12,15,18}, {334,346,358,24,30,36}};
    auto tHostCauchyStress = Kokkos::create_mirror(tCauchyStress);
    Kokkos::deep_copy(tHostCauchyStress, tCauchyStress);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumStressTerms; tDofIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostCauchyStress(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
            //printf("HostCauchyStress(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostCauchyStress(tCellIndex, tDofIndex));
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_computeCauchyStress2D)
{
    //1. SET DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumStressTerms = 4;

    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStressTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumStressTerms; tDofIndex++)
        {
            tHostElasticStrain(tCellIndex, tDofIndex) = (tCellIndex + 1.0) * (tDofIndex + 1.0);
            //printf("ElasticStrain(%d,%d) = %f\n", tCellIndex,tDofIndex, tHostElasticStrain(tCellIndex, tDofIndex));
        }
    }
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);

    // 2. CALL FUNCTION
    constexpr Plato::Scalar tBulkModulus = 25;
    constexpr Plato::Scalar tShearModulus = 3;
    Plato::ComputeCauchyStress<tSpaceDim> tComputeCauchyStress;
    Plato::ScalarMultiVector tCauchyStress("cauchy stress", tNumCells, tNumStressTerms);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeCauchyStress(tCellOrdinal, tBulkModulus, tShearModulus, tElasticStrain, tCauchyStress);
    }, "Unit Test");

    // 3. TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{167,173,9,185}, {334,346,18,370}};
    auto tHostCauchyStress = Kokkos::create_mirror(tCauchyStress);
    Kokkos::deep_copy(tHostCauchyStress, tCauchyStress);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumStressTerms; tDofIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostCauchyStress(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
            //printf("HostCauchyStress(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostCauchyStress(tCellIndex, tDofIndex));
        }
    }
}



TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_computeCauchyStress1D)
{
    //1. SET DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDim = 1;
    constexpr Plato::OrdinalType tNumStressTerms = 1;

    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStressTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumStressTerms; tDofIndex++)
        {
            tHostElasticStrain(tCellIndex, tDofIndex) = (tCellIndex + 1.0) * (tDofIndex + 1.0);
            //printf("ElasticStrain(%d,%d) = %f\n", tCellIndex,tDofIndex, tHostElasticStrain(tCellIndex, tDofIndex));
        }
    }
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);

    // 2. CALL FUNCTION
    constexpr Plato::Scalar tBulkModulus = 25;
    constexpr Plato::Scalar tShearModulus = 3;
    Plato::ComputeCauchyStress<tSpaceDim> tComputeCauchyStress;
    Plato::ScalarMultiVector tCauchyStress("cauchy stress", tNumCells, tNumStressTerms);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeCauchyStress(tCellOrdinal, tBulkModulus, tShearModulus, tElasticStrain, tCauchyStress);
    }, "Unit Test");

    // 3. TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{12.33333333}, {24.66666667}};
    auto tHostCauchyStress = Kokkos::create_mirror(tCauchyStress);
    Kokkos::deep_copy(tHostCauchyStress, tCauchyStress);
    for (unsigned int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (unsigned int tDofIndex = 0; tDofIndex < tNumStressTerms; tDofIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostCauchyStress(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
            //printf("HostCauchyStress(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostCauchyStress(tCellIndex, tDofIndex));
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeDeviatoricStrain_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::ComputeDeviatoricStrain<tSpaceDim> tComputeDeviatoricStrain;

    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumStrainTerms = 6;
    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStrainTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    tHostElasticStrain(0,0) = 1; tHostElasticStrain(0,1) = 2; tHostElasticStrain(0,2) = 3;
    tHostElasticStrain(0,3) = 4; tHostElasticStrain(0,4) = 5; tHostElasticStrain(0,5) = 6;
    tHostElasticStrain(1,0) = 7;  tHostElasticStrain(1,1) = 8;  tHostElasticStrain(1,2) = 9;
    tHostElasticStrain(1,3) = 10; tHostElasticStrain(1,4) = 11; tHostElasticStrain(1,5) = 12;
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);
    Plato::ScalarMultiVector tDeviatoricStrain("elastic strain", tNumCells, tNumStrainTerms);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeDeviatoricStrain(aCellIndex, tElasticStrain, tDeviatoricStrain);
    }, "test compute deviatoric strain functor");

    const Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = { {-1.0, 0.0, 1.0, 4.0, 5.0, 6.0}, {-1.0, 0.0, 1.0, 10.0, 11.0, 12.0} };
    auto tHostDeviatoricStrain = Kokkos::create_mirror(tDeviatoricStrain);
    Kokkos::deep_copy(tHostDeviatoricStrain, tDeviatoricStrain);
    for (size_t tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (size_t tDofIndex = 0; tDofIndex < tNumStrainTerms; tDofIndex++)
        {
            //printf("(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostDeviatoricStrain(tCellIndex, tDofIndex));
            TEST_FLOATING_EQUALITY(tHostDeviatoricStrain(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeDeviatoricStrain_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::ComputeDeviatoricStrain<tSpaceDim> tComputeDeviatoricStrain;

    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumStrainTerms = 4;
    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStrainTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    tHostElasticStrain(0,0) = 1; tHostElasticStrain(0,1) = 2; tHostElasticStrain(0,2) = 3; tHostElasticStrain(0,3) = 4;
    tHostElasticStrain(1,0) = 5; tHostElasticStrain(1,1) = 6; tHostElasticStrain(1,2) = 7; tHostElasticStrain(1,3) = 8;
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);
    Plato::ScalarMultiVector tDeviatoricStrain("elastic strain", tNumCells, tNumStrainTerms);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeDeviatoricStrain(aCellIndex, tElasticStrain, tDeviatoricStrain);
    }, "test compute deviatoric strain functor");

    const Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = { {-1.333333, -0.333333, 3.0, 1.666667}, {-1.333333, -0.333333, 7.0, 1.666667} };
    auto tHostDeviatoricStrain = Kokkos::create_mirror(tDeviatoricStrain);
    Kokkos::deep_copy(tHostDeviatoricStrain, tDeviatoricStrain);
    for (size_t tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (size_t tDofIndex = 0; tDofIndex < tNumStrainTerms; tDofIndex++)
        {
            //printf("(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostDeviatoricStrain(tCellIndex, tDofIndex));
            TEST_FLOATING_EQUALITY(tHostDeviatoricStrain(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeDeviatoricStrain_1D)
{
    constexpr Plato::OrdinalType tSpaceDim = 1;
    Plato::ComputeDeviatoricStrain<tSpaceDim> tComputeDeviatoricStrain;

    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumStrainTerms = 1;
    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStrainTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    tHostElasticStrain(0,0) = 1;
    tHostElasticStrain(1,0) = 2;
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);
    Plato::ScalarMultiVector tDeviatoricStrain("elastic strain", tNumCells, tNumStrainTerms);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeDeviatoricStrain(aCellIndex, tElasticStrain, tDeviatoricStrain);
    }, "test compute deviatoric strain functor");

    const Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = { {0.666667}, {1.333333} };
    auto tHostDeviatoricStrain = Kokkos::create_mirror(tDeviatoricStrain);
    Kokkos::deep_copy(tHostDeviatoricStrain, tDeviatoricStrain);
    for (size_t tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (size_t tDofIndex = 0; tDofIndex < tNumStrainTerms; tDofIndex++)
        {
            //printf("(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostDeviatoricStrain(tCellIndex, tDofIndex));
            TEST_FLOATING_EQUALITY(tHostDeviatoricStrain(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeElasticWork_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::ComputeElasticWork<tSpaceDim> tComputeElasticWork;
    Plato::ComputeDeviatoricStrain<tSpaceDim> tComputeDeviatoricStrain;

    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumStrainTerms = 6;
    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStrainTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    tHostElasticStrain(0,0) = 1; tHostElasticStrain(0,1) = 2; tHostElasticStrain(0,2) = 3;
    tHostElasticStrain(0,3) = 4; tHostElasticStrain(0,4) = 5; tHostElasticStrain(0,5) = 6;
    tHostElasticStrain(1,0) = 7;  tHostElasticStrain(1,1) = 8;  tHostElasticStrain(1,2) = 9;
    tHostElasticStrain(1,3) = 10; tHostElasticStrain(1,4) = 11; tHostElasticStrain(1,5) = 12;
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);
    Plato::ScalarMultiVector tDeviatoricStrain("elastic strain", tNumCells, tNumStrainTerms);

    constexpr Plato::Scalar tBulkModulus = 2;
    constexpr Plato::Scalar tShearModulus = 0.5;
    Plato::ScalarVector tElasticWork("elastic work", tNumCells);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeDeviatoricStrain(aCellIndex, tElasticStrain, tDeviatoricStrain);
        tComputeElasticWork(aCellIndex, tShearModulus, tBulkModulus, tElasticStrain, tDeviatoricStrain, tElasticWork);
    }, "test compute deviatoric strain functor");

    const Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {114, 942};
    auto tHostElasticWork = Kokkos::create_mirror(tElasticWork);
    Kokkos::deep_copy(tHostElasticWork, tElasticWork);
    for (size_t tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        //printf("(%d) = %f\n", tCellIndex, tHostElasticWork(tCellIndex));
        TEST_FLOATING_EQUALITY(tHostElasticWork(tCellIndex), tGold[tCellIndex], tTolerance);
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeElasticWork_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::ComputeElasticWork<tSpaceDim> tComputeElasticWork;
    Plato::ComputeDeviatoricStrain<tSpaceDim> tComputeDeviatoricStrain;

    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumStrainTerms = 4;
    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStrainTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    tHostElasticStrain(0,0) = 1; tHostElasticStrain(0,1) = 2; tHostElasticStrain(0,2) = 3; tHostElasticStrain(0,3) = 4;
    tHostElasticStrain(1,0) = 5; tHostElasticStrain(1,1) = 6; tHostElasticStrain(1,2) = 7; tHostElasticStrain(1,3) = 8;
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);
    Plato::ScalarMultiVector tDeviatoricStrain("elastic strain", tNumCells, tNumStrainTerms);

    constexpr Plato::Scalar tBulkModulus = 2;
    constexpr Plato::Scalar tShearModulus = 0.5;
    Plato::ScalarVector tElasticWork("elastic work", tNumCells);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeDeviatoricStrain(aCellIndex, tElasticStrain, tDeviatoricStrain);
        tComputeElasticWork(aCellIndex, tShearModulus, tBulkModulus, tElasticStrain, tDeviatoricStrain, tElasticWork);
    }, "test compute deviatoric strain functor");

    const Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {60.333333, 412.333333};
    auto tHostElasticWork = Kokkos::create_mirror(tElasticWork);
    Kokkos::deep_copy(tHostElasticWork, tElasticWork);
    for (size_t tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        //printf("(%d) = %f\n", tCellIndex, tHostElasticWork(tCellIndex));
        TEST_FLOATING_EQUALITY(tHostElasticWork(tCellIndex), tGold[tCellIndex], tTolerance);
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeElasticWork_1D)
{
    constexpr Plato::OrdinalType tSpaceDim = 1;
    Plato::ComputeElasticWork<tSpaceDim> tComputeElasticWork;
    Plato::ComputeDeviatoricStrain<tSpaceDim> tComputeDeviatoricStrain;

    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumStrainTerms = 1;
    Plato::ScalarMultiVector tElasticStrain("elastic strain", tNumCells, tNumStrainTerms);
    auto tHostElasticStrain = Kokkos::create_mirror(tElasticStrain);
    tHostElasticStrain(0,0) = 1;
    tHostElasticStrain(1,0) = 2;
    Kokkos::deep_copy(tElasticStrain, tHostElasticStrain);
    Plato::ScalarMultiVector tDeviatoricStrain("elastic strain", tNumCells, tNumStrainTerms);

    constexpr Plato::Scalar tBulkModulus = 2;
    constexpr Plato::Scalar tShearModulus = 0.5;
    Plato::ScalarVector tElasticWork("elastic work", tNumCells);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeDeviatoricStrain(aCellIndex, tElasticStrain, tDeviatoricStrain);
        tComputeElasticWork(aCellIndex, tShearModulus, tBulkModulus, tElasticStrain, tDeviatoricStrain, tElasticWork);
    }, "test compute deviatoric strain functor");

    const Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {1.222222, 4.888889};
    auto tHostElasticWork = Kokkos::create_mirror(tElasticWork);
    Kokkos::deep_copy(tHostElasticWork, tElasticWork);
    for (size_t tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        //printf("(%d) = %f\n", tCellIndex, tHostElasticWork(tCellIndex));
        TEST_FLOATING_EQUALITY(tHostElasticWork(tCellIndex), tGold[tCellIndex], tTolerance);
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_FlattenVectorWorkset_Errors)
{
    // CALL FUNCTION - TEST tLocalStateWorset IS EMPTY
    Plato::ScalarVector tAssembledLocalState;
    Plato::ScalarMultiVector tLocalStateWorset;
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    TEST_THROW(Plato::flatten_vector_workset<tNumLocalDofsPerCell>(tNumCells, tLocalStateWorset, tAssembledLocalState), std::runtime_error);

    // CALL FUNCTION - TEST tAssembledLocalState IS EMPTY
    tLocalStateWorset = Plato::ScalarMultiVector("local state WS", tNumCells, tNumLocalDofsPerCell);
    TEST_THROW(Plato::flatten_vector_workset<tNumLocalDofsPerCell>(tNumCells, tLocalStateWorset, tAssembledLocalState), std::runtime_error);

    // CALL FUNCTION - TEST NUMBER OF CELLS IS EMPTY
    constexpr Plato::OrdinalType tEmptyNumCells = 0;
    tAssembledLocalState = Plato::ScalarVector("assembled local state", tNumCells * tNumLocalDofsPerCell);
    TEST_THROW(Plato::flatten_vector_workset<tNumLocalDofsPerCell>(tEmptyNumCells, tLocalStateWorset, tAssembledLocalState), std::runtime_error);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_FlattenVectorWorkset)
{
    // PREPARE DATA
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 14;
    Plato::ScalarMultiVector tLocalStateWorset("local state WS", tNumCells, tNumLocalDofsPerCell);
    auto tHostLocalStateWorset = Kokkos::create_mirror(tLocalStateWorset);

    for (size_t tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (size_t tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; tDofIndex++)
        {
            tHostLocalStateWorset(tCellIndex, tDofIndex) = (tNumLocalDofsPerCell * tCellIndex) + (tDofIndex + 1.0);
            //printf("(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostLocalStateWorset(tCellIndex, tDofIndex));
        }
    }
    Kokkos::deep_copy(tLocalStateWorset, tHostLocalStateWorset);

    Plato::ScalarVector tAssembledLocalState("assembled local state", tNumCells * tNumLocalDofsPerCell);

    // CALL FUNCTION
    TEST_NOTHROW(Plato::flatten_vector_workset<tNumLocalDofsPerCell>(tNumCells, tLocalStateWorset, tAssembledLocalState));

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostAssembledLocalState = Kokkos::create_mirror(tAssembledLocalState);
    Kokkos::deep_copy(tHostAssembledLocalState, tAssembledLocalState);
    std::vector<std::vector<Plato::Scalar>> tGold =
      {{1,2,3,4,5,6,7,8,9,10,11,12,13,14},
       {15,16,17,18,19,20,21,22,23,24,25,26,27,28},
       {29,30,31,32,33,34,35,36,37,38,39,40,41,42}};
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        const auto tDofOffset = tCellIndex * tNumLocalDofsPerCell;
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; tDofIndex++)
        {
            //printf("(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostAssembledLocalState(tDofOffset + tDofIndex));
            TEST_FLOATING_EQUALITY(tHostAssembledLocalState(tDofOffset + tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_fill3DView_Error)
{
    // PREPARE DATA
    constexpr Plato::OrdinalType tNumRows = 14;
    constexpr Plato::OrdinalType tNumCols = 14;
    constexpr Plato::OrdinalType tNumCells = 2;

    // CALL FUNCTION - TEST tMatrixWorkSet IS EMPTY
    constexpr Plato::Scalar tAlpha = 2.0;
    Plato::ScalarArray3D tMatrixWorkSet;
    TEST_THROW( (Plato::blas3::fill<tNumRows, tNumCols>(tNumCells, tAlpha, tMatrixWorkSet)), std::runtime_error );

    // CALL FUNCTION - TEST tNumCells IS ZERO
    Plato::OrdinalType tBadNumCells = 0;
    tMatrixWorkSet = Plato::ScalarArray3D("Matrix A WS", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::blas3::fill<tNumRows, tNumCols>(tBadNumCells, tAlpha, tMatrixWorkSet)), std::runtime_error );

    // CALL FUNCTION - TEST tNumCells IS NEGATIVE
    tBadNumCells = -1;
    TEST_THROW( (Plato::blas3::fill<tNumRows, tNumCols>(tBadNumCells, tAlpha, tMatrixWorkSet)), std::runtime_error );
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_fill_3D_View)
{
    // PREPARE DATA
    constexpr Plato::OrdinalType tNumRows = 14;
    constexpr Plato::OrdinalType tNumCols = 14;
    constexpr Plato::OrdinalType tNumCells = 2;
    Plato::ScalarArray3D tA("Matrix A WS", tNumCells, tNumRows, tNumCols);

    // CALL FUNCTION
    Plato::Scalar tAlpha = 2.0;
    TEST_NOTHROW( (Plato::blas3::fill<tNumRows, tNumCols>(tNumCells, tAlpha, tA)) );

    // TEST RESULTS
    constexpr Plato::Scalar tGold = 2.0;
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostA = Kokkos::create_mirror(tA);
    Kokkos::deep_copy(tHostA, tA);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                //printf("(%d,%d,%d) = %f\n", aCellOrdinal, tRowIndex, tColIndex, tHostA(tCellIndex, tRowIndex, tColIndex));
                TEST_FLOATING_EQUALITY(tHostA(tCellIndex, tRowIndex, tColIndex), tGold, tTolerance);
            }
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_fill_2D_View)
{
    // PREPARE DATA
    constexpr Plato::OrdinalType tNumRows = 14;
    constexpr Plato::OrdinalType tNumCols = 14;
    Plato::ScalarMultiVector tA("Matrix A", tNumRows, tNumCols);

    // CALL FUNCTION
    constexpr Plato::Scalar tAlpha = 2.0;
    TEST_NOTHROW( (Plato::blas2::fill(tAlpha, tA)) );

    // TEST RESULTS
    constexpr Plato::Scalar tGold = 2.0;
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostA = Kokkos::create_mirror(tA);
    Kokkos::deep_copy(tHostA, tA);
    for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
    {
        for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
        {
            //printf("(%d,%d,%d) = %f\n", aCellOrdinal, tRowIndex, tColIndex, tHostA(tCellIndex, tRowIndex, tColIndex));
            TEST_FLOATING_EQUALITY(tHostA(tRowIndex, tColIndex), tGold, tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_scale_2D_View)
{
    // PREPARE DATA
    constexpr Plato::OrdinalType tNumRows = 14;
    constexpr Plato::OrdinalType tNumCols = 14;
    Plato::ScalarMultiVector tA("Matrix A", tNumRows, tNumCols);

    // CALL FUNCTION
    constexpr Plato::Scalar tAlpha = 2.0;
    TEST_NOTHROW( (Plato::blas2::fill(tAlpha, tA)) );
    TEST_NOTHROW( (Plato::blas2::scale(tAlpha, tA)) );

    // TEST RESULTS
    constexpr Plato::Scalar tGold = 4.0;
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostA = Kokkos::create_mirror(tA);
    Kokkos::deep_copy(tHostA, tA);
    for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
    {
        for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
        {
            //printf("(%d,%d,%d) = %f\n", aCellOrdinal, tRowIndex, tColIndex, tHostA(tCellIndex, tRowIndex, tColIndex));
            TEST_FLOATING_EQUALITY(tHostA(tRowIndex, tColIndex), tGold, tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_UpdateMatrixWorkset_Error)
{
    // CALL FUNCTION - INPUT VIEW IS EMPTY
    Plato::ScalarArray3D tB;
    Plato::ScalarArray3D tA;
    constexpr Plato::OrdinalType tNumCells = 2;
    Plato::Scalar tAlpha = 1; Plato::Scalar tBeta = 3;
    TEST_THROW( (Plato::blas3::update(tNumCells, tAlpha, tA, tBeta, tB)), std::runtime_error );

    // CALL FUNCTION - OUTPUT VIEW IS EMPTY
    Plato::OrdinalType tNumRows = 4;
    Plato::OrdinalType tNumCols = 4;
    tA = Plato::ScalarArray3D("Matrix A WS", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::blas3::update(tNumCells, tAlpha, tA, tBeta, tB)), std::runtime_error );

    // CALL FUNCTION - ROW DIM MISTMATCH
    tNumRows = 3;
    Plato::ScalarArray3D tC = Plato::ScalarArray3D("Matrix C WS", tNumCells, tNumRows, tNumCols);
    tNumRows = 4;
    Plato::ScalarArray3D tD = Plato::ScalarArray3D("Matrix D WS", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::blas3::update(tNumCells, tAlpha, tC, tBeta, tD)), std::runtime_error );

    // CALL FUNCTION - COLUMN DIM MISTMATCH
    tNumCols = 5;
    Plato::ScalarArray3D tE = Plato::ScalarArray3D("Matrix E WS", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::blas3::update(tNumCells, tAlpha, tD, tBeta, tE)), std::runtime_error );

    // CALL FUNCTION - NEGATIVE NUMBER OF CELLS
    tNumRows = 4; tNumCols = 4;
    Plato::OrdinalType tBadNumCells = -1;
    tB = Plato::ScalarArray3D("Matrix B WS", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::blas3::update(tBadNumCells, tAlpha, tA, tBeta, tB)), std::runtime_error );

    // CALL FUNCTION - ZERO NUMBER OF CELLS
    tBadNumCells = 0;
    TEST_THROW( (Plato::blas3::update(tBadNumCells, tAlpha, tA, tBeta, tB)), std::runtime_error );
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_UpdateMatrixWorkset)
{
    // PREPARE DATA
    constexpr Plato::OrdinalType tNumRows = 14;
    constexpr Plato::OrdinalType tNumCols = 14;
    constexpr Plato::OrdinalType tNumCells = 2;
    Plato::ScalarArray3D tA("Matrix A WS", tNumCells, tNumRows, tNumCols);
    Plato::Scalar tAlpha = 2;
    TEST_NOTHROW( (Plato::blas3::fill<tNumRows, tNumCols>(tNumCells, tAlpha, tA)) );

    tAlpha = 1;
    Plato::ScalarArray3D tB("Matrix A WS", tNumCells, tNumRows, tNumCols);
    TEST_NOTHROW( (Plato::blas3::fill<tNumRows, tNumCols>(tNumCells, tAlpha, tB)) );

    // CALL FUNCTION
    tAlpha = 2;
    Plato::Scalar tBeta = 3;
    TEST_NOTHROW( (Plato::blas3::update(tNumCells, tAlpha, tA, tBeta, tB)) );

    // TEST RESULTS
    constexpr Plato::Scalar tGold = 7.0;
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostB = Kokkos::create_mirror(tB);
    Kokkos::deep_copy(tHostB, tB);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                //printf("(%d,%d,%d) = %f\n", aCellOrdinal, tRowIndex, tColIndex, tHostB(tCellIndex, tRowIndex, tColIndex));
                TEST_FLOATING_EQUALITY(tHostB(tCellIndex, tRowIndex, tColIndex), tGold, tTolerance);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_UpdateVectorWorkset_Error)
{
    // CALL FUNCTION - DIM(1) MISMATCH
    Plato::OrdinalType tNumDofsPerCell = 3;
    constexpr Plato::OrdinalType tNumCells = 2;
    Plato::ScalarMultiVector tVecX("vector X WS", tNumCells, tNumDofsPerCell);
    tNumDofsPerCell = 4;
    Plato::ScalarMultiVector tVecY("vector Y WS", tNumCells, tNumDofsPerCell);
    Plato::Scalar tAlpha = 1; Plato::Scalar tBeta = 3;
    TEST_THROW( (Plato::blas2::update(tAlpha, tVecX, tBeta, tVecY)), std::runtime_error );

    // CALL FUNCTION - DIM(0) MISMATCH
    Plato::OrdinalType tBadNumCells = 4;
    Plato::ScalarMultiVector tVecZ("vector Y WS", tBadNumCells, tNumDofsPerCell);
    TEST_THROW( (Plato::blas2::update(tAlpha, tVecY, tBeta, tVecZ)), std::runtime_error );
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_UpdateVectorWorkset)
{
    // PREPARE DATA
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumLocalDofsPerCell = 6;
    Plato::ScalarMultiVector tVecX("vector X WS", tNumCells, tNumLocalDofsPerCell);
    Plato::ScalarMultiVector tVecY("vector Y WS", tNumCells, tNumLocalDofsPerCell);
    auto tHostVecX = Kokkos::create_mirror(tVecX);
    auto tHostVecY = Kokkos::create_mirror(tVecY);

    for (size_t tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (size_t tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; tDofIndex++)
        {
            tHostVecX(tCellIndex, tDofIndex) = (tNumLocalDofsPerCell * tCellIndex) + (tDofIndex + 1.0);
            tHostVecY(tCellIndex, tDofIndex) = (tNumLocalDofsPerCell * tCellIndex) + (tDofIndex + 1.0);
            //printf("X(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostVecX(tCellIndex, tDofIndex));
            //printf("Y(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostVecY(tCellIndex, tDofIndex));
        }
    }
    Kokkos::deep_copy(tVecX, tHostVecX);
    Kokkos::deep_copy(tVecY, tHostVecY);

    // CALL FUNCTION
    Plato::Scalar tAlpha = 1; Plato::Scalar tBeta = 2;
    TEST_NOTHROW( (Plato::blas2::update(tAlpha, tVecX, tBeta, tVecY)) );

    // TEST OUTPUT
    constexpr Plato::Scalar tTolerance = 1e-4;
    tHostVecY = Kokkos::create_mirror(tVecY);
    Kokkos::deep_copy(tHostVecY, tVecY);
    std::vector<std::vector<Plato::Scalar>> tGold =
      {{3, 6, 9, 12, 15, 18}, {21, 24, 27, 30, 33, 36}};
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumLocalDofsPerCell; tDofIndex++)
        {
            //printf("(%d,%d) = %f\n", tCellIndex, tDofIndex, tHostVecY(tCellIndex, tDofIndex));
            TEST_FLOATING_EQUALITY(tHostVecY(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_MultiplyMatrixWorkset_Error)
{
    // PREPARE DATA
    Plato::ScalarArray3D tA;
    Plato::ScalarArray3D tB;
    Plato::ScalarArray3D tC;

    // CALL FUNCTION - A IS EMPTY
    constexpr Plato::OrdinalType tNumCells = 2;
    Plato::Scalar tAlpha = 1; Plato::Scalar tBeta = 1;
    TEST_THROW( (Plato::blas3::multiply(tNumCells, tAlpha, tA, tB, tBeta, tC)), std::runtime_error );

    // CALL FUNCTION - B IS EMPTY
    constexpr Plato::OrdinalType tNumRows = 4;
    constexpr Plato::OrdinalType tNumCols = 4;
    tA = Plato::ScalarArray3D("Matrix A", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::blas3::multiply(tNumCells, tAlpha, tA, tB, tBeta, tC)), std::runtime_error );

    // CALL FUNCTION - C IS EMPTY
    tB = Plato::ScalarArray3D("Matrix B", tNumCells, tNumRows + 1, tNumCols);
    TEST_THROW( (Plato::blas3::multiply(tNumCells, tAlpha, tA, tB, tBeta, tC)), std::runtime_error );

    // CALL FUNCTION - NUM ROWS/COLUMNS MISMATCH IN INPUT MATRICES
    tC = Plato::ScalarArray3D("Matrix C", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::blas3::multiply(tNumCells, tAlpha, tA, tB, tBeta, tC)), std::runtime_error );

    // CALL FUNCTION - NUM ROWS MISMATCH IN INPUT AND OUTPUT MATRICES
    Plato::ScalarArray3D tD("Matrix D", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::blas3::multiply(tNumCells, tAlpha, tA, tD, tBeta, tB)), std::runtime_error );

    // CALL FUNCTION - NUM COLUMNS MISMATCH IN INPUT AND OUTPUT MATRICES
    Plato::ScalarArray3D tH("Matrix H", tNumCells, tNumRows, tNumCols + 1);
    TEST_THROW( (Plato::blas3::multiply(tNumCells, tAlpha, tA, tC, tBeta, tH)), std::runtime_error );

    // CALL FUNCTION - NUM CELLS MISMATCH IN A
    Plato::ScalarArray3D tE("Matrix E", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::blas3::multiply(tNumCells + 1, tAlpha, tA, tD, tBeta, tE)), std::runtime_error );

    // CALL FUNCTION - NUM CELLS MISMATCH IN F
    Plato::ScalarArray3D tF("Matrix F", tNumCells + 1, tNumRows, tNumCols);
    TEST_THROW( (Plato::blas3::multiply(tNumCells, tAlpha, tA, tF, tBeta, tE)), std::runtime_error );

    // CALL FUNCTION - NUM CELLS MISMATCH IN E
    Plato::ScalarArray3D tG("Matrix G", tNumCells + 1, tNumRows, tNumCols);
    TEST_THROW( (Plato::blas3::multiply(tNumCells, tAlpha, tA, tD, tBeta, tG)), std::runtime_error );
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_MultiplyMatrixWorkset_One)
{
    // PREPARE DATA FOR TEST ONE
    constexpr Plato::OrdinalType tNumRows = 4;
    constexpr Plato::OrdinalType tNumCols = 4;
    constexpr Plato::OrdinalType tNumCells = 3;
    Plato::ScalarArray3D tA("Matrix A WS", tNumCells, tNumRows, tNumCols);
    Plato::Scalar tAlpha = 2;
    TEST_NOTHROW( (Plato::blas3::fill<tNumRows, tNumCols>(tNumCells, tAlpha, tA)) );
    Plato::ScalarArray3D tB("Matrix B WS", tNumCells, tNumRows, tNumCols);
    tAlpha = 1;
    TEST_NOTHROW( (Plato::blas3::fill<tNumRows, tNumCols>(tNumCells, tAlpha, tB)) );
    Plato::ScalarArray3D tC("Matrix C WS", tNumCells, tNumRows, tNumCols);
    tAlpha = 3;
    TEST_NOTHROW( (Plato::blas3::fill<tNumRows, tNumCols>(tNumCells, tAlpha, tC)) );

    // CALL FUNCTION
    Plato::Scalar tBeta = 1;
    TEST_NOTHROW( (Plato::blas3::multiply(tNumCells, tAlpha, tA, tB, tBeta, tC)) );

    // TEST RESULTS
    constexpr Plato::Scalar tGold = 27.0;
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostC = Kokkos::create_mirror(tC);
    Kokkos::deep_copy(tHostC, tC);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                //printf("(%d,%d,%d) = %f\n", tCellIndex, tRowIndex, tColIndex, tHostC(tCellIndex, tRowIndex, tColIndex));
                TEST_FLOATING_EQUALITY(tHostC(tCellIndex, tRowIndex, tColIndex), tGold, tTolerance);
            }
        }
    }

    // PREPARE DATA FOR TEST TWO
    constexpr Plato::OrdinalType tNumRows2 = 3;
    constexpr Plato::OrdinalType tNumCols2 = 3;
    Plato::ScalarArray3D tD("Matrix D WS", tNumCells, tNumRows2, tNumCols2);
    Plato::ScalarArray3D tE("Matrix E WS", tNumCells, tNumRows2, tNumCols2);
    Plato::ScalarArray3D tF("Matrix F WS", tNumCells, tNumRows2, tNumCols2);
    std::vector<std::vector<Plato::Scalar>> tData = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    auto tHostD = Kokkos::create_mirror(tD);
    auto tHostE = Kokkos::create_mirror(tE);
    auto tHostF = Kokkos::create_mirror(tF);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows2; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols2; tColIndex++)
            {
                tHostD(tCellIndex, tRowIndex, tColIndex) = tData[tRowIndex][tColIndex];
                tHostE(tCellIndex, tRowIndex, tColIndex) = tData[tRowIndex][tColIndex];
                tHostF(tCellIndex, tRowIndex, tColIndex) = tData[tRowIndex][tColIndex];
            }
        }
    }
    Kokkos::deep_copy(tD, tHostD);
    Kokkos::deep_copy(tE, tHostE);
    Kokkos::deep_copy(tF, tHostF);

    // CALL FUNCTION - NO TRANSPOSE
    tAlpha = 1.5; tBeta = 2.5;
    TEST_NOTHROW( (Plato::blas3::multiply(tNumCells, tAlpha, tD, tE, tBeta, tF)) );

    // 2. TEST RESULTS
    std::vector<std::vector<Plato::Scalar>> tGoldOut = { {47.5, 59, 70.5}, {109, 134, 159}, {170.5, 209, 247.5} };
    tHostF = Kokkos::create_mirror(tF);
    Kokkos::deep_copy(tHostF, tF);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows2; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols2; tColIndex++)
            {
                //printf("Result(%d,%d,%d) = %f\n", tCellIndex, tRowIndex, tColIndex, tHostF(tCellIndex, tRowIndex, tColIndex));
                TEST_FLOATING_EQUALITY(tHostF(tCellIndex, tRowIndex, tColIndex), tGoldOut[tRowIndex][tColIndex], tTolerance);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_MultiplyMatrixWorkset_Two)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 1;
    constexpr Plato::OrdinalType tNumOutCols = 9;
    constexpr Plato::OrdinalType tNumOutRows = 10;
    constexpr Plato::OrdinalType tNumInnrCols = 10;
    Plato::ScalarArray3D tA("Matrix A WS", tNumCells, tNumOutRows, tNumInnrCols);
    auto tHostA = Kokkos::create_mirror(tA);
    tHostA(0,0,0) = 0.999134832918946; tHostA(0,0,1) = -8.65167081054137e-7; tHostA(0,0,2) = -0.665513165892955; tHostA(0,0,3) = 0.332756499757352; tHostA(0,0,4) = 0;
      tHostA(0,0,5) = 0.332756499757352; tHostA(0,0,6) = -8.65167382846366e-7; tHostA(0,0,7) = 4.32583520111433e-7; tHostA(0,0,8) = 0; tHostA(0,0,9) = 4.32583520113168e-7;
    tHostA(0,1,0) = -0.000865167081054158; tHostA(0,1,1) = -8.65167081054158e-7; tHostA(0,1,2) = -0.665513165892955; tHostA(0,1,3) = 0.332756499757352; tHostA(0,1,4) = 0;
      tHostA(0,1,5) = 0.332756499757352; tHostA(0,1,6) = -8.65167382846366e-7; tHostA(0,1,7) = 4.32583520111433e-7; tHostA(0,1,8) = 0; tHostA(0,1,9) = 4.32583520111433e-7;
    tHostA(0,2,0) = -0.000865167081030844; tHostA(0,2,1) = -8.65167081030844e-7; tHostA(0,2,2) = 0.334486834124979; tHostA(0,2,3) = 0.332756499748386; tHostA(0,2,4) = 0;
      tHostA(0,2,5) = 0.332756499748385; tHostA(0,2,6) = -9.31701002265914e-7; tHostA(0,2,7) = 3.66049931096926e-7; tHostA(0,2,8) = 0; tHostA(0,2,9) = 3.66049931099094e-7;
    tHostA(0,3,0) = 0.000432583432413186; tHostA(0,3,1) = 4.32583432413186e-7; tHostA(0,3,2) = 0.332756499781941; tHostA(0,3,3) = 0.767070265244303; tHostA(0,3,4) = 0;
      tHostA(0,3,5) = -0.0998269318370781; tHostA(0,3,6) = 3.66049980498706e-7; tHostA(0,3,7) = -3.69341927428275e-7; tHostA(0,3,8) = 0; tHostA(0,3,9) = -1.96308599308918e-7;
    tHostA(0,4,0) = 0; tHostA(0,4,1) = 0; tHostA(0,4,2) = 0; tHostA(0,4,3) = 0; tHostA(0,4,4) = 0.928703624178876;
      tHostA(0,4,5) = 0; tHostA(0,4,6) = 0; tHostA(0,4,7) = 0; tHostA(0,4,8) = -1.85370035651194e-7; tHostA(0,4,9) = 0;
    tHostA(0,5,0) = 0.000432583432413187; tHostA(0,5,1) = 4.32583432413187e-7; tHostA(0,5,2) = 0.332756499781942; tHostA(0,5,3) = -0.0998269318370783; tHostA(0,5,4) = 0;
      tHostA(0,5,5) = 0.767070265244303; tHostA(0,5,6) = 3.66049980498706e-7; tHostA(0,5,7) = -1.96308599309351e-7; tHostA(0,5,8) = 0; tHostA(0,5,9) = -3.69341927426107e-07;
    tHostA(0,6,0) = -0.576778291445566; tHostA(0,6,1) = -0.000576778291445566; tHostA(0,6,2) = -443.675626551306; tHostA(0,6,3) = 221.837757816214; tHostA(0,6,4) = 0;
      tHostA(0,6,5) = 221.837757816214; tHostA(0,6,6) = 0.999379227378489; tHostA(0,6,7) = 0.000244033383405728; tHostA(0,6,8) = 0; tHostA(0,6,9) = 0.000244033383405728;
    tHostA(0,7,0) = 0.288388970538191; tHostA(0,7,1) = 0.000288388970538191; tHostA(0,7,2) = 221.837678518269; tHostA(0,7,3) = -155.286336004547; tHostA(0,7,4) = 0;
      tHostA(0,7,5) = -66.5512870543163; tHostA(0,7,6) = 0.000244033322428616; tHostA(0,7,7) = 0.999753676091541; tHostA(0,7,8) = 0; tHostA(0,7,9) = -0.000130872405284865;
    tHostA(0,8,0) = 0; tHostA(0,8,1) = 0; tHostA(0,8,2) = 0; tHostA(0,8,3) = 0; tHostA(0,8,4) = -47.5307664670919;
      tHostA(0,8,5) = 0; tHostA(0,8,6) = 0; tHostA(0,8,7) = 0; tHostA(0,8,8) = 0.999876504868183; tHostA(0,8,9) = 0;
    tHostA(0,9,0) = 0.288388970538190; tHostA(0,9,1) = 0.000288388970538190; tHostA(0,9,2) = 221.837678518269; tHostA(0,9,3) = -66.5512870543163; tHostA(0,9,4) = 0;
      tHostA(0,9,5) = -155.286336004547; tHostA(0,9,6) = 0.000244033322428672; tHostA(0,9,7) = -0.000130872405284421; tHostA(0,9,8) = 0; tHostA(0,9,9) = 0.999753676091540;
    Kokkos::deep_copy(tA, tHostA);

    Plato::ScalarArray3D tB("Matrix B WS", tNumCells, tNumInnrCols, tNumOutCols);
    auto tHostB = Kokkos::create_mirror(tB);
    tHostB(0,0,0) = 0; tHostB(0,0,1) = 0; tHostB(0,0,2) = 0; tHostB(0,0,3) = 0; tHostB(0,0,4) = 0;
      tHostB(0,0,5) = 0; tHostB(0,0,6) = 0; tHostB(0,0,7) = 0; tHostB(0,0,8) = 0;
    tHostB(0,1,0) = -769230.8; tHostB(0,1,1) = 0;; tHostB(0,1,2) = 0; tHostB(0,1,3) = 769230.8; tHostB(0,1,4) = 384615.4;
      tHostB(0,1,5) = 0; tHostB(0,1,6) = 0; tHostB(0,1,7) = -384615.4; tHostB(0,1,8) = 0;
    tHostB(0,2,0) = 0; tHostB(0,2,1) = 0; tHostB(0,2,2) = 0; tHostB(0,2,3) = 0; tHostB(0,2,4) = 0;
      tHostB(0,2,5) = 0; tHostB(0,2,6) = 0; tHostB(0,2,7) = 0; tHostB(0,2,8) = 0;
    tHostB(0,3,0) = 0; tHostB(0,3,1) = 0; tHostB(0,3,2) = 0; tHostB(0,3,3) = 0; tHostB(0,3,4) = 0.076779750;
      tHostB(0,3,5) = 0; tHostB(0,3,6) = 0; tHostB(0,3,7) = -0.07677975; tHostB(0,3,8) = 0;
    tHostB(0,4,0) = 0; tHostB(0,4,1) = 0.07677975; tHostB(0,4,2) = 0; tHostB(0,4,3) = 0.07677975; tHostB(0,4,4) = -0.07677975;
      tHostB(0,4,5) = 0; tHostB(0,4,6) = -0.07677975; tHostB(0,4,7) = 0; tHostB(0,4,8) = 0;
    tHostB(0,5,0) = 0; tHostB(0,5,1) = 0; tHostB(0,5,2) = 0; tHostB(0,5,3) = 0; tHostB(0,5,4) = -0.07677975;
      tHostB(0,5,5) = 0; tHostB(0,5,6) = 0; tHostB(0,5,7) = 0.07677975; tHostB(0,5,8) = 0;
    tHostB(0,6,0) = 0; tHostB(0,6,1) = 0; tHostB(0,6,2) = 0; tHostB(0,6,3) = 0; tHostB(0,6,4) = 0;
      tHostB(0,6,5) = 0; tHostB(0,6,6) = 0; tHostB(0,6,7) = 0; tHostB(0,6,8) = 0;
    tHostB(0,7,0) = 0; tHostB(0,7,1) = 0; tHostB(0,7,2) = 0; tHostB(0,7,3) = 0; tHostB(0,7,4) = 51.1865;
      tHostB(0,7,5) = 0; tHostB(0,7,6) = 0; tHostB(0,7,7) = -51.1865; tHostB(0,7,8) = 0;
    tHostB(0,8,0) = 0; tHostB(0,8,1) = 51.1865; tHostB(0,8,2) = 0; tHostB(0,8,3) = 51.1865; tHostB(0,8,4) = -51.1865;
      tHostB(0,8,5) = 0; tHostB(0,8,6) = -51.1865; tHostB(0,8,7) = 0; tHostB(0,8,8) = 0;
    tHostB(0,9,0) = 0; tHostB(0,9,1) = 0; tHostB(0,9,2) = 0; tHostB(0,9,3) = 0; tHostB(0,9,4) = -51.1865;
      tHostB(0,9,5) = 0; tHostB(0,9,6) = 0; tHostB(0,9,7) = 51.1865; tHostB(0,9,8) = 0;
    Kokkos::deep_copy(tB, tHostB);

    // CALL FUNCTION
    constexpr Plato::Scalar tBeta = 0.0;
    constexpr Plato::Scalar tAlpha = 1.0;
    Plato::ScalarArray3D tC("Matrix C WS", tNumCells, tNumOutRows, tNumOutCols);
    TEST_NOTHROW( (Plato::blas3::multiply(tNumCells, tAlpha, tA, tB, tBeta, tC)) );

    // 2. TEST RESULTS
    Plato::ScalarArray3D tGold("Gold", tNumCells, tNumOutRows, tNumOutCols);
    auto tHostGold = Kokkos::create_mirror(tGold);
    tHostGold(0,0,0) = 0.665513165892939; tHostGold(0,0,1) = 0; tHostGold(0,0,2) = 0; tHostGold(0,0,3) = -0.665513165892939; tHostGold(0,0,4) = -0.332756582946470;
      tHostGold(0,0,5) = 0; tHostGold(0,0,6) = 0; tHostGold(0,0,7) = 0.332756582946470; tHostGold(0,0,8) = 0;
    tHostGold(0,1,0) = 0.665513165892955; tHostGold(0,1,1) = 0; tHostGold(0,1,2) = 0; tHostGold(0,1,3) = -0.665513165892955; tHostGold(0,1,4) = -0.332756582946477;
      tHostGold(0,1,5) = 0; tHostGold(0,1,6) = 0; tHostGold(0,1,7) = 0.332756582946477; tHostGold(0,1,8) = 0;
    tHostGold(0,2,0) = 0.665513165875021; tHostGold(0,2,1) = 0; tHostGold(0,2,2) = 0; tHostGold(0,2,3) = -0.665513165875021; tHostGold(0,2,4) = -0.332756582937511;
      tHostGold(0,2,5) = 0; tHostGold(0,2,6) = 0; tHostGold(0,2,7) = 0.332756582937511; tHostGold(0,2,8) = 0;
    tHostGold(0,3,0) = -0.332756499781941;tHostGold(0,3,1) = 0; tHostGold(0,3,2) = 0; tHostGold(0,3,3) = 0.332756499781941; tHostGold(0,3,4) = 0.232929542988130 ;
      tHostGold(0,3,5) = 0; tHostGold(0,3,6) = 0; tHostGold(0,3,7) = -0.23292954298813; tHostGold(0,3,8) = 0;
    tHostGold(0,4,0) = 0; tHostGold(0,4,1) = 0.0712961436452182; tHostGold(0,4,2) = 0; tHostGold(0,4,3) = 0.0712961436452182; tHostGold(0,4,4) = -0.0712961436452182;
      tHostGold(0,4,5) = 0; tHostGold(0,4,6) = -0.0712961436452182; tHostGold(0,4,7) = 0; tHostGold(0,4,8) = 0;
    tHostGold(0,5,0) = -0.332756499781942; tHostGold(0,5,1) = 0; tHostGold(0,5,2) = 0; tHostGold(0,5,3) = 0.332756499781942; tHostGold(0,5,4) = 0.0998269567938113;
      tHostGold(0,5,5) = 0; tHostGold(0,5,6) = 0; tHostGold(0,5,7) = -0.0998269567938113; tHostGold(0,5,8) = 0;
    tHostGold(0,6,0) = 443.675626551306; tHostGold(0,6,1) = 0; tHostGold(0,6,2) = 0; tHostGold(0,6,3) = -443.675626551306; tHostGold(0,6,4) = -221.837813275653;
      tHostGold(0,6,5) = 0; tHostGold(0,6,6) = 0; tHostGold(0,6,7) = 221.837813275653; tHostGold(0,6,8) = 0;
    tHostGold(0,7,0) = -221.837678518269; tHostGold(0,7,1) = 0; tHostGold(0,7,2) = 0; tHostGold(0,7,3) = 221.837678518269; tHostGold(0,7,4) = 155.286374826131;
      tHostGold(0,7,5) = 0; tHostGold(0,7,6) = 0; tHostGold(0,7,7) = -155.286374826131; tHostGold(0,7,8) = 0;
    tHostGold(0,8,0) = 0; tHostGold(0,8,1) = 47.5307783497835; tHostGold(0,8,2) = 0; tHostGold(0,8,3) = 47.5307783497835; tHostGold(0,8,4) = -47.5307783497835;
      tHostGold(0,8,5) = 0; tHostGold(0,8,6) = -47.5307783497835; tHostGold(0,8,7) = 0; tHostGold(0,8,8) = 0;
    tHostGold(0,9,0) = -221.837678518269; tHostGold(0,9,1) = 0; tHostGold(0,9,2) = 0; tHostGold(0,9,3) = 221.837678518269; tHostGold(0,9,4) = 66.5513036921381;
      tHostGold(0,9,5) = 0; tHostGold(0,9,6) = 0; tHostGold(0,9,7) = -66.5513036921381; tHostGold(0,9,8) = 0;

    auto tHostC = Kokkos::create_mirror(tC);
    Kokkos::deep_copy(tHostC, tC);
    constexpr Plato::Scalar tTolerance = 1e-4;
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tC.extent(0); tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tC.extent(1); tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tC.extent(2); tColIndex++)
            {
                //printf("Result(%d,%d,%d) = %f\n", tCellIndex + 1, tRowIndex + 1, tColIndex+ 1, tHostC(tCellIndex, tRowIndex, tColIndex));
                TEST_FLOATING_EQUALITY(tHostGold(tCellIndex, tRowIndex, tColIndex), tHostC(tCellIndex, tRowIndex, tColIndex), tTolerance);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_MatrixTimesVectorWorkset_Error)
{
    // PREPARE DATA
    Plato::ScalarArray3D tA;
    Plato::ScalarMultiVector tX;
    Plato::ScalarMultiVector tY;

    // CALL FUNCTION - MATRIX A IS EMPTY
    constexpr Plato::OrdinalType tNumCells = 3;
    Plato::Scalar tAlpha = 1.5; Plato::Scalar tBeta = 2.5;
    TEST_THROW( (Plato::blas2::matrix_times_vector("N", tAlpha, tA, tX, tBeta, tY)), std::runtime_error );

    // CALL FUNCTION - VECTOR X IS EMPTY
    constexpr Plato::OrdinalType tNumCols = 2;
    constexpr Plato::OrdinalType tNumRows = 3;
    tA = Plato::ScalarArray3D("A Matrix WS", tNumCells, tNumRows, tNumCols);
    TEST_THROW( (Plato::blas2::matrix_times_vector("N", tAlpha, tA, tX, tBeta, tY)), std::runtime_error );

    // CALL FUNCTION - VECTOR Y IS EMPTY
    tX = Plato::ScalarMultiVector("X Vector WS", tNumCells, tNumCols);
    TEST_THROW( (Plato::blas2::matrix_times_vector("N", tAlpha, tA, tX, tBeta, tY)), std::runtime_error );

    // CALL FUNCTION - NUM CELL MISMATCH IN INPUT MATRIX
    tY = Plato::ScalarMultiVector("Y Vector WS", tNumCells + 1, tNumRows);
    TEST_THROW( (Plato::blas2::matrix_times_vector("N", tAlpha, tA, tX, tBeta, tY)), std::runtime_error );

    // CALL FUNCTION - NUM CELL MISMATCH IN INPUT VECTOR X
    Plato::ScalarMultiVector tVecX("X Vector WS", tNumCells + 1, tNumRows);
    TEST_THROW( (Plato::blas2::matrix_times_vector("N", tAlpha, tA, tVecX, tBeta, tY)), std::runtime_error );
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_MatrixTimesVectorWorkset)
{
    // 1. PREPARE DATA FOR TEST ONE
    constexpr Plato::OrdinalType tNumRows = 3;
    constexpr Plato::OrdinalType tNumCols = 2;
    constexpr Plato::OrdinalType tNumCells = 3;

    // 1.1 PREPARE MATRIX DATA
    Plato::ScalarArray3D tA("A Matrix WS", tNumCells, tNumRows, tNumCols);
    std::vector<std::vector<Plato::Scalar>> tMatrixData = {{1, 2}, {3, 4}, {5, 6}};
    auto tHostA = Kokkos::create_mirror(tA);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                tHostA(tCellIndex, tRowIndex, tColIndex) =
                        static_cast<Plato::Scalar>(tCellIndex + 1) * tMatrixData[tRowIndex][tColIndex];
            }
        }
    }
    Kokkos::deep_copy(tA, tHostA);

    // 1.2 PREPARE X VECTOR DATA
    Plato::ScalarMultiVector tX("X Vector WS", tNumCells, tNumCols);
    std::vector<Plato::Scalar> tXdata = {1, 2};
    auto tHostX = Kokkos::create_mirror(tX);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
        {
            tHostX(tCellIndex, tColIndex) = static_cast<Plato::Scalar>(tCellIndex + 1) * tXdata[tColIndex];
        }
    }
    Kokkos::deep_copy(tX, tHostX);

    // 1.3 PREPARE Y VECTOR DATA
    Plato::ScalarMultiVector tY("Y Vector WS", tNumCells, tNumRows);
    std::vector<Plato::Scalar> tYdata = {1, 2, 3};
    auto tHostY = Kokkos::create_mirror(tY);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            tHostY(tCellIndex, tRowIndex) = static_cast<Plato::Scalar>(tCellIndex + 1) * tYdata[tRowIndex];
        }
    }
    Kokkos::deep_copy(tY, tHostY);

    // 1.4 CALL FUNCTION - NO TRANSPOSE
    Plato::Scalar tAlpha = 1.5; Plato::Scalar tBeta = 2.5;
    TEST_NOTHROW( (Plato::blas2::matrix_times_vector("N", tAlpha, tA, tX, tBeta, tY)) );

    // 1.5 TEST RESULTS
    tHostY = Kokkos::create_mirror(tY);
    Kokkos::deep_copy(tHostY, tY);
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGoldOne = { {10, 21.5, 33}, {35, 76, 117}, {75, 163.5, 252} };
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            //printf("(%d,%d) = %f\n", tCellIndex, tRowIndex, tHostY(tCellIndex, tRowIndex));
            TEST_FLOATING_EQUALITY(tHostY(tCellIndex, tRowIndex), tGoldOne[tCellIndex][tRowIndex], tTolerance);
        }
    }

    // 2.1 PREPARE DATA FOR X VECTOR - TEST TWO
    Plato::ScalarMultiVector tVecX("X Vector WS", tNumCells, tNumRows);
    std::vector<Plato::Scalar> tVecXdata = {1, 2, 3};
    auto tHostVecX = Kokkos::create_mirror(tVecX);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            tHostVecX(tCellIndex, tRowIndex) = static_cast<Plato::Scalar>(tCellIndex + 1) * tVecXdata[tRowIndex];
        }
    }
    Kokkos::deep_copy(tVecX, tHostVecX);

    // 2.2 PREPARE Y VECTOR DATA
    Plato::ScalarMultiVector tVecY("Y Vector WS", tNumCells, tNumCols);
    std::vector<Plato::Scalar> tVecYdata = {1, 2};
    auto tHostVecY = Kokkos::create_mirror(tVecY);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
        {
            tHostVecY(tCellIndex, tColIndex) = static_cast<Plato::Scalar>(tCellIndex + 1) * tVecYdata[tColIndex];
        }
    }
    Kokkos::deep_copy(tVecY, tHostVecY);

    // 2.2 CALL FUNCTION - TRANSPOSE
    TEST_NOTHROW( (Plato::blas2::matrix_times_vector("T", tAlpha, tA, tVecX, tBeta, tVecY)) );

    // 2.3 TEST RESULTS
    tHostVecY = Kokkos::create_mirror(tVecY);
    Kokkos::deep_copy(tHostVecY, tVecY);
    std::vector<std::vector<Plato::Scalar>> tGoldTwo = { {35.5, 47}, {137, 178}, {304.5, 393} };
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
        {
            //printf("(%d,%d) = %f\n", tCellIndex, tColIndex, tHostVecY(tCellIndex, tColIndex));
            TEST_FLOATING_EQUALITY(tHostVecY(tCellIndex, tColIndex), tGoldTwo[tCellIndex][tColIndex], tTolerance);
        }
    }

    // 3. TEST VALIDITY OF TRANSPOSE
    TEST_THROW( (Plato::blas2::matrix_times_vector("C", tAlpha, tA, tVecX, tBeta, tVecY)), std::runtime_error );
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_IdentityWorkset)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumRows = 4;
    constexpr Plato::OrdinalType tNumCols = 4;
    constexpr Plato::OrdinalType tNumCells = 3;
    Plato::ScalarArray3D tIdentity("tIdentity WS", tNumCells, tNumRows, tNumCols);

    // CALL FUNCTION
    Plato::blas3::identity<tNumRows, tNumCols>(tNumCells, tIdentity);

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = { {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1} };
    auto tHostIdentity = Kokkos::create_mirror(tIdentity);
    Kokkos::deep_copy(tHostIdentity, tIdentity);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                TEST_FLOATING_EQUALITY(tHostIdentity(tCellIndex, tRowIndex, tColIndex), tGold[tRowIndex][tColIndex], tTolerance);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_InverseMatrixWorkset)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumRows = 2;
    constexpr Plato::OrdinalType tNumCols = 2;
    constexpr Plato::OrdinalType tNumCells = 3; // Number of matrices to invert
    Plato::ScalarArray3D tMatrix("Matrix A", tNumCells, 2, 2);
    auto tHostMatrix = Kokkos::create_mirror(tMatrix);
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; ++tCellIndex)
    {
        const Plato::Scalar tScaleFactor = 1.0 / (1.0 + tCellIndex);
        tHostMatrix(tCellIndex, 0, 0) = -2.0 * tScaleFactor;
        tHostMatrix(tCellIndex, 1, 0) = 1.0 * tScaleFactor;
        tHostMatrix(tCellIndex, 0, 1) = 1.5 * tScaleFactor;
        tHostMatrix(tCellIndex, 1, 1) = -0.5 * tScaleFactor;
    }
    Kokkos::deep_copy(tMatrix, tHostMatrix);

    // CALL FUNCTION
    Plato::ScalarArray3D tAInverse("A Inverse", tNumCells, 2, 2);
    Plato::blas3::inverse<tNumRows, tNumCols>(tNumCells, tMatrix, tAInverse);

    constexpr Plato::Scalar tTolerance = 1e-6;
    std::vector<std::vector<Plato::Scalar> > tGoldMatrixInverse = { { 1.0, 3.0 }, { 2.0, 4.0 } };
    auto tHostAInverse = Kokkos::create_mirror(tAInverse);
    Kokkos::deep_copy(tHostAInverse, tAInverse);
    for (Plato::OrdinalType tMatrixIndex = 0; tMatrixIndex < tNumCells; tMatrixIndex++)
    {
        for (Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for (Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                //printf("Matrix %d Inverse (%d,%d) = %f\n", n, i, j, tHostAInverse(n, i, j));
                const Plato::Scalar tScaleFactor = (1.0 + tMatrixIndex);
                TEST_FLOATING_EQUALITY(tHostAInverse(tMatrixIndex, tRowIndex, tColIndex), tScaleFactor * tGoldMatrixInverse[tRowIndex][tColIndex], tTolerance);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ApplyPenalty)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumRows = 3;
    constexpr Plato::OrdinalType tNumCols = 3;
    Plato::ScalarMultiVector tA("A: 2-D View", tNumRows, tNumCols);
    std::vector<std::vector<Plato::Scalar>> tData = { {10, 20, 30}, {35, 76, 117}, {75, 163, 252} };

    auto tHostA = Kokkos::create_mirror(tA);
    for (Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; ++tRowIndex)
    {
        for (Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; ++tColIndex)
        {
            tHostA(tRowIndex, tColIndex) = tData[tRowIndex][tColIndex];
        }
    }
    Kokkos::deep_copy(tA, tHostA);

    // CALL FUNCTION
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aRowIndex)
    {
        Plato::apply_penalty<tNumCols>(aRowIndex, 0.5, tA);
    }, "identity workset");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    tHostA = Kokkos::create_mirror(tA);
    Kokkos::deep_copy(tHostA, tA);
    std::vector<std::vector<Plato::Scalar>> tGold = { {5, 10, 15}, {17.5, 38, 58.5}, {37.5, 81.5, 126} };
    for (Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
    {
        for (Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostA(tRowIndex, tColIndex), tGold[tRowIndex][tColIndex], tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeBulkModulusError)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                 \n"
      "  <ParameterList name='Material Model'>                              \n"
      "  </ParameterList>                                                   \n"
      "</ParameterList>                                                     \n"
    );

    TEST_THROW( (Plato::compute_bulk_modulus(*tParams)), std::runtime_error );
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeBulkModulus)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                 \n"
      "  <ParameterList name='Material Model'>                              \n"
      "    <ParameterList name='Isotropic Linear Elastic'>                  \n"
      "      <Parameter  name='Density' type='double' value='1000'/>        \n"
      "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Youngs Modulus' type='double' value='1.0'/>  \n"
      "    </ParameterList>                                                 \n"
      "  </ParameterList>                                                   \n"
      "</ParameterList>                                                     \n"
    );

    auto tBulk = Plato::compute_bulk_modulus(*tParams);
    constexpr Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tBulk, 0.833333333333333, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeShearModulusError)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                 \n"
      "  <ParameterList name='Material Model'>                              \n"
      "  </ParameterList>                                                   \n"
      "</ParameterList>                                                     \n"
    );

    TEST_THROW( (Plato::compute_shear_modulus(*tParams)), std::runtime_error );
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeShearModulus)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                 \n"
      "  <ParameterList name='Material Model'>                              \n"
      "    <ParameterList name='Isotropic Linear Elastic'>                  \n"
      "      <Parameter  name='Density' type='double' value='1000'/>        \n"
      "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Youngs Modulus' type='double' value='1.0'/>  \n"
      "    </ParameterList>                                                 \n"
      "  </ParameterList>                                                   \n"
      "</ParameterList>                                                     \n"
    );

    auto tShear = Plato::compute_shear_modulus(*tParams);
    constexpr Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tShear, 0.384615384615385, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeShearAndBulkModulus)
{
    const Plato::Scalar tPoisson = 0.3;
    const Plato::Scalar tElasticModulus = 1;
    auto tBulk = Plato::compute_bulk_modulus(tElasticModulus, tPoisson);
    constexpr Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tBulk, 0.833333333333333, tTolerance);
    auto tShear = Plato::compute_shear_modulus(tElasticModulus, tPoisson);
    TEST_FLOATING_EQUALITY(tShear, 0.384615384615385, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_StrainDivergence3D)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::ScalarVector tOutput("strain tensor divergence", tNumCells);
    Plato::ScalarMultiVector tStrainTensor("strain tensor", tNumCells, tNumVoigtTerms);
    auto tHostStrainTensor = Kokkos::create_mirror(tStrainTensor);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostStrainTensor(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
        tHostStrainTensor(tCellIndex, 1) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.2;
        tHostStrainTensor(tCellIndex, 2) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.3;
        tHostStrainTensor(tCellIndex, 3) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.4;
        tHostStrainTensor(tCellIndex, 4) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.5;
        tHostStrainTensor(tCellIndex, 5) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.6;
    }
    Kokkos::deep_copy(tStrainTensor, tHostStrainTensor);

    // CALL FUNCTION
    Plato::StrainDivergence<tSpaceDim> tComputeStrainDivergence;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeStrainDivergence(aCellIndex, tStrainTensor, tOutput);
    }, "test strain divergence functor");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = {0.6, 1.2, 1.8};
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostOutput(tCellIndex), tGold[tCellIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_StrainDivergence2D)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;
    Plato::ScalarVector tOutput("strain tensor divergence", tNumCells);
    Plato::ScalarMultiVector tStrainTensor("strain tensor", tNumCells, tNumVoigtTerms);
    auto tHostStrainTensor = Kokkos::create_mirror(tStrainTensor);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostStrainTensor(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
        tHostStrainTensor(tCellIndex, 1) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.2;
        tHostStrainTensor(tCellIndex, 2) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.3;
    }
    Kokkos::deep_copy(tStrainTensor, tHostStrainTensor);

    // CALL FUNCTION
    Plato::StrainDivergence<tSpaceDim> tComputeStrainDivergence;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeStrainDivergence(aCellIndex, tStrainTensor, tOutput);
    }, "test strain divergence functor");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = {0.3, 0.6, 0.9};
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostOutput(tCellIndex), tGold[tCellIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_StrainDivergence1D)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 1;
    constexpr Plato::OrdinalType tNumVoigtTerms = 1;
    Plato::ScalarVector tOutput("strain tensor divergence", tNumCells);
    Plato::ScalarMultiVector tStrainTensor("strain tensor", tNumCells, tNumVoigtTerms);
    auto tHostStrainTensor = Kokkos::create_mirror(tStrainTensor);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostStrainTensor(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
    }
    Kokkos::deep_copy(tStrainTensor, tHostStrainTensor);

    // CALL FUNCTION
    Plato::StrainDivergence<tSpaceDim> tComputeStrainDivergence;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeStrainDivergence(aCellIndex, tStrainTensor, tOutput);
    }, "test strain divergence functor");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGold = {0.1, 0.2, 0.3};
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostOutput(tCellIndex), tGold[tCellIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeStabilization3D)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 3;

    Plato::ScalarVector tCellVolume("volume", tNumCells);
    auto tHostCellVolume = Kokkos::create_mirror(tCellVolume);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostCellVolume(tCellIndex) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
    }
    Kokkos::deep_copy(tCellVolume, tHostCellVolume);

    Plato::ScalarMultiVector tPressureGrad("pressure gradient", tNumCells, tSpaceDim);
    auto tHostPressureGrad = Kokkos::create_mirror(tPressureGrad);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostPressureGrad(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
        tHostPressureGrad(tCellIndex, 1) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.2;
        tHostPressureGrad(tCellIndex, 2) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.3;
    }
    Kokkos::deep_copy(tPressureGrad, tHostPressureGrad);

    Plato::ScalarMultiVector tProjectedPressureGrad("projected pressure gradient - gauss pt", tNumCells, tSpaceDim);
    auto tHostProjectedPressureGrad = Kokkos::create_mirror(tProjectedPressureGrad);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostProjectedPressureGrad(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 1;
        tHostProjectedPressureGrad(tCellIndex, 1) = static_cast<Plato::Scalar>(1+tCellIndex) * 2;
        tHostProjectedPressureGrad(tCellIndex, 2) = static_cast<Plato::Scalar>(1+tCellIndex) * 3;
    }
    Kokkos::deep_copy(tProjectedPressureGrad, tHostProjectedPressureGrad);

    // CALL FUNCTION
    constexpr Plato::Scalar tScaling = 0.5;
    constexpr Plato::Scalar tShearModulus = 2;
    Plato::ScalarMultiVector tStabilization("cell stabilization", tNumCells, tSpaceDim);
    Plato::ComputeStabilization<tSpaceDim> tComputeStabilization(tScaling, tShearModulus);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeStabilization(aCellIndex, tCellVolume, tPressureGrad, tProjectedPressureGrad, tStabilization);
    }, "test compute stabilization functor");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    auto tHostStabilization = Kokkos::create_mirror(tStabilization);
    Kokkos::deep_copy(tHostStabilization, tStabilization);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.0255839119441290, -0.0511678238882572, -0.0767517358323859},
                                                     {-0.0812238574671431, -0.1624477149342860, -0.2436715724014290},
                                                     {-0.1596500440960990, -0.3193000881921980, -0.4789501322882970}};
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (Plato::OrdinalType tDimIndex = 0; tDimIndex < tSpaceDim; tDimIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostStabilization(tCellIndex, tDimIndex), tGold[tCellIndex][tDimIndex], tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeStabilization2D)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 2;

    Plato::ScalarVector tCellVolume("volume", tNumCells);
    auto tHostCellVolume = Kokkos::create_mirror(tCellVolume);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostCellVolume(tCellIndex) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
    }
    Kokkos::deep_copy(tCellVolume, tHostCellVolume);

    Plato::ScalarMultiVector tPressureGrad("pressure gradient", tNumCells, tSpaceDim);
    auto tHostPressureGrad = Kokkos::create_mirror(tPressureGrad);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostPressureGrad(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
        tHostPressureGrad(tCellIndex, 1) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.2;
    }
    Kokkos::deep_copy(tPressureGrad, tHostPressureGrad);

    Plato::ScalarMultiVector tProjectedPressureGrad("projected pressure gradient - gauss pt", tNumCells, tSpaceDim);
    auto tHostProjectedPressureGrad = Kokkos::create_mirror(tProjectedPressureGrad);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostProjectedPressureGrad(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 1;
        tHostProjectedPressureGrad(tCellIndex, 1) = static_cast<Plato::Scalar>(1+tCellIndex) * 2;
    }
    Kokkos::deep_copy(tProjectedPressureGrad, tHostProjectedPressureGrad);

    // CALL FUNCTION
    constexpr Plato::Scalar tScaling = 0.5;
    constexpr Plato::Scalar tShearModulus = 2;
    Plato::ScalarMultiVector tStabilization("cell stabilization", tNumCells, tSpaceDim);
    Plato::ComputeStabilization<tSpaceDim> tComputeStabilization(tScaling, tShearModulus);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeStabilization(aCellIndex, tCellVolume, tPressureGrad, tProjectedPressureGrad, tStabilization);
    }, "test compute stabilization functor");

    // TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-6;
    auto tHostStabilization = Kokkos::create_mirror(tStabilization);
    Kokkos::deep_copy(tHostStabilization, tStabilization);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.0255839119441290, -0.0511678238882572},
                                                     {-0.0812238574671431, -0.1624477149342860},
                                                     {-0.1596500440960990, -0.3193000881921980}};
    for (Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for (Plato::OrdinalType tDimIndex = 0; tDimIndex < tSpaceDim; tDimIndex++)
        {
            TEST_FLOATING_EQUALITY(tHostStabilization(tCellIndex, tDimIndex), tGold[tCellIndex][tDimIndex], tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeStabilization1D)
{
    // PREPARE DATA FOR TEST
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 1;

    Plato::ScalarVector tCellVolume("volume", tNumCells);
    auto tHostCellVolume = Kokkos::create_mirror(tCellVolume);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostCellVolume(tCellIndex) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
    }
    Kokkos::deep_copy(tCellVolume, tHostCellVolume);

    Plato::ScalarMultiVector tPressureGrad("pressure gradient", tNumCells, tSpaceDim);
    auto tHostPressureGrad = Kokkos::create_mirror(tPressureGrad);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostPressureGrad(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 0.1;
    }
    Kokkos::deep_copy(tPressureGrad, tHostPressureGrad);

    Plato::ScalarMultiVector tProjectedPressureGrad("projected pressure gradient - gauss pt", tNumCells, tSpaceDim);
    auto tHostProjectedPressureGrad = Kokkos::create_mirror(tProjectedPressureGrad);
    for(Plato::OrdinalType tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        tHostProjectedPressureGrad(tCellIndex, 0) = static_cast<Plato::Scalar>(1+tCellIndex) * 1;
    }
    Kokkos::deep_copy(tProjectedPressureGrad, tHostProjectedPressureGrad);

    // CALL FUNCTION
    constexpr Plato::Scalar tScaling = 0.5;
    constexpr Plato::Scalar tShearModulus = 2;
    Plato::ScalarMultiVector tStabilization("cell stabilization", tNumCells, tSpaceDim);
    Plato::ComputeStabilization<tSpaceDim> tComputeStabilization(tScaling, tShearModulus);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        tComputeStabilization(aCellIndex, tCellVolume, tPressureGrad, tProjectedPressureGrad, tStabilization);
    }, "test compute stabilization functor");

    // TEST RESULTS
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{-0.0255839119441290}, {-0.0812238574671431}, {-0.1596500440960990}};
    auto tHostStabilization = Kokkos::create_mirror(tStabilization);
    Kokkos::deep_copy(tHostStabilization, tStabilization);

    const Plato::Scalar tTolerance = 1e-4;
    const Plato::OrdinalType tDim0 = tStabilization.extent(0);
    const Plato::OrdinalType tDim1 = tStabilization.extent(1);
    for (Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            //printf("X(%d,%d) = %f\n", tIndexI, tIndexJ, tHostInput(tIndexI, tIndexJ));
            TEST_FLOATING_EQUALITY(tHostStabilization(tIndexI, tIndexJ), tGold[tIndexI][tIndexJ], tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ComputeElasticStrain3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // Set configuration workset
    auto tNumCells = tMesh->nelems();
    using PhysicsT = Plato::SimplexPlasticity<tSpaceDim>;
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);
    Plato::ScalarArray3D tConfig("configuration", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfig);

    // Set state workset
    auto tNumNodes = tMesh->nverts();
    decltype(tNumNodes) tNumState = 4;
    auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    Plato::ScalarVectorT<Plato::Scalar> tState("state", tNumState * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // disp_z
        tState(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVector tStateWS("current state", tNumCells, PhysicsT::mNumDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    Plato::ScalarVectorT<Plato::Scalar> tCellVolume("cell volume", tNumCells);
    Plato::ScalarMultiVectorT<Plato::Scalar> tPressGrad("pressure grad", tNumCells, tSpaceDim);
    Plato::ScalarMultiVectorT<Plato::Scalar> tElasticStrains("strains", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVectorT<Plato::Scalar> tTotalStrain("total strain", tNumCells, tNumVoigtTerms);
    Plato::ScalarArray3DT<Plato::Scalar> tGradient("gradient", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);
    Plato::ScalarMultiVectorT<Plato::Scalar> tLocalState("local state", tNumCells, PhysicsT::mNumLocalDofsPerCell);

    Plato::ComputeGradientWorkset <tSpaceDim> tComputeGradient;
    Plato::LinearTetCubRuleDegreeOne <tSpaceDim> tCubatureRule;
    Plato::Strain<tSpaceDim, PhysicsT::mNumDofsPerNode> tComputeTotalStrain;
    Plato::ThermoPlasticityUtilities <tSpaceDim, PhysicsT> tThermoPlasticityUtils;

    auto tBasisFunctions = tCubatureRule.getBasisFunctions();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfig, tCellVolume);
        tComputeTotalStrain(aCellOrdinal, tTotalStrain, tStateWS, tGradient);
        tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, tStateWS, tLocalState,
                                                    tBasisFunctions, tTotalStrain, tElasticStrains);
    }, "compute elastic strain test");

    std::vector<std::vector<Plato::Scalar>> tGold =
        { {1e-7,  6e-7,  3e-7, 1.1e-6,   4e-7,   5e-7},
          {3e-7,  6e-7, -3e-7,   7e-7,   8e-7,   9e-7},
          {3e-7,  2e-7,  3e-7,   5e-7,   1e-6,   7e-7},
          {5e-7, -2e-7,  3e-7,  -1e-7, 1.6e-6,   9e-7},
          {7e-7, -2e-7, -3e-7,  -5e-7,   2e-6, 1.3e-6},
          {7e-7, -6e-7,  3e-7,  -7e-7, 2.2e-6, 1.1e-6} };
    auto tHostStrains = Kokkos::create_mirror(tElasticStrains);
    Kokkos::deep_copy(tHostStrains, tElasticStrains);

    const Plato::Scalar tTolerance = 1e-4;
    const Plato::OrdinalType tDim0 = tElasticStrains.extent(0);
    const Plato::OrdinalType tDim1 = tElasticStrains.extent(1);
    for (Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            //printf("X(%d,%d) = %f\n", tIndexI, tIndexJ, tHostInput(tIndexI, tIndexJ));
            TEST_FLOATING_EQUALITY(tHostStrains(tIndexI, tIndexJ), tGold[tIndexI][tIndexJ], tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_Residual2D_Elastic)
{
    // 1. PREPARE PROBLEM INPUTS FOR TEST
    Teuchos::RCP<Teuchos::ParameterList> tElastoPlasticityInputs =
        Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                        \n"
        "  <ParameterList name='Spatial Model'>                                      \n"
        "    <ParameterList name='Domains'>                                          \n"
        "      <ParameterList name='Design Volume'>                                  \n"
        "        <Parameter name='Element Block' type='string' value='body'/>        \n"
        "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
        "      </ParameterList>                                                      \n"
        "    </ParameterList>                                                        \n"
        "  </ParameterList>                                                          \n"
        "  <ParameterList name='Material Models'>                                    \n"
        "    <ParameterList name='Unobtainium'>                                      \n"
        "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
        "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>       \n"
        "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>     \n"
        "      </ParameterList>                                                      \n"
        "    </ParameterList>                                                        \n"
        "  </ParameterList>                                                          \n"
        "  <ParameterList name='Elliptic'>                                           \n"
        "    <ParameterList name='Penalty Function'>                                 \n"
        "      <Parameter name='Type' type='string' value='SIMP'/>                   \n"
        "      <Parameter name='Exponent' type='double' value='3.0'/>                \n"
        "      <Parameter name='Minimum Value' type='double' value='1.0e-6'/>        \n"
        "    </ParameterList>                                                        \n"
        "  </ParameterList>                                                          \n"
        "</ParameterList>                                                            \n"
      );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    Plato::DataMap tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tElastoPlasticityInputs);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    // 2. PREPARE FUNCTION INPUTS FOR TEST
    const Plato::OrdinalType tNumNodes = tMesh->nverts();
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    using PhysicsT = Plato::SimplexPlasticity<tSpaceDim>;
    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);

    // 2.1 SET CONFIGURATION
    Plato::ScalarArray3DT<EvalType::ConfigScalarType> tConfiguration("configuration", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfiguration);

    // 2.2 SET DESIGN VARIABLES
    Plato::ScalarMultiVectorT<EvalType::ControlScalarType> tDesignVariables("design variables", tNumCells, PhysicsT::mNumNodesPerCell);
    Kokkos::deep_copy(tDesignVariables, 1.0);

    // 2.3 SET GLOBAL STATE
    auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    Plato::ScalarVector tGlobalState("global state", tNumDofsPerNode * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVectorT<EvalType::StateScalarType> tCurrentGlobalState("current global state", tNumCells, PhysicsT::mNumDofsPerCell);
    tWorksetBase.worksetState(tGlobalState, tCurrentGlobalState);
    Plato::ScalarMultiVectorT<EvalType::PrevStateScalarType> tPrevGlobalState("previous global state", tNumCells, PhysicsT::mNumDofsPerCell);

    // 2.4 SET PROJECTED PRESSURE GRADIENT
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    Plato::ScalarMultiVectorT<EvalType::NodeStateScalarType> tProjectedPressureGrad("projected pressure grad", tNumCells, PhysicsT::mNumNodeStatePerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex=0; tNodeIndex< tNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex=0; tDimIndex< tSpaceDim; tDimIndex++)
            {
                tProjectedPressureGrad(aCellOrdinal, tNodeIndex*tSpaceDim+tDimIndex) = (4e-7)*(tNodeIndex+1)*(tDimIndex+1)*(aCellOrdinal+1);
            }
        }
    }, "set projected pressure grad");

    // 2.5 SET LOCAL STATE
    Plato::ScalarMultiVectorT<EvalType::LocalStateScalarType> tCurrentLocalState("current local state", tNumCells, PhysicsT::mNumLocalDofsPerCell);
    Plato::ScalarMultiVectorT<EvalType::PrevLocalStateScalarType> tPrevLocalState("previous local state", tNumCells, PhysicsT::mNumLocalDofsPerCell);

    // 3. CALL FUNCTION
    Plato::InfinitesimalStrainPlasticityResidual<EvalType, PhysicsT> tComputeResidual(tOnlyDomain, tDataMap, *tElastoPlasticityInputs);
    Plato::ScalarMultiVectorT<EvalType::ResultScalarType> tResidualWS("residual", tNumCells, PhysicsT::mNumDofsPerCell);
    Plato::TimeData tTimeData(*tElastoPlasticityInputs);
    tComputeResidual.evaluate(tCurrentGlobalState, tPrevGlobalState, tCurrentLocalState, tPrevLocalState,
                              tProjectedPressureGrad, tDesignVariables, tConfiguration, tResidualWS, tTimeData);

    // 5. TEST RESULTS
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostResidualWS = Kokkos::create_mirror(tResidualWS);
    Kokkos::deep_copy(tHostResidualWS, tResidualWS);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {
          { -0.102564, -0.0961538, 1.66666e-08, 0.00641051, 0.185897, 1.66666e-08, 9.615385e-02, -0.0897433, 1.66673e-08},
          { 1.5e-07, 5.769231e-02, 5.00005e-08, 0.0576922, -0.0192306, 5.00005e-08, -0.0576923, -0.0384617, 5.00006e-08}
        };
    for(Plato::OrdinalType tCellIndex=0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tDofIndex=0; tDofIndex< PhysicsT::mNumDofsPerCell; tDofIndex++)
        {
            //printf("residual(%d,%d) = %.10f\n", tCellIndex, tDofIndex, tHostResidualWS(tCellIndex, tDofIndex));
            TEST_FLOATING_EQUALITY(tHostResidualWS(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_Residual3D_Elastic)
{
    // 1. PREPARE PROBLEM INPUS FOR TEST
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                        \n"
        "  <ParameterList name='Spatial Model'>                                      \n"
        "    <ParameterList name='Domains'>                                          \n"
        "      <ParameterList name='Design Volume'>                                  \n"
        "        <Parameter name='Element Block' type='string' value='body'/>        \n"
        "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
        "      </ParameterList>                                                      \n"
        "    </ParameterList>                                                        \n"
        "  </ParameterList>                                                          \n"
        "  <ParameterList name='Material Models'>                                    \n"
        "    <ParameterList name='Unobtainium'>                                      \n"
        "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
        "        <Parameter  name='Poissons Ratio' type='double' value='0.35'/>      \n"
        "        <Parameter  name='Youngs Modulus' type='double' value='1e11'/>      \n"
        "      </ParameterList>                                                      \n"
        "    </ParameterList>                                                        \n"
        "  </ParameterList>                                                          \n"
        "  <ParameterList name='Elliptic'>                                           \n"
        "    <ParameterList name='Penalty Function'>                                 \n"
        "      <Parameter name='Type' type='string' value='SIMP'/>                   \n"
        "      <Parameter name='Exponent' type='double' value='3.0'/>                \n"
        "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>        \n"
        "    </ParameterList>                                                        \n"
        "  </ParameterList>                                                          \n"
        "</ParameterList>                                                            \n"
      );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    Plato::DataMap tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tInputs);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    // 2. PREPARE FUNCTION INPUTS FOR TEST
    const Plato::OrdinalType tNumNodes = tMesh->nverts();
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    using PhysicsT = Plato::SimplexPlasticity<tSpaceDim>;
    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);

    // 2.1 SET CONFIGURATION
    Plato::ScalarArray3DT<EvalType::ConfigScalarType> tConfiguration("configuration", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfiguration);

    // 2.2 SET DESIGN VARIABLES
    Plato::ScalarMultiVectorT<EvalType::ControlScalarType> tDesignVariables("design variables", tNumCells, PhysicsT::mNumNodesPerCell);
    Kokkos::deep_copy(tDesignVariables, 1.0);

    // 2.3 SET GLOBAL STATE
    auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    decltype(tNumDofsPerNode) tNumState = 4;
    Plato::ScalarVector tGlobalState("global state", tNumState * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // disp_z
        tGlobalState(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVectorT<EvalType::StateScalarType> tCurrentGlobalState("current global state", tNumCells, PhysicsT::mNumDofsPerCell);
    tWorksetBase.worksetState(tGlobalState, tCurrentGlobalState);
    Plato::ScalarMultiVectorT<EvalType::PrevStateScalarType> tPrevGlobalState("previous global state", tNumCells, PhysicsT::mNumDofsPerCell);

    // 2.4 SET PROJECTED PRESSURE GRADIENT
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    Plato::ScalarMultiVectorT<EvalType::NodeStateScalarType> tProjectedPressureGrad("projected pressure grad", tNumCells, PhysicsT::mNumNodeStatePerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex=0; tNodeIndex< tNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex=0; tDimIndex< tSpaceDim; tDimIndex++)
            {
                tProjectedPressureGrad(aCellOrdinal, tNodeIndex*tSpaceDim+tDimIndex) = (4e-7)*(tNodeIndex+1)*(tDimIndex+1)*(aCellOrdinal+1);
            }
        }
    }, "set projected pressure grad");

    // 2.5 SET LOCAL STATE
    Plato::ScalarMultiVectorT<EvalType::LocalStateScalarType> tCurrentLocalState("current local state", tNumCells, PhysicsT::mNumLocalDofsPerCell);
    Plato::ScalarMultiVectorT<EvalType::PrevLocalStateScalarType> tPrevLocalState("previous local state", tNumCells, PhysicsT::mNumLocalDofsPerCell);

    // 3. CALL FUNCTION
    Plato::InfinitesimalStrainPlasticityResidual<EvalType, PhysicsT> tComputeResidual(tOnlyDomain, tDataMap, *tInputs);
    Plato::ScalarMultiVectorT<EvalType::ResultScalarType> tResidual("residual", tNumCells, PhysicsT::mNumDofsPerCell);
    Plato::TimeData tTimeData(*tInputs);
    tComputeResidual.evaluate(tCurrentGlobalState, tPrevGlobalState, tCurrentLocalState, tPrevLocalState,
                              tProjectedPressureGrad, tDesignVariables, tConfiguration, tResidual, tTimeData);

    // 5. TEST RESULTS
    std::vector<std::vector<Plato::Scalar>> tGold =
    {
      {-3.086420e+03, -3.292181e+03, -6.790123e+03,  4.166667e-08,  -5.349794e+03, -3.703704e+03, 2.880658e+03, 4.166667e-08,
       5.967078e+03,  2.057613e+02,  4.320988e+03, 4.166667e-08,  2.469136e+03,  6.790123e+03,  -4.115226e+02, 4.166667e-08},

      {-5.555556e+03, -4.938272e+03, -4.320988e+03,  2.500000e-08,  6.172840e+02,  6.172840e+02, 1.049383e+04, 2.500000e-08,
       3.703704e+03, -1.234568e+03,  -1.111111e+04,  2.500000e-08,  1.234568e+03,  5.555556e+03,  4.938272e+03, 2.500000e-08},

      {-6.172840e+03, -3.086420e+03, -4.115226e+02,  3.333333e-08, 3.909465e+03,  -5.144033e+03, -3.086420e+03,  3.333333e-08,
        1.851852e+03, 3.909465e+03,  -2.674897e+03,  3.333333e-08,  4.115226e+02,  4.320988e+03,  6.172840e+03, 3.333333e-08},

      {-9.876543e+03,  6.172840e+02, -1.234568e+03,  2.500000e-08,  -1.851852e+03, 1.049383e+04,  1.049383e+04, 2.500000e-08,
        5.555556e+03,  -4.938272e+03, -6.172840e+02, 2.500000e-08, 6.172840e+03, -6.172840e+03,  -8.641975e+03, 2.500000e-08},

      { -4.526749e+03,  1.111111e+04, 1.687243e+04, 8.333333e-09,  4.320988e+03, 2.057613e+02,  -1.440329e+03, 8.333333e-09,
        8.024691e+03, -3.292181e+03, -3.086420e+03, 8.333333e-09, -7.818930e+03, -8.024691e+03, -1.234568e+04, 8.333333e-09},

      { 2.057613e+02, 1.584362e+04,  1.790123e+04, 1.666667e-08,  1.358025e+04, -4.320988e+03, 2.057613e+03, 1.666667e-08,
       -6.790123e+03, -4.732510e+03, -6.378601e+03, 1.666667e-08, -6.995885e+03, -6.790123e+03, -1.358025e+04, 1.666667e-08}
    };

    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    constexpr Plato::Scalar tTolerance = 1e-4;
    for(Plato::OrdinalType tCellIndex=0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tDofIndex=0; tDofIndex< PhysicsT::mNumDofsPerCell; tDofIndex++)
        {
            //printf("residual(%d,%d) = %.10f\n", tCellIndex, tDofIndex, tHostElastoPlasticityResidual(tCellIndex, tDofIndex));
            TEST_FLOATING_EQUALITY(tHostResidual(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ElasticSolution3D)
{
    // 1. DEFINE PROBLEM
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Density' type='double' value='1e3'/>                           \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.35'/>                   \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1e11'/>                   \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1e9'/>         \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1e9'/>         \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1e9'/>                \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='-0.001*t'/>                   \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solve problem
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls);

    std::vector<std::vector<Plato::Scalar>> tGold =
        {
          {0, 0, 0, -418444.0, 0, 0, 0, -3.06295e+06, 0, 0, 0,  7.8685e+06, 0, 0, 0, 9.58504e+06,
           3.118233e-4, -1.0e-3, 4.815153e-5, 1.97175e+06, 2.340348e-4, -1.0e-3, 4.357691e-5, -418444.0,
           -3.927496e-4, -1.0e-3, 5.100447e-5, -1.10956e+07, -1.803906e-4, -1.0e-3, 9.081316e-5, -7.77742e+06}
        };
    auto tState = tSolution.get("State");
    auto tHostState = Kokkos::create_mirror(tState);
    Kokkos::deep_copy(tHostState, tState);

    const Plato::Scalar tTolerance = 1e-4;
    const Plato::OrdinalType tDim0 = tState.extent(0);
    const Plato::OrdinalType tDim1 = tState.extent(1);
    for (Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            //printf("X(%d,%d) = %f\n", tIndexI, tIndexJ, tHostInput(tIndexI, tIndexJ));
            TEST_FLOATING_EQUALITY(tHostState(tIndexI, tIndexJ), tGold[tIndexI][tIndexJ], tTolerance);
        }
    }

    // 6. Output Data
    if(tOutputData)
    {
        tPlasticityProblem.output("ElasticSolution");
    }

    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_SimplySupportedBeamTractionForce2D_Elastic)
{
    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,10,2);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.24'/>                   \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e7'/>                  \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0'/>        \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0'/>        \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='16e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                          \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='10'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "  <ParameterList  name='Mechanical Natural Boundary Conditions'>                         \n"
      "   <ParameterList  name='Traction Vector Boundary Condition'>                            \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(string)' value='{0.0, -100*t, 0.0}'/>         \n"
      "     <Parameter  name='Sides'    type='string'        value='Load'/>                     \n"
      "   </ParameterList>                                                                      \n"
      "  </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='XFixed'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;

    // 1. Construct plasticity problem
    auto tFaceIDs1 = PlatoUtestHelpers::get_edge_ids_on_y1(*tMesh);
    tMeshSets[Omega_h::SIDE_SET]["Load"] = tFaceIDs1;

    auto tDirichletBoundaryNodesX0 = PlatoUtestHelpers::get_2D_boundary_nodes_x0(*tMesh);
    tMeshSets[Omega_h::NODE_SET]["XFixed"] = tDirichletBoundaryNodesX0;

    Omega_h::LOs tPinnedNodeLOs({32});
    tMeshSets[Omega_h::NODE_SET]["Pinned"] = tPinnedNodeLOs;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls);

    // 5. Test results
    auto tState = tSolution.get("State");
    Plato::ScalarMultiVector tPressure("Pressure", tState.extent(0), tNumVertices);
    Plato::ScalarMultiVector tDisplacements("Displacements", tState.extent(0), tNumVertices * tSpaceDim);
    Plato::blas2::extract<PhysicsT::mNumDofsPerNode, PhysicsT::mPressureDofOffset>(tState, tPressure);
    Plato::blas2::extract<PhysicsT::mNumDofsPerNode, tSpaceDim>(tNumVertices, tState, tDisplacements);

    // 5.1 test pressure
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostPressure = Kokkos::create_mirror(tPressure);
    Kokkos::deep_copy(tHostPressure, tPressure);
    std::vector<std::vector<Plato::Scalar>> tGoldPress =
        {
         {-2.115362e+02, -1.874504e+03, -2.294189e+02, -1.516672e+03, -2.785281e+03, -2.925495e+03, -2.970340e+03, -4.293099e+02,
          -1.685521e+03, -5.322904e+02, -1.665030e+03, -2.835582e+03, -6.988780e+02, -1.668066e+03, -2.687101e+03, -2.258380e+03,
          -2.495897e+03, -1.672543e+03, -9.116663e+02, -1.675849e+03, -1.168386e+03, -1.974995e+03, -1.677669e+03, -1.470044e+03,
          -1.702233e+03, -1.860586e+03, -1.668134e+03, -1.143118e+03, -1.319865e+03, -1.653114e+03, -2.204908e+03, -1.995014e+03,
          -2.705687e+03}
        };
    for(Plato::OrdinalType tTimeStep=0; tTimeStep < tPressure.extent(0); tTimeStep++)
    {
        for(Plato::OrdinalType tOrdinal=0; tOrdinal< tPressure.extent(1); tOrdinal++)
        {
            //printf("X(%d,%d) = %e\n", tTimeStep, tOrdinal, tHostPressure(tTimeStep, tOrdinal));
            TEST_FLOATING_EQUALITY(tHostPressure(tTimeStep, tOrdinal), tGoldPress[tTimeStep][tOrdinal], tTolerance);
        }
    }

    // 5.2 test displacement
    auto tHostDisplacements = Kokkos::create_mirror(tDisplacements);
    Kokkos::deep_copy(tHostDisplacements, tDisplacements);
    std::vector<std::vector<Plato::Scalar>> tGoldDisp =
        {
         {0.0, -6.267770e-02, 0.0, -6.250715e-02, 5.054901e-04, -6.174772e-02, -3.494325e-04, -6.164854e-02, -1.189951e-03,
          -6.163677e-02, 0.0, -6.243005e-02, -2.395852e-03, -5.908745e-02, 9.381758e-04, -5.919208e-02, -7.291411e-04, -5.909716e-02,
          1.326328e-03, -5.503616e-02, -1.099687e-03, -5.494402e-02, -3.525908e-03, -5.492911e-02, 1.629318e-03, -4.941788e-02,  -1.472318e-03,
          -4.933201e-02, -4.573797e-03, -4.931350e-02, -6.306177e-03, -3.454268e-02, -5.510012e-03, -4.243363e-02, -1.845476e-03, -4.245746e-02,
          1.819180e-03, -4.253584e-02, -2.219095e-03, -3.457328e-02, 1.868041e-03, -3.464274e-02, -6.934208e-03, -2.594957e-02, -2.593272e-03,
          -2.598862e-02, 1.747752e-03, -2.604802e-02, -2.966076e-03, -1.706881e-02, 1.432299e-03, -1.711426e-02, -7.365046e-03, -1.702033e-02,
          -7.602023e-03, 1.234104e-04,  -7.582309e-03, -8.182097e-03, -3.335936e-03, -8.239034e-03, 8.764536e-04, -8.256626e-03, -3.587180e-03,
          1.188541e-05, 0.0, 0.0}
        };
    for(Plato::OrdinalType tTimeStep=0; tTimeStep < tDisplacements.extent(0); tTimeStep++)
    {
        for(Plato::OrdinalType tOrdinal=0; tOrdinal< tDisplacements.extent(1); tOrdinal++)
        {
            //printf("X(%d,%d) = %e\n", tTimeStep, tOrdinal, tHostDisplacements(tTimeStep, tOrdinal));
            TEST_FLOATING_EQUALITY(tHostDisplacements(tTimeStep, tOrdinal), tGoldDisp[tTimeStep][tOrdinal], tTolerance);
        }
    }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.output("SimplySupportedBeamTraction2D");
    }
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_SimplySupportedBeamPressure2D_Elastic)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.24'/>                     \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e7'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0'/>         \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0'/>         \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='16e3'/>               \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='10'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "  <ParameterList  name='Mechanical Natural Boundary Conditions'>                         \n"
      "   <ParameterList  name='Traction Vector Boundary Condition'>                            \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform Pressure'/>         \n"
      "     <Parameter  name='Value'    type='string'        value='100*t'/>                    \n"
      "     <Parameter  name='Sides'    type='string'        value='Load'/>                     \n"
      "   </ParameterList>                                                                      \n"
      "  </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='XFixed'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );
    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,10,2);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;

    // 1. Construct plasticity problem
    auto tFaceIDs1 = PlatoUtestHelpers::get_edge_ids_on_y1(*tMesh);
    tMeshSets[Omega_h::SIDE_SET]["Load"] = tFaceIDs1;

    auto tDirichletBoundaryNodesX0 = PlatoUtestHelpers::get_2D_boundary_nodes_x0(*tMesh);
    tMeshSets[Omega_h::NODE_SET]["XFixed"] = tDirichletBoundaryNodesX0;

    Omega_h::LOs tPinnedNodeLOs({32});
    tMeshSets[Omega_h::NODE_SET]["Pinned"] = tPinnedNodeLOs;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls);

    // 5. Test results
    auto tState = tSolution.get("State");
    Plato::ScalarMultiVector tPressure("Pressure", tState.extent(0), tNumVertices);
    Plato::ScalarMultiVector tDisplacements("Displacements", tState.extent(0), tNumVertices*tSpaceDim);
    Plato::blas2::extract<PhysicsT::mNumDofsPerNode, PhysicsT::mPressureDofOffset>(tState, tPressure);
    Plato::blas2::extract<PhysicsT::mNumDofsPerNode, tSpaceDim>(tNumVertices, tState, tDisplacements);

    // 5.1 test pressure
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostPressure = Kokkos::create_mirror(tPressure);
    Kokkos::deep_copy(tHostPressure, tPressure);
    std::vector<std::vector<Plato::Scalar>> tGoldPress =
        {
         {-2.115362e+02, -1.874504e+03, -2.294189e+02, -1.516672e+03, -2.785281e+03, -2.925495e+03, -2.970340e+03, -4.293099e+02,
          -1.685521e+03, -5.322904e+02, -1.665030e+03, -2.835582e+03, -6.988780e+02, -1.668066e+03, -2.687101e+03, -2.258380e+03,
          -2.495897e+03, -1.672543e+03, -9.116663e+02, -1.675849e+03, -1.168386e+03, -1.974995e+03, -1.677669e+03, -1.470044e+03,
          -1.702233e+03, -1.860586e+03, -1.668134e+03, -1.143118e+03, -1.319865e+03, -1.653114e+03, -2.204908e+03, -1.995014e+03,
          -2.705687e+03}
        };
    for(Plato::OrdinalType tTimeStep=0; tTimeStep < tPressure.extent(0); tTimeStep++)
    {
        for(Plato::OrdinalType tOrdinal=0; tOrdinal< tPressure.extent(1); tOrdinal++)
        {
            //printf("X(%d,%d) = %e\n", tTimeStep, tOrdinal, tHostPressure(tTimeStep, tOrdinal));
            TEST_FLOATING_EQUALITY(tHostPressure(tTimeStep, tOrdinal), tGoldPress[tTimeStep][tOrdinal], tTolerance);
        }
    }

    // 5.2 test displacement
    auto tHostDisplacements = Kokkos::create_mirror(tDisplacements);
    Kokkos::deep_copy(tHostDisplacements, tDisplacements);
    std::vector<std::vector<Plato::Scalar>> tGoldDisp =
        {
         {0.0, -6.267770e-02, 0.0, -6.250715e-02, 5.054901e-04, -6.174772e-02, -3.494325e-04, -6.164854e-02, -1.189951e-03,
          -6.163677e-02, 0.0, -6.243005e-02, -2.395852e-03, -5.908745e-02, 9.381758e-04, -5.919208e-02, -7.291411e-04, -5.909716e-02,
          1.326328e-03, -5.503616e-02, -1.099687e-03, -5.494402e-02, -3.525908e-03, -5.492911e-02, 1.629318e-03, -4.941788e-02,  -1.472318e-03,
          -4.933201e-02, -4.573797e-03, -4.931350e-02, -6.306177e-03, -3.454268e-02, -5.510012e-03, -4.243363e-02, -1.845476e-03, -4.245746e-02,
          1.819180e-03, -4.253584e-02, -2.219095e-03, -3.457328e-02, 1.868041e-03, -3.464274e-02, -6.934208e-03, -2.594957e-02, -2.593272e-03,
          -2.598862e-02, 1.747752e-03, -2.604802e-02, -2.966076e-03, -1.706881e-02, 1.432299e-03, -1.711426e-02, -7.365046e-03, -1.702033e-02,
          -7.602023e-03, 1.234104e-04,  -7.582309e-03, -8.182097e-03, -3.335936e-03, -8.239034e-03, 8.764536e-04, -8.256626e-03, -3.587180e-03,
          1.188541e-05, 0.0, 0.0}
        };
    for(Plato::OrdinalType tTimeStep=0; tTimeStep < tDisplacements.extent(0); tTimeStep++)
    {
        for(Plato::OrdinalType tOrdinal=0; tOrdinal< tDisplacements.extent(1); tOrdinal++)
        {
            //printf("X(%d,%d) = %e\n", tTimeStep, tOrdinal, tHostDisplacements(tTimeStep, tOrdinal));
            TEST_FLOATING_EQUALITY(tHostDisplacements(tTimeStep, tOrdinal), tGoldDisp[tTimeStep][tOrdinal], tTolerance);
        }
    }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.output("SimplySupportedBeamPressure2D");
    }
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_SimplySupportedBeamPressure2D_2ElasticSteps_EPETRA)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Linear Solver'>                                                   \n"
      "    <Parameter name='Solver Stack' type='string' value='Epetra'/>                         \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.24'/>                     \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e7'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0'/>         \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0'/>         \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='16e3'/>               \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='2'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='2'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='10'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "  <ParameterList  name='Mechanical Natural Boundary Conditions'>                         \n"
      "   <ParameterList  name='Traction Vector Boundary Condition'>                            \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform Pressure'/>         \n"
      "     <Parameter  name='Value'    type='string'        value='100*t'/>                     \n"
      "     <Parameter  name='Sides'    type='string'        value='Load'/>                     \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='XFixed'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,10,2);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;

    // 1. Construct plasticity problem
    auto tFaceIDs1 = PlatoUtestHelpers::get_edge_ids_on_y1(*tMesh);
    tMeshSets[Omega_h::SIDE_SET]["Load"] = tFaceIDs1;

    auto tDirichletBoundaryNodesX0 = PlatoUtestHelpers::get_2D_boundary_nodes_x0(*tMesh);
    tMeshSets[Omega_h::NODE_SET]["XFixed"] = tDirichletBoundaryNodesX0;

    Omega_h::LOs tPinnedNodeLOs({32});
    tMeshSets[Omega_h::NODE_SET]["Pinned"] = tPinnedNodeLOs;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls);

    // 5. Test results - test only final time step
    auto tState = tSolution.get("State");
    Plato::ScalarMultiVector tPressure("Pressure", tState.extent(0), tNumVertices);
    Plato::ScalarMultiVector tDisplacements("Displacements", tState.extent(0), tNumVertices*tSpaceDim);
    Plato::blas2::extract<PhysicsT::mNumDofsPerNode, PhysicsT::mPressureDofOffset>(tState, tPressure);
    Plato::blas2::extract<PhysicsT::mNumDofsPerNode, tSpaceDim>(tNumVertices, tState, tDisplacements);

    // 5.1 test pressure
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostPressure = Kokkos::create_mirror(tPressure);
    Kokkos::deep_copy(tHostPressure, tPressure);
    std::vector<Plato::Scalar> tGoldPress =
        {
         {2.311752e+02,-1.827696e+03,2.247582e+02,-1.485268e+03,-3.159835e+03,-3.310358e+03,
         -3.355509e+03,-1.893416e+01,-1.667250e+03,-1.538996e+02,-1.643017e+03,-3.189432e+03,
         -3.715330e+02,-1.644462e+03,-2.992111e+03,-2.432445e+03,-2.743322e+03,-1.650144e+03,
         -6.506333e+02,-1.652914e+03,-9.859537e+02,-2.062321e+03,-1.655154e+03,-1.383581e+03,
         -1.678354e+03,-1.883450e+03,-1.657927e+03,-9.691324e+02,-1.185495e+03,-1.607423e+03,
         -2.312137e+03,-1.975908e+03,-2.933662e+03}
        };
    for(Plato::OrdinalType tTimeStep=1; tTimeStep < tPressure.extent(0); tTimeStep++)
    {
        for(Plato::OrdinalType tOrdinal=0; tOrdinal< tPressure.extent(1); tOrdinal++)
        {
            //printf("%e\n", tHostPressure(tTimeStep, tOrdinal));
            TEST_FLOATING_EQUALITY(tHostPressure(tTimeStep, tOrdinal), tGoldPress[tOrdinal], tTolerance);
        }
    }

    // 5.2 test displacement
    auto tHostDisplacements = Kokkos::create_mirror(tDisplacements);
    Kokkos::deep_copy(tHostDisplacements, tDisplacements);
    std::vector<Plato::Scalar> tGoldDisp =
         {0.000000e+00,-6.134811e-02,0.000000e+00,-6.119720e-02,4.890601e-04,-6.043469e-02,
         -3.458934e-04,-6.035885e-02,-1.165818e-03,-6.032728e-02,0.000000e+00,-6.109650e-02,
         -2.347652e-03,-5.783824e-02,9.053655e-04,-5.794102e-02,-7.213166e-04,-5.786790e-02,
         1.280989e-03,-5.388596e-02,-1.085814e-03,-5.381352e-02,-3.452896e-03,-5.378025e-02,
         1.573984e-03,-4.840284e-02,-1.452818e-03,-4.833418e-02,-4.479451e-03,-4.830023e-02,
         -6.182190e-03,-3.387425e-02,-5.398447e-03,-4.158341e-02,-1.820375e-03,-4.161922e-02,
         1.757783e-03,-4.168384e-02,-2.188273e-03,-3.391259e-02,1.805654e-03,-3.397251e-02,
         -6.803693e-03,-2.547071e-02,-2.556661e-03,-2.551252e-02,1.690486e-03,-2.556736e-02,
         -2.923613e-03,-1.677144e-02,1.387154e-03,-1.681750e-02,-7.234932e-03,-1.672620e-02,
         -7.493506e-03,1.246463e-04,-7.461686e-03,-8.052684e-03,-3.289559e-03,-8.100860e-03,
         8.484803e-04,-8.124464e-03,-3.535933e-03,2.311506e-05,0.000000e+00,0.000000e+00};
    for(Plato::OrdinalType tTimeStep=1; tTimeStep < tDisplacements.extent(0); tTimeStep++)
    {
        for(Plato::OrdinalType tOrdinal=0; tOrdinal< tDisplacements.extent(1); tOrdinal++)
        {
            //printf("%e\n", tHostDisplacements(tTimeStep, tOrdinal));
            TEST_FLOATING_EQUALITY(tHostDisplacements(tTimeStep, tOrdinal), tGoldDisp[tOrdinal], tTolerance);
        }
    }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.output("SimplySupportedBeamPressure2D");
    }
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}


#ifdef PLATO_TPETRA
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_SimplySupportedBeamPressure2D_2ElasticSteps_TPETRA)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Linear Solver'>                                                   \n"
      "    <Parameter name='Solver Stack' type='string' value='Tpetra'/>                         \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.24'/>                     \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e7'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0'/>         \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0'/>         \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='16e3'/>               \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='2'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='2'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='10'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "  <ParameterList  name='Mechanical Natural Boundary Conditions'>                         \n"
      "   <ParameterList  name='Traction Vector Boundary Condition'>                            \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform Pressure'/>         \n"
      "     <Parameter  name='Value'    type='string'        value='100*t'/>                     \n"
      "     <Parameter  name='Sides'    type='string'        value='Load'/>                     \n"
      "   </ParameterList>                                                                      \n"
      "  </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='XFixed'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,10,2);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;

    // 1. Construct plasticity problem
    auto tFaceIDs1 = PlatoUtestHelpers::get_edge_ids_on_y1(*tMesh);
    tMeshSets[Omega_h::SIDE_SET]["Load"] = tFaceIDs1;

    auto tDirichletBoundaryNodesX0 = PlatoUtestHelpers::get_2D_boundary_nodes_x0(*tMesh);
    tMeshSets[Omega_h::NODE_SET]["XFixed"] = tDirichletBoundaryNodesX0;

    Omega_h::LOs tPinnedNodeLOs({32});
    tMeshSets[Omega_h::NODE_SET]["Pinned"] = tPinnedNodeLOs;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls);

    // 5. Test results - test only final time step
    auto tState = tSolution.get("State");
    Plato::ScalarMultiVector tPressure("Pressure", tState.extent(0), tNumVertices);
    Plato::ScalarMultiVector tDisplacements("Displacements", tState.extent(0), tNumVertices*tSpaceDim);
    Plato::blas2::extract<PhysicsT::mNumDofsPerNode, PhysicsT::mPressureDofOffset>(tState, tPressure);
    Plato::blas2::extract<PhysicsT::mNumDofsPerNode, tSpaceDim>(tNumVertices, tState, tDisplacements);

    // 5.1 test pressure
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostPressure = Kokkos::create_mirror(tPressure);
    Kokkos::deep_copy(tHostPressure, tPressure);
    std::vector<Plato::Scalar> tGoldPress =
        {
         {2.311752e+02,-1.827696e+03,2.247582e+02,-1.485268e+03,-3.159835e+03,-3.310358e+03,
         -3.355509e+03,-1.893416e+01,-1.667250e+03,-1.538996e+02,-1.643017e+03,-3.189432e+03,
         -3.715330e+02,-1.644462e+03,-2.992111e+03,-2.432445e+03,-2.743322e+03,-1.650144e+03,
         -6.506333e+02,-1.652914e+03,-9.859537e+02,-2.062321e+03,-1.655154e+03,-1.383581e+03,
         -1.678354e+03,-1.883450e+03,-1.657927e+03,-9.691324e+02,-1.185495e+03,-1.607423e+03,
         -2.312137e+03,-1.975908e+03,-2.933662e+03}
        };
    for(Plato::OrdinalType tTimeStep=1; tTimeStep < tPressure.extent(0); tTimeStep++)
    {
        for(Plato::OrdinalType tOrdinal=0; tOrdinal< tPressure.extent(1); tOrdinal++)
        {
            //printf("X(%d,%d) = %e\n", tTimeStep, tOrdinal, tHostPressure(tTimeStep, tOrdinal));
            TEST_FLOATING_EQUALITY(tHostPressure(tTimeStep, tOrdinal), tGoldPress[tOrdinal], tTolerance);
        }
    }

    // 5.2 test displacement
    auto tHostDisplacements = Kokkos::create_mirror(tDisplacements);
    Kokkos::deep_copy(tHostDisplacements, tDisplacements);
    std::vector<Plato::Scalar> tGoldDisp =
         {0.000000e+00,-6.134811e-02,0.000000e+00,-6.119720e-02,4.890601e-04,-6.043469e-02,
         -3.458934e-04,-6.035885e-02,-1.165818e-03,-6.032728e-02,0.000000e+00,-6.109650e-02,
         -2.347652e-03,-5.783824e-02,9.053655e-04,-5.794102e-02,-7.213166e-04,-5.786790e-02,
         1.280989e-03,-5.388596e-02,-1.085814e-03,-5.381352e-02,-3.452896e-03,-5.378025e-02,
         1.573984e-03,-4.840284e-02,-1.452818e-03,-4.833418e-02,-4.479451e-03,-4.830023e-02,
         -6.182190e-03,-3.387425e-02,-5.398447e-03,-4.158341e-02,-1.820375e-03,-4.161922e-02,
         1.757783e-03,-4.168384e-02,-2.188273e-03,-3.391259e-02,1.805654e-03,-3.397251e-02,
         -6.803693e-03,-2.547071e-02,-2.556661e-03,-2.551252e-02,1.690486e-03,-2.556736e-02,
         -2.923613e-03,-1.677144e-02,1.387154e-03,-1.681750e-02,-7.234932e-03,-1.672620e-02,
         -7.493506e-03,1.246463e-04,-7.461686e-03,-8.052684e-03,-3.289559e-03,-8.100860e-03,
         8.484803e-04,-8.124464e-03,-3.535933e-03,2.311506e-05,0.000000e+00,0.000000e+00};
    for(Plato::OrdinalType tTimeStep=1; tTimeStep < tDisplacements.extent(0); tTimeStep++)
    {
        for(Plato::OrdinalType tOrdinal=0; tOrdinal< tDisplacements.extent(1); tOrdinal++)
        {
            //printf("X(%d,%d) = %e\n", tTimeStep, tOrdinal, tHostDisplacements(tTimeStep, tOrdinal));
            TEST_FLOATING_EQUALITY(tHostDisplacements(tTimeStep, tOrdinal), tGoldDisp[tOrdinal], tTolerance);
        }
    }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.output("SimplySupportedBeamPressure2D");
    }
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}
#endif


#ifdef HAVE_AMGX
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_SimplySupportedBeamPressure2D_2ElasticSteps_AMGX)
{
    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Linear Solver'>                                                   \n"
      "    <Parameter name='Solver Stack' type='string' value='AmgX'/>                         \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.24'/>                     \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e7'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0'/>         \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0'/>         \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='16e3'/>               \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='2'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='2'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='10'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "  <ParameterList  name='Mechanical Natural Boundary Conditions'>                         \n"
      "   <ParameterList  name='Traction Vector Boundary Condition'>                            \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform Pressure'/>         \n"
      "     <Parameter  name='Value'    type='string'        value='100*t'/>                    \n"
      "     <Parameter  name='Sides'    type='string'        value='Load'/>                     \n"
      "   </ParameterList>                                                                      \n"
      "  </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='XFixed'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,10,2);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    auto tFaceIDs = PlatoUtestHelpers::get_edge_ids_on_y1(*tMesh);
    tMeshSets[Omega_h::SIDE_SET]["Load"] = tFaceIDs;
    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;

    // 1. Construct plasticity problem
    auto tFaceIDs1 = PlatoUtestHelpers::get_edge_ids_on_y1(*tMesh);
    tMeshSets[Omega_h::SIDE_SET]["Load"] = tFaceIDs1;

    auto tDirichletBoundaryNodesX0 = PlatoUtestHelpers::get_2D_boundary_nodes_x0(*tMesh);
    tMeshSets[Omega_h::NODE_SET]["XFixed"] = tDirichletBoundaryNodesX0;

    Omega_h::LOs tPinnedNodeLOs({32});
    tMeshSets[Omega_h::NODE_SET]["Pinned"] = tPinnedNodeLOs;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls);

    // 5. Test results - test only final time step
    auto tState = tSolution.get("State");
    Plato::ScalarMultiVector tPressure("Pressure", tState.extent(0), tNumVertices);
    Plato::ScalarMultiVector tDisplacements("Displacements", tState.extent(0), tNumVertices*tSpaceDim);
    Plato::blas2::extract<PhysicsT::mNumDofsPerNode, PhysicsT::mPressureDofOffset>(tState, tPressure);
    Plato::blas2::extract<PhysicsT::mNumDofsPerNode, tSpaceDim>(tNumVertices, tState, tDisplacements);

    // 5.1 test pressure
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostPressure = Kokkos::create_mirror(tPressure);
    Kokkos::deep_copy(tHostPressure, tPressure);
    std::vector<Plato::Scalar> tGoldPress =
        {
         {2.311752e+02,-1.827696e+03,2.247582e+02,-1.485268e+03,-3.159835e+03,-3.310358e+03,
         -3.355509e+03,-1.893416e+01,-1.667250e+03,-1.538996e+02,-1.643017e+03,-3.189432e+03,
         -3.715330e+02,-1.644462e+03,-2.992111e+03,-2.432445e+03,-2.743322e+03,-1.650144e+03,
         -6.506333e+02,-1.652914e+03,-9.859537e+02,-2.062321e+03,-1.655154e+03,-1.383581e+03,
         -1.678354e+03,-1.883450e+03,-1.657927e+03,-9.691324e+02,-1.185495e+03,-1.607423e+03,
         -2.312137e+03,-1.975908e+03,-2.933662e+03}
        };
    for(Plato::OrdinalType tTimeStep=1; tTimeStep < tPressure.extent(0); tTimeStep++)
    {
        for(Plato::OrdinalType tOrdinal=0; tOrdinal< tPressure.extent(1); tOrdinal++)
        {
            //printf("X(%d,%d) = %e\n", tTimeStep, tOrdinal, tHostPressure(tTimeStep, tOrdinal));
            TEST_FLOATING_EQUALITY(tHostPressure(tTimeStep, tOrdinal), tGoldPress[tOrdinal], tTolerance);
        }
    }

    // 5.2 test displacement
    auto tHostDisplacements = Kokkos::create_mirror(tDisplacements);
    Kokkos::deep_copy(tHostDisplacements, tDisplacements);
    std::vector<Plato::Scalar> tGoldDisp =
         {0.000000e+00,-6.134811e-02,0.000000e+00,-6.119720e-02,4.890601e-04,-6.043469e-02,
         -3.458934e-04,-6.035885e-02,-1.165818e-03,-6.032728e-02,0.000000e+00,-6.109650e-02,
         -2.347652e-03,-5.783824e-02,9.053655e-04,-5.794102e-02,-7.213166e-04,-5.786790e-02,
         1.280989e-03,-5.388596e-02,-1.085814e-03,-5.381352e-02,-3.452896e-03,-5.378025e-02,
         1.573984e-03,-4.840284e-02,-1.452818e-03,-4.833418e-02,-4.479451e-03,-4.830023e-02,
         -6.182190e-03,-3.387425e-02,-5.398447e-03,-4.158341e-02,-1.820375e-03,-4.161922e-02,
         1.757783e-03,-4.168384e-02,-2.188273e-03,-3.391259e-02,1.805654e-03,-3.397251e-02,
         -6.803693e-03,-2.547071e-02,-2.556661e-03,-2.551252e-02,1.690486e-03,-2.556736e-02,
         -2.923613e-03,-1.677144e-02,1.387154e-03,-1.681750e-02,-7.234932e-03,-1.672620e-02,
         -7.493506e-03,1.246463e-04,-7.461686e-03,-8.052684e-03,-3.289559e-03,-8.100860e-03,
         8.484803e-04,-8.124464e-03,-3.535933e-03,2.311506e-05,0.000000e+00,0.000000e+00};
    for(Plato::OrdinalType tTimeStep=1; tTimeStep < tDisplacements.extent(0); tTimeStep++)
    {
        for(Plato::OrdinalType tOrdinal=0; tOrdinal< tDisplacements.extent(1); tOrdinal++)
        {
            //printf("X(%d,%d) = %e\n", tTimeStep, tOrdinal, tHostDisplacements(tTimeStep, tOrdinal));
            TEST_FLOATING_EQUALITY(tHostDisplacements(tTimeStep, tOrdinal), tGoldDisp[tOrdinal], tTolerance);
        }
    }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.output("SimplySupportedBeamPressure2D");
    }
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}
#endif

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_SimplySupportedBeamPressure2D_PlasticSteps)
{
    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    const bool tDeleteSolverStats = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,10,2);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.24'/>                     \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e7'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0'/>         \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0'/>         \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='16e3'/>               \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                      \n"
      "    <Parameter name='Plottable' type='Array(string)' value='{elastic stress, accumulated plastic strain, plastic multiplier increment, elastic strain, deviatoric stress, plastic strain, backstress}'/>\n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='10'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "  <ParameterList  name='Mechanical Natural Boundary Conditions'>                         \n"
      "   <ParameterList  name='Traction Vector Boundary Condition'>                            \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform Pressure'/>         \n"
      "     <Parameter  name='Value'    type='string'        value='150*t'/>                    \n"
      "     <Parameter  name='Sides'    type='string'        value='Load'/>                     \n"
      "   </ParameterList>                                                                      \n"
      "  </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='XFixed'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;

    // 1. Construct plasticity problem
    auto tFaceIDs1 = PlatoUtestHelpers::get_edge_ids_on_y1(*tMesh);
    tMeshSets[Omega_h::SIDE_SET]["Load"] = tFaceIDs1;

    auto tDirichletBoundaryNodesX0 = PlatoUtestHelpers::get_2D_boundary_nodes_x0(*tMesh);
    tMeshSets[Omega_h::NODE_SET]["XFixed"] = tDirichletBoundaryNodesX0;

    Omega_h::LOs tPinnedNodeLOs({32});
    tMeshSets[Omega_h::NODE_SET]["Pinned"] = tPinnedNodeLOs;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls);
    
    auto tState = tSolution.get("State");
    auto tHostSolution = Kokkos::create_mirror(tState);
    Kokkos::deep_copy(tHostSolution, tState);

    std::vector<Plato::Scalar> tGold =
        {
           0.00000e+00, -9.24987e-02,  1.20097e+03,
           0.00000e+00, -9.23105e-02, -2.62729e+03,
           7.51325e-04, -9.09949e-02,  1.15762e+03,
          -5.84138e-04, -9.09250e-02, -2.29655e+03,
          -1.84799e-03, -9.08052e-02, -5.65113e+03,
           0.00000e+00, -9.20274e-02, -5.76246e+03,
          -3.73117e-03, -8.68735e-02, -5.90141e+03,
           1.35680e-03, -8.70326e-02,  7.69127e+02,
          -1.17885e-03, -8.69632e-02, -2.51624e+03,
           1.89395e-03, -8.07844e-02,  5.18011e+02,
          -1.71680e-03, -8.07148e-02, -2.41709e+03,
          -5.32817e-03, -8.06262e-02, -5.47771e+03,
           2.30488e-03, -7.24758e-02,  7.65981e+01,
          -2.26024e-03, -7.24074e-02, -2.43157e+03,
          -6.82502e-03, -7.23248e-02, -5.11104e+03,
          -9.31590e-03, -5.06970e-02, -4.01498e+03,
          -8.16719e-03, -6.22313e-02, -4.62453e+03,
          -2.80340e-03, -6.23101e-02, -2.44061e+03,
           2.56041e-03, -6.23794e-02, -4.71812e+02,
          -3.34687e-03, -5.07713e-02, -2.44270e+03,
           2.62205e-03, -5.08418e-02, -1.12974e+03,
          -1.02327e-02, -3.81471e-02, -3.29142e+03,
          -3.89088e-03, -3.82169e-02, -2.44644e+03,
           2.45115e-03, -3.82896e-02, -1.91496e+03,
          -4.43252e-03, -2.51506e-02, -2.47836e+03,
           2.01166e-03, -2.52203e-02, -2.87279e+03,
          -1.08775e-02, -2.50869e-02, -2.49433e+03,
          -1.13011e-02,  1.88454e-04, -1.14844e+03,
          -1.12318e-02, -1.21057e-02, -1.56731e+03,
          -4.97535e-03, -1.21635e-02, -2.33822e+03,
           1.22980e-03, -1.22093e-02, -3.68515e+03,
          -5.33903e-03,  5.43424e-05, -2.89933e+03,
           0.00000e+00,  0.00000e+00, -4.82496e+03
        };

    const Plato::Scalar tTolerance = 1e-4;
    const Plato::OrdinalType tDim1 = tState.extent(1);
    for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
    {
        //printf("X(%d,%d) = %f\n", 3, tIndexJ, tHostInput(3, tIndexJ));
        TEST_FLOATING_EQUALITY(tHostSolution(3, tIndexJ), tGold[tIndexJ], tTolerance);
    }

    // auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    // Plato::OrdinalType tIdx = 0;
    // for (Plato::OrdinalType tIndexK = 0; tIndexK < tNumVertices; tIndexK++)
    // {
    //  tIdx = tIndexK * tNumDofsPerNode + 0; printf("%12.5e, ",  tHostSolution(3, tIdx));
    //  tIdx = tIndexK * tNumDofsPerNode + 1; printf("%12.5e, ",  tHostSolution(3, tIdx));
    //  tIdx = tIndexK * tNumDofsPerNode + 2; printf("%12.5e,\n", tHostSolution(3, tIdx));
    // }

    // 6. Output Data
    if(tOutputData)
    {
        tPlasticityProblem.output("SimplySupportedBeamPressure2D");
    }

    if(tDeleteSolverStats)
    {
        auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
        if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
    }
}

//#ifdef NOPE

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_CriterionTest_2D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-6'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.002*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Evaluate criterion
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);

    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tSolution = tPlasticityProblem.solution(tControls);
    std::string tCriterionName("Plastic Work");
    auto tCriterionValue = tPlasticityProblem.criterionValue(tControls, tCriterionName);
    TEST_FLOATING_EQUALITY(tCriterionValue, -1.07121, tTolerance);

    auto tCriterionGrad = tPlasticityProblem.criterionGradient(tControls, tCriterionName);
    std::vector<Plato::Scalar> tGold = {-1.128948e+00, -5.644739e-01, -1.128948e+00, -5.644739e-01};
    auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        //printf("%e\n", tHostGrad(tIndex));
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_TestCriterionGradientZ_2D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='2.0e3'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e-8'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='2'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='1.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='2.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Elastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='2.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function' type='string' value='0.005*t'/>                       \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. TEST PARTIAL DERIVATIVE
    std::string tCriterionName("Plastic Work");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    const Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_CriterionTest_3D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-6'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='My Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y1'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.002*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Evaluate criterion
    std::string tCriterionName("My Plastic Work");

    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);

    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tSolution = tPlasticityProblem.solution(tControls);
    auto tObjValue = tPlasticityProblem.criterionValue(tControls, tCriterionName);
    TEST_FLOATING_EQUALITY(tObjValue, -1.07121, tTolerance);

    auto tObjGrad = tPlasticityProblem.criterionGradient(tControls, tCriterionName);
    std::vector<Plato::Scalar> tGold =
        {
         -1.058389e-01,-1.411185e-01,-3.527962e-02,-2.116777e-01,-7.055924e-02,-3.527962e-02,
         -7.055924e-02,-3.527962e-02,-1.411185e-01,-2.116777e-01,-7.055924e-02,-3.527962e-02,
         -7.055924e-02,-4.233554e-01,-2.116777e-01,-1.411185e-01,-2.116777e-01,-1.411185e-01,
         -1.058389e-01,-1.411185e-01,-2.116777e-01,-2.116777e-01,-7.055924e-02,-3.527962e-02,
         -7.055924e-02,-1.411185e-01,-3.527962e-02
        };
    auto tHostGrad = Kokkos::create_mirror(tObjGrad);
    Kokkos::deep_copy(tHostGrad, tObjGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        //printf("%e\n", tHostGrad(tIndex));
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_TestCriterionGradientZ_3D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Iterations' type='int' value='5000'/>                             \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-16'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='1.0'/>                   \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e4'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e2'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e2'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e2'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='2'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='1.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='2.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='2.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='2'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='2'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='50'/>                  \n"
      "    <Parameter name='Stopping Tolerance' type='double' value='1e-14'/>                    \n"
      "    <Parameter name='Current Residual Norm Stopping Tolerance' type='double' value='1e-14'/> \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.035*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. TEST PARTIAL DERIVATIVE
    std::string tCriterionName("Plastic Work");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    constexpr Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}

#ifdef HAVE_AMGX
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ObjectiveTest_2D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-6'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "    <Parameter name='Stopping Tolerance' type='double' value='1e-14'/>                    \n"
      "    <Parameter name='Current Residual Norm Stopping Tolerance' type='double' value='1e-14'/> \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.002*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Evaluate criterion
    std::string tCriterionName("Plastic Work");

    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);

    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tSolution = tPlasticityProblem.solution(tControls);
    auto tObjValue = tPlasticityProblem.criterionValue(tControls, tSolution, tCriterionName);
    TEST_FLOATING_EQUALITY(tObjValue, -1.07121, tTolerance);

    auto tObjGrad = tPlasticityProblem.criterionGradient(tControls, tSolution, tCriterionName);
    std::vector<Plato::Scalar> tGold = {-1.128948e+00,-5.644739e-01,-1.128948e+00,-5.644739e-01};
    auto tHostGrad = Kokkos::create_mirror(tObjGrad);
    Kokkos::deep_copy(tHostGrad, tObjGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        //printf("%e\n", tHostGrad(tIndex));
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}
#endif

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_TestObjectiveGradientZ_2D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='2'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='1.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='2.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='My Plastic Work'>                                               \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='2.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.002*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. TEST PARTIAL DERIVATIVE
    std::string tCriterionName("My Plastic Work");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    const Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_ObjectiveTest_3D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-6'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y1'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.002*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Evaluate criterion
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);

    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tSolution = tPlasticityProblem.solution(tControls);
    std::string tCriterionName("Plastic Work");
    auto tObjValue = tPlasticityProblem.criterionValue(tControls, tSolution, tCriterionName);
    TEST_FLOATING_EQUALITY(tObjValue, -1.07121, tTolerance);

    auto tObjGrad = tPlasticityProblem.criterionGradient(tControls, tSolution, tCriterionName);
    std::vector<Plato::Scalar> tGold =
        {
         -1.058389e-01,-1.411185e-01,-3.527962e-02,-2.116777e-01,-7.055924e-02,-3.527962e-02,
         -7.055924e-02,-3.527962e-02,-1.411185e-01,-2.116777e-01,-7.055924e-02,-3.527962e-02,
         -7.055924e-02,-4.233554e-01,-2.116777e-01,-1.411185e-01,-2.116777e-01,-1.411185e-01,
         -1.058389e-01,-1.411185e-01,-2.116777e-01,-2.116777e-01,-7.055924e-02,-3.527962e-02,
         -7.055924e-02,-1.411185e-01,-3.527962e-02
        };
    auto tHostGrad = Kokkos::create_mirror(tObjGrad);
    Kokkos::deep_copy(tHostGrad, tObjGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        //printf("%e\n", tHostGrad(tIndex));
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_TestObjectiveGradientZ_3D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='1.0e6'/>                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='2'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='1.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='2.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='2.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y1'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.002*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. TEST PARTIAL DERIVATIVE
    std::string tCriterionName("Plastic Work");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    constexpr Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_TestElasticWorkCriterion_GradientZ_2D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='2'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='1.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='2.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Elastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Elastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='2.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.002*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. TEST PARTIAL DERIVATIVE
    std::string tCriterionName("Elastic Work");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    const Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_TestElasticWorkCriterion_GradientZ_3D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='1.0e2'/>                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='2'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='1.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='2.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Elastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Elastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='2.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y1'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.002*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. TEST PARTIAL DERIVATIVE
    std::string tCriterionName("Elastic Work");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    constexpr Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_TestWeightedSumCriterion_GradientZ_2D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='2'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='1.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='2.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Objective'>                                                               \n"
      "      <Parameter name='Type' type='string' value='Weighted Sum'/>                                  \n"
      "      <Parameter name='Functions' type='Array(string)' value='{My Elastic Work,My Plastic Work}'/> \n"
      "      <Parameter name='Weights' type='Array(double)' value='{-1.0,-1.0}'/>                         \n"
      "    </ParameterList>                                                                               \n"
      "    <ParameterList name='My Elastic Work'>                                                         \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>               \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Elastic Work'/>                  \n"
      "      <Parameter name='Exponent'             type='double' value='2.0'/>                           \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>                        \n"
      "    </ParameterList>                                                                               \n"
      "    <ParameterList name='My Plastic Work'>                                                         \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>               \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>                  \n"
      "      <Parameter name='Exponent'             type='double' value='2.0'/>                           \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>                        \n"
      "    </ParameterList>                                                                               \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.005*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. TEST PARTIAL DERIVATIVE
    std::string tCriterionName("Objective");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    const Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ElastoPlasticity_TestWeightedSumCriterion_GradientZ_3D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 2;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='1.0e6'/>                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                      \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                            \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                    \n"
      "      </ParameterList>                                                                     \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='2'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-9'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='1.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='2.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Objective'>                                                               \n"
      "      <Parameter name='Type' type='string' value='Weighted Sum'/>                                  \n"
      "      <Parameter name='Functions' type='Array(string)' value='{My Elastic Work,My Plastic Work}'/> \n"
      "      <Parameter name='Weights' type='Array(double)' value='{-1.0,-1.0}'/>                         \n"
      "    </ParameterList>                                                                               \n"
      "    <ParameterList name='My Elastic Work'>                                                         \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>               \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Elastic Work'/>                  \n"
      "      <Parameter name='Exponent'             type='double' value='2.0'/>                           \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>                        \n"
      "    </ParameterList>                                                                               \n"
      "    <ParameterList name='My Plastic Work'>                                                         \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>               \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>                  \n"
      "      <Parameter name='Exponent'             type='double' value='2.0'/>                           \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>                        \n"
      "    </ParameterList>                                                                               \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y1'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.002*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. TEST PARTIAL DERIVATIVE
    std::string tCriterionName("Objective");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    constexpr Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    auto tSysMsg = std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
    if(false){ std::cout << std::to_string(tSysMsg) << "\n"; }
}

//#endif
}

/*
 * StabilizedMechanicsTests.cpp
 *
 *  Created on: Mar 26, 2020
 */

#include <Teuchos_XMLParameterListCoreHelpers.hpp>

#include "Teuchos_UnitTestHarness.hpp"

#include "UtilsOmegaH.hpp"
#include "PlatoUtilities.hpp"
#include "PlatoTestHelpers.hpp"
#include "EllipticVMSProblem.hpp"


namespace StabilizedMechanicsTests
{


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StabilizedMechanics_Kinematics3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // Set configuration workset
    auto tNumCells = tMesh->nelems();
    using PhysicsT = Plato::StabilizedMechanics<tSpaceDim>;
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);
    Plato::ScalarArray3D tConfig("configuration", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfig);

    // Set state workset
    auto tNumNodes = tMesh->nverts();
    auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    Plato::ScalarVector tState("state", tNumDofsPerNode * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // disp_z
        tState(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVector tStateWS("current state", tNumCells, PhysicsT::mNumDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    Plato::ScalarVector tCellVolume("cell volume", tNumCells);
    Plato::ScalarMultiVector tStrains("strains", tNumCells, tNumVoigtTerms);
    Plato::ScalarMultiVector tPressGrad("pressure grad", tNumCells, tSpaceDim);
    Plato::ScalarArray3D tGradient("gradient", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);

    Plato::StabilizedKinematics <tSpaceDim> tKinematics;
    Plato::ComputeGradientWorkset <tSpaceDim> tComputeGradient;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfig, tCellVolume);
        tKinematics(aCellOrdinal, tStrains, tPressGrad, tStateWS, tGradient);
    }, "kinematics test");

    std::vector<std::vector<Plato::Scalar>> tGold =
        { {1e-7,  6e-7,  3e-7, 1.1e-6,   4e-7,   5e-7},
          {3e-7,  6e-7, -3e-7,   7e-7,   8e-7,   9e-7},
          {3e-7,  2e-7,  3e-7,   5e-7,   1e-6,   7e-7},
          {5e-7, -2e-7,  3e-7,  -1e-7, 1.6e-6,   9e-7},
          {7e-7, -2e-7, -3e-7,  -5e-7,   2e-6, 1.3e-6},
          {7e-7, -6e-7,  3e-7,  -7e-7, 2.2e-6, 1.1e-6} };
    auto tHostStrains = Kokkos::create_mirror(tStrains);
    Kokkos::deep_copy(tHostStrains, tStrains);

    const Plato::Scalar tTolerance = 1e-4;
    const Plato::OrdinalType tDim0 = tStrains.extent(0);
    const Plato::OrdinalType tDim1 = tStrains.extent(1);
    for (Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            //printf("X(%d,%d) = %f\n", tIndexI, tIndexJ, tHostInput(tIndexI, tIndexJ));
            TEST_FLOATING_EQUALITY(tHostStrains(tIndexI, tIndexJ), tGold[tIndexI][tIndexJ], tTolerance);
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StabilizedMechanics_Solution3D)
{
    // 1. DEFINE PROBLEM
    const bool tOutputData = false; // for debugging purpose, set true to enable the Paraview output file
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                               \n"
        "<Parameter name='Physics'         type='string'  value='Stabilized Mechanical'/> \n"
        "<Parameter name='PDE Constraint'  type='string'  value='Elliptic'/>              \n"
        "<ParameterList name='Elliptic'>                                                  \n"
          "<ParameterList name='Penalty Function'>                                        \n"
            "<Parameter name='Type' type='string' value='SIMP'/>                          \n"
            "<Parameter name='Exponent' type='double' value='3.0'/>                       \n"
            "<Parameter name='Minimum Value' type='double' value='1.0e-9'/>               \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Time Stepping'>                                             \n"
          "<Parameter name='Number Time Steps' type='int' value='2'/>                     \n"
          "<Parameter name='Time Step' type='double' value='1.0'/>                        \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Newton Iteration'>                                          \n"
          "<Parameter name='Number Iterations' type='int' value='3'/>                     \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Spatial Model'>                                             \n"
          "<ParameterList name='Domains'>                                                 \n"
            "<ParameterList name='Design Volume'>                                         \n"
              "<Parameter name='Element Block' type='string' value='body'/>               \n"
              "<Parameter name='Material Model' type='string' value='Playdoh'/>           \n"
            "</ParameterList>                                                             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Material Models'>                                           \n"
          "<ParameterList name='Playdoh'>                                                 \n"
            "<ParameterList name='Isotropic Linear Elastic'>                              \n"
              "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>             \n"
              "<Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>           \n"
            "</ParameterList>                                                             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
    "</ParameterList>                                                                     \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    using PhysicsT = Plato::StabilizedMechanics<tSpaceDim>;
    Plato::EllipticVMSProblem<PhysicsT> tEllipticVMSProblem(*tMesh, tMeshSets, *tParamList, tMachine);

    // 2. Get Dirichlet Boundary Conditions
    Plato::OrdinalType tDispDofX = 0;
    Plato::OrdinalType tDispDofY = 1;
    Plato::OrdinalType tDispDofZ = 2;
    constexpr Plato::OrdinalType tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    auto tDirichletIndicesBoundaryX0_Xdof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x0", tNumDofsPerNode, tDispDofX);
    auto tDirichletIndicesBoundaryX0_Ydof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x0", tNumDofsPerNode, tDispDofY);
    auto tDirichletIndicesBoundaryX0_Zdof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x0", tNumDofsPerNode, tDispDofZ);
    auto tDirichletIndicesBoundaryX1_Ydof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x1", tNumDofsPerNode, tDispDofY);

    // 3. Set Dirichlet Boundary Conditions
    Plato::Scalar tValueToSet = 0;
    auto tNumDirichletDofs = tDirichletIndicesBoundaryX0_Xdof.size() + tDirichletIndicesBoundaryX0_Ydof.size() +
            tDirichletIndicesBoundaryX0_Zdof.size() + tDirichletIndicesBoundaryX1_Ydof.size();
    Plato::ScalarVector tDirichletValues("Dirichlet Values", tNumDirichletDofs);
    Plato::LocalOrdinalVector tDirichletDofs("Dirichlet Dofs", tNumDirichletDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Xdof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tDirichletValues(aIndex) = tValueToSet;
        tDirichletDofs(aIndex) = tDirichletIndicesBoundaryX0_Xdof(aIndex);
    }, "set dirichlet values and indices");

    auto tOffset = tDirichletIndicesBoundaryX0_Xdof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Ydof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX0_Ydof(aIndex);
    }, "set dirichlet values and indices");

    tOffset += tDirichletIndicesBoundaryX0_Ydof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Zdof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX0_Zdof(aIndex);
    }, "set dirichlet values and indices");

    tValueToSet = -1e-3;
    tOffset += tDirichletIndicesBoundaryX0_Zdof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX1_Ydof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX1_Ydof(aIndex);
    }, "set dirichlet values and indices");
    tEllipticVMSProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);

    // 4. Solve problem
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tEllipticVMSProblem.solution(tControls);
    auto tState = tSolution.get("State");

    // 5. Test Results
    std::vector<std::vector<Plato::Scalar>> tGold =
        { {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, 0, -3.765995e-6, 0, 0, 0, -2.756658e-5, 0, 0, 0, 7.081654e-5, 0, 0, 0, 8.626534e-05,
           3.118233e-4, -1.0e-3, 4.815153e-5, 1.774578e-5, 2.340348e-4, -1.0e-3, 4.357691e-5, -3.765995e-6,
           -3.927496e-4, -1.0e-3, 5.100447e-5, -9.986030e-5, -1.803906e-4, -1.0e-3, 9.081316e-5, -6.999675e-5}};
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
        tEllipticVMSProblem.output("Output");
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StabilizedMechanics_Residual3D)
{
    // 1. PREPARE PROBLEM INPUS FOR TEST
    Plato::DataMap tDataMap;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tPDEInputs =
        Teuchos::getParametersFromXmlString(
        "<ParameterList name='Plato Problem'>                                           \n"
        "  <ParameterList name='Spatial Model'>                                         \n"
        "    <ParameterList name='Domains'>                                             \n"
        "      <ParameterList name='Design Volume'>                                     \n"
        "        <Parameter name='Element Block' type='string' value='body'/>           \n"
        "        <Parameter name='Material Model' type='string' value='Fancy Feast'/>   \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"
        "  <ParameterList name='Material Models'>                                       \n"
        "    <ParameterList name='Fancy Feast'>                                         \n"
        "      <ParameterList name='Isotropic Linear Elastic'>                          \n"
        "        <Parameter  name='Poissons Ratio' type='double' value='0.35'/>         \n"
        "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>       \n"
        "      </ParameterList>                                                         \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"
        "  <ParameterList name='Elliptic'>                                              \n"
        "    <ParameterList name='Penalty Function'>                                    \n"
        "      <Parameter name='Type' type='string' value='SIMP'/>                      \n"
        "      <Parameter name='Exponent' type='double' value='3.0'/>                   \n"
        "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>           \n"
        "    </ParameterList>                                                           \n"
        "  </ParameterList>                                                             \n"
        "</ParameterList>                                                               \n"
      );

    Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tPDEInputs);

    // 2. PREPARE FUNCTION INPUTS FOR TEST
    const Plato::OrdinalType tNumNodes = tMesh->nverts();
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    using PhysicsT = Plato::StabilizedMechanics<tSpaceDim>;
    using EvalType = typename Plato::Evaluation<PhysicsT>::Residual;
    Plato::WorksetBase<PhysicsT> tWorksetBase(*tMesh);

    // 2.1 SET CONFIGURATION
    Plato::ScalarArray3DT<EvalType::ConfigScalarType> tConfigWS("configuration", tNumCells, PhysicsT::mNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // 2.2 SET DESIGN VARIABLES
    Plato::ScalarMultiVectorT<EvalType::ControlScalarType> tControlWS("design variables", tNumCells, PhysicsT::mNumNodesPerCell);
    Kokkos::deep_copy(tControlWS, 1.0);

    // 2.3 SET GLOBAL STATE
    auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    Plato::ScalarVector tState("state", tNumDofsPerNode * tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tState(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal; // disp_x
        tState(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal; // disp_y
        tState(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal; // disp_z
        tState(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal; // press
    }, "set global state");
    Plato::ScalarMultiVectorT<EvalType::StateScalarType> tStateWS("current global state", tNumCells, PhysicsT::mNumDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // 2.4 SET PROJECTED PRESSURE GRADIENT
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    Plato::ScalarMultiVectorT<EvalType::NodeStateScalarType> tProjPressGradWS("projected pressure grad", tNumCells, PhysicsT::mNumNodeStatePerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNodeIndex=0; tNodeIndex< tNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDimIndex=0; tDimIndex< tSpaceDim; tDimIndex++)
            {
                tProjPressGradWS(aCellOrdinal, tNodeIndex*tSpaceDim+tDimIndex) = (4e-7)*(tNodeIndex+1)*(tDimIndex+1)*(aCellOrdinal+1);
            }
        }
    }, "set projected pressure grad");

    auto tOnlyDomain = tSpatialModel.Domains.front();

    // 3. CALL FUNCTION
    auto tPenaltyParams = tPDEInputs->sublist("Elliptic").sublist("Penalty Function");
    Plato::StabilizedElastostaticResidual<EvalType, Plato::MSIMP> tComputeResidual(tOnlyDomain, tDataMap, *tPDEInputs, tPenaltyParams);
    Plato::ScalarMultiVectorT<EvalType::ResultScalarType> tResidualWS("residual", tNumCells, PhysicsT::mNumDofsPerCell);
    tComputeResidual.evaluate(tStateWS, tProjPressGradWS, tControlWS, tConfigWS, tResidualWS);

    // 5. TEST RESULTS
    std::vector<std::vector<Plato::Scalar>> tGold =
    {
      {-3.086420e+03, -2.551440e+04, -6.790123e+03,  9.169188e+03,  1.687243e+04, -3.703704e+03, -1.934156e+04, -9.259259e+02,
       -1.625514e+04,  2.242798e+04,  4.320988e+03, -7.656002e+03,  2.469136e+03,  6.790123e+03,  2.181070e+04, -4.290964e+03},
      {-5.555556e+03, -2.345679e+04, -4.320988e+03,  8.243263e+03,  6.172840e+02,  1.913580e+04, -8.024691e+03, -1.531200e+04,
       -1.481481e+04, -1.234568e+03,  7.407407e+03,  1.160830e+04,  1.975309e+04,  5.555556e+03,  4.938272e+03, -1.194697e+04},
      {-6.172840e+03, -3.086420e+03, -1.522634e+04,  3.365038e+03, -1.090535e+04,  9.670782e+03, -3.086420e+03,  6.730076e+03,
        1.851852e+03, -1.090535e+04,  1.213992e+04,  2.271404e-07,  1.522634e+04,  4.320988e+03,  6.172840e+03, -1.009511e+04},
      {-9.876543e+03,  6.172840e+02, -2.345679e+04,  5.872604e+02,  2.037037e+04, -1.172840e+04,  1.049383e+04, -2.296801e+04,
        5.555556e+03,  1.728395e+04, -6.172840e+02,  5.872604e+02, -1.604938e+04, -6.172840e+03,  1.358025e+04,  1.068237e+04},
      { 2.880658e+04,  1.111111e+04, -1.646091e+04, -3.432771e+04,  4.320988e+03, -3.312757e+04,  3.189300e+04, -7.407407e+03,
        8.024691e+03,  3.004115e+04, -3.086420e+03, -4.042369e+03, -4.115226e+04, -8.024691e+03, -1.234568e+04,  1.614786e+04},
      { 2.983539e+04, -1.378601e+04,  1.790123e+04, -3.920594e+04,  1.358025e+04, -4.320988e+03,  3.168724e+04, -8.920594e+03,
        -6.790123e+03,  2.489712e+04, -3.600823e+04,  7.904597e+03, -3.662551e+04, -6.790123e+03, -1.358025e+04,  1.799971e+04}
    };

    auto tHostResidualWS = Kokkos::create_mirror(tResidualWS);
    Kokkos::deep_copy(tHostResidualWS, tResidualWS);
    constexpr Plato::Scalar tTolerance = 1e-4;
    for(Plato::OrdinalType tCellIndex=0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tDofIndex=0; tDofIndex< PhysicsT::mNumDofsPerCell; tDofIndex++)
        {
            //printf("residual(%d,%d) = %.10f\n", tCellIndex, tDofIndex, tHostElastoPlasticityResidual(tCellIndex, tDofIndex));
            TEST_FLOATING_EQUALITY(tHostResidualWS(tCellIndex, tDofIndex), tGold[tCellIndex][tDofIndex], tTolerance);
        }
    }
}


}
// namespace StabilizedMechanicsTests

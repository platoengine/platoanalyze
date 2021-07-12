/*
 *  VolumeAverageCriterionTests.cpp
 *
 *  Created on: July 8, 2021
 */

#include "Teuchos_UnitTestHarness.hpp"
#include "PlatoTestHelpers.hpp"
#include "Solutions.hpp"
#include "Plato_Diagnostics.hpp"
#include "elliptic/Problem.hpp"
#include "elliptic/WeightedSumFunction.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/VolumeAverageCriterion.hpp"


namespace VolumeAverageCriterionTests
{

TEUCHOS_UNIT_TEST(VolumeAverageCriterionTests, Test1)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::blas1::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            {   tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal);}, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                               \n"
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

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);
    Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::AugLagStressCriterionQuadratic<Residual,Plato::SimplexMechanics<tSpaceDim>> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("Lagrange Multiplier", tNumCells);
    Plato::blas1::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    const auto tLocalMeasure =
        std::make_shared<Plato::VonMisesLocalMeasure<Residual,Plato::SimplexMechanics<tSpaceDim>>>
        (tOnlyDomain, tCellStiffMatrix, "VonMises");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.00140738, 0.00750674, 0.00140738, 0.0183732, 0.0861314, 0.122407};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(0.237233, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(VolumeAverageCriterionTests, VolumeAverageVonMisesStressAxial_3D)
{
    const bool tOutputData = false;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    const Plato::Scalar tBoxWidth = 2.0;
    const Plato::OrdinalType tNumElemX = 1;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = PlatoUtestHelpers::build_3d_box_mesh(tBoxWidth,tBoxWidth,tBoxWidth,tNumElemX,tNumElemY,tNumElemZ);
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
      "  <Parameter name='Physics'          type='string'  value='Mechanical'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "      <Parameter name='Iterations' type='int' value='500'/>                              \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Density' type='double' value='1'/>                             \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.2'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e4'/>                  \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='VolAvgMisesStress'>                                             \n"
      "      <Parameter name='Type' type='string' value='Volume Average Criterion'/>            \n"
      "      <Parameter name='Local Measure Type' type='string' value='VonMises'/>              \n"
      "      <ParameterList name='Penalty Function'>                                            \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>                              \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
      "        <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                   \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Natural Boundary Conditions'>                                   \n"
      "   </ParameterList>                                                                      \n"
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
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied X Displacement Boundary Condition'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.2'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::Mechanics<tSpaceDim>;

    Plato::Elliptic::Problem<PhysicsT> tProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);

    // 5. Test results
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::string tCriterionName("VolAvgMisesStress");
    auto tCriterionValue = tProblem.criterionValue(tControls, tCriterionName);
    TEST_FLOATING_EQUALITY(tCriterionValue, 1000.0, tTolerance);

    auto tCriterionGrad = tProblem.criterionGradient(tControls, tSolution, tCriterionName);
    std::vector<Plato::Scalar> tGold = { -8.23158e-01,-2.74211e-01,-2.74205e-01,-2.74211e-01,-5.46915e-01,
                                         -1.09598e+00,-5.46915e-01,-1.09091e+00,-1.07737e+00,-5.40880e-01,
                                         -1.08590e+00,-5.40880e-01,-1.05793e+00,-5.26599e-01,-1.04844e+00,
                                         -5.26599e-01,-1.07226e+00,-5.33831e-01,-1.06304e+00,-5.33831e-01,
                                         -5.04852e-01,-1.00493e+00,-5.04852e-01,-1.01433e+00,-5.12007e-01,
                                         -1.01919e+00,-1.03386e+00,-5.19301e-01,-1.04332e+00,-5.19301e-01,
                                         -5.12007e-01,-1.02878e+00,-4.98050e-01,-1.00060e+00,-4.98050e-01,
                                         -9.91065e-01,-9.80656e-01,-4.92349e-01,-9.88215e-01,-4.92349e-01,
                                         -2.44243e-01,-7.35025e-01,-2.44243e-01,-2.43315e-01};
    auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        printf("%12.5e\n", tHostGrad(tIndex));
        //TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }

    // 6. Output Data
    if (tOutputData)
    {
        tProblem.output("VolumeAverageVonMisesStressAxial_3D");
    }
}

TEUCHOS_UNIT_TEST(VolumeAverageCriterionTests, VolumeAverageVonMisesStressShear_3D)
{
    const bool tOutputData = false;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    const Plato::Scalar tBoxWidth = 2.0;
    const Plato::OrdinalType tNumElemX = 1;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = PlatoUtestHelpers::build_3d_box_mesh(tBoxWidth,tBoxWidth,tBoxWidth,tNumElemX,tNumElemY,tNumElemZ);
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
      "  <Parameter name='Physics'          type='string'  value='Mechanical'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "      <Parameter name='Iterations' type='int' value='500'/>                              \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Density' type='double' value='1'/>                             \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.2'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e4'/>                  \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='VolAvgMisesStress'>                                             \n"
      "      <Parameter name='Type' type='string' value='Volume Average Criterion'/>            \n"
      "      <Parameter name='Local Measure Type' type='string' value='VonMises'/>              \n"
      "      <ParameterList name='Penalty Function'>                                            \n"
      "        <Parameter name='Type' type='string' value='SIMP'/>                              \n"
      "        <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
      "        <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                   \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Natural Boundary Conditions'>                                   \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X0 Fixed Y'>                                                  \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y0 Fixed X'>                                                  \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='X1 Applied Y'>                                                \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.2'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y1 Applied X'>                                                \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y1'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.2'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::Mechanics<tSpaceDim>;

    Plato::Elliptic::Problem<PhysicsT> tProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    Plato::Scalar tDensity = 0.9;
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(tDensity, tControls);
    auto tSolution = tProblem.solution(tControls);


    // 5. Test results
    Plato::Scalar tSimpPenalty = 1.0e-8 + (1.0 - 1.0e-8) * std::pow(tDensity, 3);
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::string tCriterionName("VolAvgMisesStress");
    auto tCriterionValue = tProblem.criterionValue(tControls, tCriterionName);
    TEST_FLOATING_EQUALITY(tCriterionValue, tSimpPenalty*1443.3756727, tTolerance);

    auto tCriterionGrad = tProblem.criterionGradient(tControls, tSolution, tCriterionName);
    std::vector<Plato::Scalar> tGold = { -8.23158e-01,-2.74211e-01,-2.74205e-01,-2.74211e-01,-5.46915e-01,
                                         -1.09598e+00,-5.46915e-01,-1.09091e+00,-1.07737e+00,-5.40880e-01,
                                         -1.08590e+00,-5.40880e-01,-1.05793e+00,-5.26599e-01,-1.04844e+00,
                                         -5.26599e-01,-1.07226e+00,-5.33831e-01,-1.06304e+00,-5.33831e-01,
                                         -5.04852e-01,-1.00493e+00,-5.04852e-01,-1.01433e+00,-5.12007e-01,
                                         -1.01919e+00,-1.03386e+00,-5.19301e-01,-1.04332e+00,-5.19301e-01,
                                         -5.12007e-01,-1.02878e+00,-4.98050e-01,-1.00060e+00,-4.98050e-01,
                                         -9.91065e-01,-9.80656e-01,-4.92349e-01,-9.88215e-01,-4.92349e-01,
                                         -2.44243e-01,-7.35025e-01,-2.44243e-01,-2.43315e-01};
    auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        printf("%12.5e\n", tHostGrad(tIndex));
        //TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }

    // 6. Output Data
    if (tOutputData)
    {
        tProblem.output("VolumeAverageVonMisesStressShear_3D");
    }
}

} // namespace VolumeAverageCriterionTests

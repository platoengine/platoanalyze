/*!
  These unit tests are for the transient dynamics formulation
*/

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"


#include "Simp.hpp"
#include "Ramp.hpp"
#include "Strain.hpp"
#include "Solutions.hpp"
#include "Heaviside.hpp"
#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "Plato_Solve.hpp"
#include "LinearStress.hpp"
#include "ProjectToNode.hpp"
#include "ApplyWeighting.hpp"
#include "StressDivergence.hpp"
#include "SimplexMechanics.hpp"
#include "ApplyConstraints.hpp"
#include "PlatoMathHelpers.hpp"
#include "InterpolateFromNodal.hpp"
#include "PlatoAbstractProblem.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "hyperbolic/HyperbolicVectorFunction.hpp"
#include "hyperbolic/HyperbolicPhysicsScalarFunction.hpp"
#include "hyperbolic/Newmark.hpp"
#include "hyperbolic/InertialContent.hpp"
#include "hyperbolic/ElastomechanicsResidual.hpp"
#include "hyperbolic/HyperbolicMechanics.hpp"
#include "hyperbolic/HyperbolicProblem.hpp"

template <class VectorFunctionT, class VectorT, class ControlT>
Plato::Scalar
testVectorFunction_Partial_z(
    VectorFunctionT& aVectorFunction,
    VectorT aU,
    VectorT aV,
    VectorT aA,
    ControlT aControl,
    Plato::Scalar aTimeStep)
{
    // compute initial R and dRdz
    auto tResidual = aVectorFunction.value(aU, aV, aA, aControl, aTimeStep);
    auto t_dRdz = aVectorFunction.gradient_z(aU, aV, aA, aControl, aTimeStep);

    Plato::ScalarVector tStep = Plato::ScalarVector("Step", aControl.extent(0));
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.025, 0.05, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);

    // compute F at z - step
    Plato::blas1::axpy(-1.0, tStep, aControl);
    auto tResidualNeg = aVectorFunction.value(aU, aV, aA, aControl, aTimeStep);

    // compute F at z + step
    Plato::blas1::axpy(2.0, tStep, aControl);
    auto tResidualPos = aVectorFunction.value(aU, aV, aA, aControl, aTimeStep);
    Plato::blas1::axpy(-1.0, tStep, aControl);

    // compute actual change in F over 2 * deltaZ
    Plato::blas1::axpy(-1.0, tResidualPos, tResidualNeg);
    auto tDeltaFD = Plato::blas1::norm(tResidualNeg);

    Plato::ScalarVector tDeltaR = Plato::ScalarVector("delta R", tResidual.extent(0));
    Plato::blas1::scale(2.0, tStep);
    Plato::VectorTimesMatrixPlusVector(tStep, t_dRdz, tDeltaR);
    auto tDeltaAD = Plato::blas1::norm(tDeltaR);

    // return error
    Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
    return std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
}

TEUCHOS_UNIT_TEST( TransientDynamicsProblemTests, 3D )
{
  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

  // create mesh sets
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(cSpaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // create input for transient mechanics problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>     \n"
    "  <Parameter name='Physics' type='string' value='Mechanical'/>            \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>              \n"
    "  <ParameterList name='Hyperbolic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>         \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Alyoominium'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Criteria'>                                         \n"
    "    <ParameterList name='Internal Energy'>                                \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>      \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/> \n"
    "      <ParameterList name='Penalty Function'>                             \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>       \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Alyoominium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>          \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>      \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>   \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Time Integration'>                                 \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>           \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>           \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>            \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>            \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                     \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>            \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>     \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0, 0}'/> \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                          \n"
  );

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  auto* tHyperbolicProblem =
    new Plato::HyperbolicProblem<Plato::Hyperbolic::Mechanics<cSpaceDim>>
    (*tMesh, tMeshSets, *tInputParams, tMachine);

  TEST_ASSERT(tHyperbolicProblem != nullptr);

  int tNumDofs = cSpaceDim*tMesh->nverts();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(1.0, tControl);


  /*****************************************************
   Test HyperbolicProblem::solution(aControl);
   *****************************************************/

  auto tSolution = tHyperbolicProblem->solution(tControl);
  auto tDisplacements = tSolution.get("State");
  auto tDisplacement = Kokkos::subview(tDisplacements, /*tStepIndex*/1, Kokkos::ALL());

  auto tDisplacement_Host = Kokkos::create_mirror_view( tDisplacement );
  Kokkos::deep_copy( tDisplacement_Host, tDisplacement );

  std::vector<Plato::Scalar> tDisplacement_gold = {
    3.61863560324768500405784e-11,  3.83849353315283790119312e-12,
    3.83849353315283790119312e-12,  1.48267383576000436797713e-12,
   -2.25624866593761909787027e-13, -2.73367053814262282760693e-13,
   -2.42591058524479259361185e-11, -4.09157004346611346509595e-12,
   -6.54292514491087131336281e-12, -2.46921404161311427477642e-11,
   -4.89891069925166286444859e-12, -4.89891069925166286444859e-12
  };

  for(int iNode=0; iNode<int(tDisplacement_gold.size()); iNode++){
    if(tDisplacement_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(tDisplacement_Host[iNode]) < 1e-3);
    } else {
      TEST_FLOATING_EQUALITY(
        tDisplacement_Host[iNode],
        tDisplacement_gold[iNode], 1e-3);
    }
  }


  /*****************************************************
   Test HyperbolicProblem::criterionValue(aControl);
   *****************************************************/

  auto tCriterionValue = tHyperbolicProblem->criterionValue(tControl, "Internal Energy");
  Plato::Scalar tCriterionValue_gold = 5.43649521380863686677761e-9;

  TEST_FLOATING_EQUALITY( tCriterionValue, tCriterionValue_gold, 1e-4);


  /*********************************************************
   Test HyperbolicProblem::criterionValue(aControl, aState);
   *********************************************************/

  tCriterionValue = tHyperbolicProblem->criterionValue(tControl, tSolution, "Internal Energy");
  TEST_FLOATING_EQUALITY( tCriterionValue, tCriterionValue_gold, 1e-4);


  /*****************************************************
   Test HyperbolicProblem::criterionGradient(aControl);
   *****************************************************/

  auto tCriterionGradient = tHyperbolicProblem->criterionGradient(tControl, "Internal Energy");


  /**************************************************************
   The gradients below are verified with FD check elsewhere. The
   calls below are to catch any signals that may be thrown.
   **************************************************************/

  /************************************************************
   Call HyperbolicProblem::criterionGradient(aControl, aState);
   ************************************************************/

  tCriterionGradient = tHyperbolicProblem->criterionGradient(tControl, tSolution, "Internal Energy");


  /*****************************************************
   Call HyperbolicProblem::criterionGradientX(aControl);
   *****************************************************/

  auto tCriterionGradientX = tHyperbolicProblem->criterionGradientX(tControl, "Internal Energy");


  /************************************************************
   Call HyperbolicProblem::criterionGradientX(aControl, aState);
   ************************************************************/

  tCriterionGradientX = tHyperbolicProblem->criterionGradientX(tControl, tSolution, "Internal Energy");



  // test criterionValue
  //
  auto tConstraintValue = tHyperbolicProblem->criterionValue(tControl, "Internal Energy");


  // test criterionValue
  //
  tConstraintValue = tHyperbolicProblem->criterionValue(tControl, tSolution, "Internal Energy");


  // test criterionGradient
  //
  auto tConstraintGradient = tHyperbolicProblem->criterionGradient(tControl, "Internal Energy");


  // test criterionGradient
  //
  tConstraintGradient = tHyperbolicProblem->criterionGradient(tControl, tSolution, "Internal Energy");


  // test criterionGradientX
  //
  auto tConstraintGradientX = tHyperbolicProblem->criterionGradientX(tControl, "Internal Energy");


  // test criterionGradientX
  //
  tConstraintGradientX = tHyperbolicProblem->criterionGradientX(tControl, tSolution, "Internal Energy");

}




TEUCHOS_UNIT_TEST( TransientMechanicsResidualTests, 3D_NoMass )
{
  // create input for transient mechanics residual
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                     \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>    \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>             \n"
    "  <ParameterList name='Hyperbolic'>                                      \n"
    "    <ParameterList name='Penalty Function'>                              \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>             \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>        \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='Soylent Green'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                   \n"
    "    <ParameterList name='Soylent Green'>                                   \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                      \n"
    "        <Parameter name='Mass Density' type='double' value='0.0'/>         \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>  \n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                       \n"
    "  <ParameterList name='Time Integration'>                                \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>           \n"
    "    <Parameter name='Time Step' type='double' value='1.0'/>              \n"
    "  </ParameterList>                                                       \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                    \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>           \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>    \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0, 0}'/>\n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>         \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "</ParameterList>                                                         \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

  // create mesh sets
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(cSpaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tInputParams);

  Plato::DataMap tDataMap;
  Plato::Hyperbolic::VectorFunction<::Plato::Hyperbolic::Mechanics<cSpaceDim>>
    tVectorFunction(tSpatialModel, tDataMap, *tInputParams, tInputParams->get<std::string>("PDE Constraint"));

  int tNumDofs = cSpaceDim*tMesh->nverts();
  Plato::ScalarVector tControl("control", tMesh->nverts());
  Plato::blas1::fill(1.0, tControl);

  // create mesh based displacement from host data
  //
  std::vector<Plato::Scalar> u_host( cSpaceDim*tMesh->nverts() );
  Plato::Scalar disp = 0.0, dval = 1.0e-7;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  Plato::ScalarVector tU = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

  Plato::ScalarVector tV("Velocity", tNumDofs);
  Plato::blas1::fill(0.0, tV);

  Plato::ScalarVector tA("Acceleration", tNumDofs);
  Plato::blas1::fill(1000.0, tA);

  auto tTimeStep = tInputParams->sublist("Time Integration").get<Plato::Scalar>("Time Step");

  /**************************************
   Test VectorFunction ref value (Fext)
   **************************************/

  auto tResidualZero = tVectorFunction.value(tV, tV, tA, tControl, tTimeStep);
  auto tResidualZero_Host = Kokkos::create_mirror_view( tResidualZero );
  Kokkos::deep_copy( tResidualZero_Host, tResidualZero );

  std::vector<Plato::Scalar> tResidualZero_gold = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 125.000000000000000000000, 0, 0,
    41.6666666666666642981909, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 125.000000000000000000000, 0, 0, 83.3333333333333285963818,
    0, 0, 125.000000000000000000000, 0, 0, 249.999999999999971578291, 0,
    0, 0, 0, 0, 0, 0, 0, 41.6666666666666642981909, 0, 0,
    125.000000000000000000000, 0, 0, 0, 0, 0, 83.3333333333333285963818,
    0, 0
  };

  for(int iNode=0; iNode<int(tResidualZero_gold.size()); iNode++){
    if(tResidualZero_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(tResidualZero_Host[iNode]) < 1e-12);
    } else {
      // R(u) = Fint - Fext;
      TEST_FLOATING_EQUALITY(-tResidualZero_Host[iNode], tResidualZero_gold[iNode], 1e-6);
    }
  }

  /**************************************
   Test VectorFunction value
   **************************************/

  auto tResidual = tVectorFunction.value(tU, tV, tA, tControl, tTimeStep);
  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  std::vector<Plato::Scalar> tResidual_gold = {
   -1.41160714285714272e6,  -755357.142857142724,  -849107.142857142724,
   -1.60178571428571409e6,  -926785.714285714319,  -450000.000000000000,
     -333928.571428571420,  -333928.571428571420,   292857.142857142782,
   -2.02232142857142864e6,  -468749.999999999767,  -634821.428571428638
  };

  for(int iNode=0; iNode<int(tResidual_gold.size()); iNode++){
    if(tResidual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(tResidual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tResidual_Host[iNode], tResidual_gold[iNode], 1e-13);
    }
  }


  /**************************************
   Test VectorFunction gradient wrt U
   **************************************/


  auto tJacobian = tVectorFunction.gradient_u(tU, tV, tA, tControl, tTimeStep);
  auto jac_entries = tJacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
    2.73809523809523804e11, 0.000000000000000000,   0.000000000000000000,
    0.000000000000000000,   2.73809523809523834e11, 0.000000000000000000,
    0.000000000000000000,   0.000000000000000000,   2.73809523809523804e11,
   -4.16666666666666718e10, 2.08333333333333359e10, 0.000000000000000000,
    5.35714285714285660e10,-1.90476190476190460e11, 5.35714285714285660e10,
    0.000000000000000000,   2.08333333333333359e10,-4.16666666666666718e10
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }

  /**************************************
   Test VectorFunction gradient wrt V
   **************************************/

  auto tJacobianV = tVectorFunction.gradient_v(tU, tV, tA, tControl, tTimeStep);
  auto jacV_entries = tJacobianV->entries();
  auto jacV_entriesHost = Kokkos::create_mirror_view( jacV_entries );
  Kokkos::deep_copy(jacV_entriesHost, jacV_entries);

  std::vector<Plato::Scalar> gold_jacV_entries = {
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00
  };

  int jacV_entriesSize = gold_jacV_entries.size();
  for(int i=0; i<jacV_entriesSize; i++){
    if(gold_jacV_entries[i] == 0.0){
      TEST_ASSERT(fabs(jacV_entriesHost[i]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jacV_entriesHost(i), gold_jacV_entries[i], 1.0e-15);
    }
  }

  /**************************************
   Test VectorFunction gradient wrt A
   **************************************/

  auto tJacobianA = tVectorFunction.gradient_a(tU, tV, tA, tControl, tTimeStep);
  auto jacA_entries = tJacobianA->entries();
  auto jacA_entriesHost = Kokkos::create_mirror_view( jacA_entries );
  Kokkos::deep_copy(jacA_entriesHost, jacA_entries);

  // density is zero, so mass matrix should be zeros
  std::vector<Plato::Scalar> gold_jacA_entries = {
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00
  };

  int jacA_entriesSize = gold_jacA_entries.size();
  for(int i=0; i<jacA_entriesSize; i++){
    if(gold_jacA_entries[i] == 0.0){
      TEST_ASSERT(fabs(jacA_entriesHost[i]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jacA_entriesHost(i), gold_jacA_entries[i], 1.0e-15);
    }
  }

  /**************************************
   Test VectorFunction gradient wrt X
   **************************************/

  auto tGradientX = tVectorFunction.gradient_x(tU, tV, tA, tControl, tTimeStep);
  auto tGradientX_entries = tGradientX->entries();
  auto tGradientX_entriesHost = Kokkos::create_mirror_view( tGradientX_entries );
  Kokkos::deep_copy(tGradientX_entriesHost, tGradientX_entries);

  std::vector<Plato::Scalar> gold_tGradientX_entries = {
    -3.36964285714285681024194e6, -1.62857142857142863795161e6,
    -2.47678571428571408614516e6,  1.24642857142857136204839e6,
    -539285.714285714435391128,    442857.142857142782304436,
    -12500.0000000000582076609,    32142.8571428571303840727,
    -101785.714285714231664315,   -7142.85714285714493598789,
    -126785.714285714144352823,   -333928.571428571362048388
  };

  int tGradientX_entriesSize = gold_tGradientX_entries.size();
  for(int i=0; i<tGradientX_entriesSize; i++){
    TEST_FLOATING_EQUALITY(tGradientX_entriesHost(i), gold_tGradientX_entries[i], 1.0e-14);
  }

  /**************************************
   Test VectorFunction gradient wrt Z
   **************************************/

  auto t_dRdz_error = testVectorFunction_Partial_z(tVectorFunction, tU, tV, tA, tControl, tTimeStep);
  TEST_ASSERT(t_dRdz_error < 1.0e-6);

  auto tGradientZ = tVectorFunction.gradient_z(tU, tV, tA, tControl, tTimeStep);
  auto tGradientZ_entries = tGradientZ->entries();
  auto tGradientZ_entriesHost = Kokkos::create_mirror_view( tGradientZ_entries );
  Kokkos::deep_copy(tGradientZ_entriesHost, tGradientZ_entries);

  std::vector<Plato::Scalar> gold_tGradientZ_entries = {
   -352901.785714285681024194, -188839.285714285710128024,
   -212276.785714285681024194, -20982.1428571428550640121,
    135714.285714285710128024, -20982.1428571428587019909,
   -83482.1428571428550640121, -83482.1428571428550640121,
    73214.2857142857101280242, -140401.785714285710128024,
    16294.6428571428550640121,  16294.6428571428550640121
  };

  int tGradientZ_entriesSize = gold_tGradientZ_entries.size();
  for(int i=0; i<tGradientZ_entriesSize; i++){
    TEST_FLOATING_EQUALITY(tGradientZ_entriesHost(i), gold_tGradientZ_entries[i], 1.0e-14);
  }

}


TEUCHOS_UNIT_TEST( TransientMechanicsResidualTests, 3D_WithMass )
{
  // create input for transient mechanics residual
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                       \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>      \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>               \n"
    "  <ParameterList name='Hyperbolic'>                                        \n"
    "    <ParameterList name='Penalty Function'>                                \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>               \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>          \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                  \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='6061-T6 Aluminum'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                   \n"
    "    <ParameterList name='6061-T6 Aluminum'>                                \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                      \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>         \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>  \n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Time Integration'>                                  \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>            \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>            \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>             \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>             \n"
    "  </ParameterList>                                                         \n"
    "</ParameterList>                                                           \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(cSpaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tInputParams);
  Plato::Hyperbolic::VectorFunction<::Plato::Hyperbolic::Mechanics<cSpaceDim>>
    tVectorFunction(tSpatialModel, tDataMap, *tInputParams, tInputParams->get<std::string>("PDE Constraint"));

  int tNumDofs = cSpaceDim*tMesh->nverts();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(1.0, tControl);

  // create mesh based displacement from host data
  //
  std::vector<Plato::Scalar> u_host( cSpaceDim*tMesh->nverts() );
  Plato::Scalar disp = 0.0, dval = 1.0e-7;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  auto tU = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

  Plato::ScalarVector tV("Velocity", tNumDofs);
  Plato::blas1::fill(0.0, tV);

  Plato::ScalarVector tA("Acceleration", tNumDofs);
  Plato::blas1::fill(1000.0, tA);

  auto tTimeStep = tInputParams->sublist("Time Integration").get<Plato::Scalar>("Time Step");


  /**************************************
   Test VectorFunction value
   **************************************/

  auto tResidual = tVectorFunction.value(tU, tV, tA, tControl, tTimeStep);

  // TODO: test Residual
  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  std::vector<Plato::Scalar> tResidual_gold = {
   -1.41152276785714272e6,  -755272.767857142724,  -849022.767857142724,
   -1.60167321428571409e6,  -926673.214285714319,  -449887.500000000000,
     -333900.446428571420,  -333900.446428571420,   292885.267857142782,
   -2.02215267857142864e6,  -468581.249999999767,  -634652.678571428638
  };

  for(int iNode=0; iNode<int(tResidual_gold.size()); iNode++){
    if(tResidual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(tResidual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tResidual_Host[iNode], tResidual_gold[iNode], 1e-13);
    }
  }


  /**************************************
   Test VectorFunction gradient wrt U
   **************************************/

  auto tJacobian = tVectorFunction.gradient_u(tU, tV, tA, tControl, tTimeStep);
  auto jac_entries = tJacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  // mass is non-zero, but this shouldn't affect stiffness:
  std::vector<Plato::Scalar> gold_jac_entries = {
    2.73809523809523804e11, 0.000000000000000000,   0.000000000000000000,
    0.000000000000000000,   2.73809523809523834e11, 0.000000000000000000,
    0.000000000000000000,   0.000000000000000000,   2.73809523809523804e11,
   -4.16666666666666718e10, 2.08333333333333359e10, 0.000000000000000000,
    5.35714285714285660e10,-1.90476190476190460e11, 5.35714285714285660e10,
    0.000000000000000000,   2.08333333333333359e10,-4.16666666666666718e10
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  /**************************************
   Test VectorFunction gradient wrt V
   **************************************/

  auto tJacobianV = tVectorFunction.gradient_v(tU, tV, tA, tControl, tTimeStep);
  auto jacV_entries = tJacobianV->entries();
  auto jacV_entriesHost = Kokkos::create_mirror_view( jacV_entries );
  Kokkos::deep_copy(jacV_entriesHost, jacV_entries);

  std::vector<Plato::Scalar> gold_jacV_entries = {
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 0.00000000000000000e+00
  };

  int jacV_entriesSize = gold_jacV_entries.size();
  for(int i=0; i<jacV_entriesSize; i++){
    if(gold_jacV_entries[i] == 0.0){
      TEST_ASSERT(fabs(jacV_entriesHost[i]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jacV_entriesHost(i), gold_jacV_entries[i], 1.0e-15);
    }
  }


  /**************************************
   Test VectorFunction gradient wrt A
   **************************************/

  auto tJacobianA = tVectorFunction.gradient_a(tU, tV, tA, tControl, tTimeStep);
  auto jacA_entries = tJacobianA->entries();
  auto jacA_entriesHost = Kokkos::create_mirror_view( jacA_entries );
  Kokkos::deep_copy(jacA_entriesHost, jacA_entries);

  std::vector<Plato::Scalar> gold_jacA_entries = {
    0.0210937500000000014, 0.00000000000000000,   0.00000000000000000,
      0.00000000000000000, 0.0210937500000000014, 0.00000000000000000,
      0.00000000000000000, 0.00000000000000000,   0.0210937500000000014
  };

  int jacA_entriesSize = gold_jacA_entries.size();
  for(int i=0; i<jacA_entriesSize; i++){
    if(gold_jacA_entries[i] == 0.0){
      TEST_ASSERT(fabs(jacA_entriesHost[i]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jacA_entriesHost(i), gold_jacA_entries[i], 1.0e-15);
    }
  }


  /**************************************
   Test VectorFunction gradient wrt X
   **************************************/

  auto tGradientX = tVectorFunction.gradient_x(tU, tV, tA, tControl, tTimeStep);
  auto tGradientX_entries = tGradientX->entries();
  auto tGradientX_entriesHost = Kokkos::create_mirror_view( tGradientX_entries );
  Kokkos::deep_copy(tGradientX_entriesHost, tGradientX_entries);

  std::vector<Plato::Scalar> gold_tGradientX_entries = {
    -3.36969910714285681024194e6, -1.62862767857142863795161e6,
    -2.47684196428571408614516e6,  1.24637232142857136204839e6,
    -539341.964285714435391128,    442800.892857142782304436,
    -12556.2500000000582076609,    32086.6071428571303840727,
    -101841.964285714231664315,   -7142.85714285714493598789,
    -126785.714285714144352823,   -333928.571428571362048388
  };

  int tGradientX_entriesSize = gold_tGradientX_entries.size();
  for(int i=0; i<tGradientX_entriesSize; i++){
    TEST_FLOATING_EQUALITY(tGradientX_entriesHost(i), gold_tGradientX_entries[i], 1.0e-14);
  }


  /**************************************
   Test VectorFunction gradient wrt Z
   **************************************/

  auto tGradientZ = tVectorFunction.gradient_z(tU, tV, tA, tControl, tTimeStep);
  auto tGradientZ_entries = tGradientZ->entries();
  auto tGradientZ_entriesHost = Kokkos::create_mirror_view( tGradientZ_entries );
  Kokkos::deep_copy(tGradientZ_entriesHost, tGradientZ_entries);

  std::vector<Plato::Scalar> gold_tGradientZ_entries = {
   -352880.691964285681024194, -188818.191964285710128024,
   -212255.691964285681024194, -20975.1116071428550640121,
    135721.316964285710128024, -20975.1116071428587019909,
   -83475.1116071428550640121, -83475.1116071428550640121,
    73221.3169642857101280242, -140394.754464285710128024,
    16301.6741071428550640121,  16301.6741071428550640121
  };

  int tGradientZ_entriesSize = gold_tGradientZ_entries.size();
  for(int i=0; i<tGradientZ_entriesSize; i++){
    TEST_FLOATING_EQUALITY(tGradientZ_entriesHost(i), gold_tGradientZ_entries[i], 1.0e-14);
  }

}

TEUCHOS_UNIT_TEST( TransientMechanicsResidualTests, NewmarkIntegrator )
{
  // create input for transient mechanics residual
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                   \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>  \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>           \n"
    "  <ParameterList name='Hyperbolic'>                                    \n"
    "    <ParameterList name='Penalty Function'>                            \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>           \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>      \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>              \n"
    "    </ParameterList>                                                   \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Material Models'>                               \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                    \n"
    "      <Parameter name='Mass Density' type='double' value='2.7'/>       \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.36'/>   \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>\n"
    "    </ParameterList>                                                   \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Time Integration'>                              \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>        \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>        \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>         \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>         \n"
    "  </ParameterList>                                                     \n"
    "</ParameterList>                                                       \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);
  int tNumDofs = cSpaceDim*tMesh->nverts();

  auto tIntegratorParams = tInputParams->sublist("Time Integration");
  Plato::NewmarkIntegrator<Plato::Hyperbolic::Mechanics<cSpaceDim>> tIntegrator(tIntegratorParams);

  Plato::Scalar tU_val = 1.0, tV_val = 2.0, tA_val = 3.0;
  Plato::Scalar tU_Prev_val = 3.0, tV_Prev_val = 2.0, tA_Prev_val = 1.0;

  Plato::ScalarVector tU("Displacement", tNumDofs);
  Plato::blas1::fill(tU_val, tU);

  Plato::ScalarVector tV("Velocity", tNumDofs);
  Plato::blas1::fill(tV_val, tV);

  Plato::ScalarVector tA("Acceleration", tNumDofs);
  Plato::blas1::fill(tA_val, tA);

  Plato::ScalarVector tU_Prev("Previous Displacement", tNumDofs);
  Plato::blas1::fill(tU_Prev_val, tU_Prev);

  Plato::ScalarVector tV_Prev("Previous Velocity", tNumDofs);
  Plato::blas1::fill(tV_Prev_val, tV_Prev);

  Plato::ScalarVector tA_Prev("Previous Acceleration", tNumDofs);
  Plato::blas1::fill(tA_Prev_val, tA_Prev);

  auto tTimeStep = tIntegratorParams.get<Plato::Scalar>("Time Step");
  auto tGamma    = tIntegratorParams.get<Plato::Scalar>("Newmark Gamma");
  auto tBeta     = tIntegratorParams.get<Plato::Scalar>("Newmark Beta");

  Plato::Scalar
    tU_pred_val = tU_Prev_val
                + tTimeStep*tV_Prev_val
                + tTimeStep*tTimeStep/2.0 * (1.0-2.0*tBeta)*tA_Prev_val;

  Plato::Scalar
    tV_pred_val = tV_Prev_val
                + (1.0-tGamma)*tTimeStep * tA_Prev_val;


  /**************************************
   Test Newmark integrator v_value
   **************************************/

  Plato::Scalar tTestVal_V = tV_val - tV_pred_val - tGamma/(tBeta*tTimeStep)*(tU_val - tU_pred_val);

  auto tResidualV = tIntegrator.v_value(tU, tU_Prev,
                                        tV, tV_Prev,
                                        tA, tA_Prev, tTimeStep);

  auto tResidualV_host = Kokkos::create_mirror_view( tResidualV );
  Kokkos::deep_copy(tResidualV_host, tResidualV);

  for( int iVal=0; iVal<tNumDofs; iVal++)
  {
      TEST_FLOATING_EQUALITY(tResidualV_host(iVal), tTestVal_V, 1.0e-15);
  }


  /**************************************
   Test Newmark integrator v_grad_u
   **************************************/

  auto tR_vu = tIntegrator.v_grad_u(tTimeStep);
  auto tTestVal_R_vu = -tGamma/(tBeta*tTimeStep);
  TEST_FLOATING_EQUALITY(tR_vu, tTestVal_R_vu, 1.0e-15);


  /**************************************
   Test Newmark integrator v_grad_u_prev
   **************************************/

  auto tR_vu_prev = tIntegrator.v_grad_u_prev(tTimeStep);
  auto tTestVal_R_vu_prev = tGamma/(tBeta*tTimeStep);
  TEST_FLOATING_EQUALITY(tR_vu_prev, tTestVal_R_vu_prev, 1.0e-15);


  /**************************************
   Test Newmark integrator v_grad_v_prev
   **************************************/

  auto tR_vv_prev = tIntegrator.v_grad_v_prev(tTimeStep);
  auto tTestVal_R_vv_prev = tGamma/tBeta - 1.0;
  TEST_FLOATING_EQUALITY(tR_vv_prev, tTestVal_R_vv_prev, 1.0e-15);


  /**************************************
   Test Newmark integrator v_grad_a_prev
   **************************************/

  auto tR_va_prev = tIntegrator.v_grad_a_prev(tTimeStep);
  auto tTestVal_R_va_prev = (tGamma/(2.0*tBeta) - 1.0) * tTimeStep;
  TEST_FLOATING_EQUALITY(tR_va_prev, tTestVal_R_va_prev, 1.0e-15);




  /**************************************
   Test Newmark integrator a_value
   **************************************/

  Plato::Scalar tTestVal_A = tA_val - 1.0/(tBeta*tTimeStep*tTimeStep)*(tU_val - tU_pred_val);

  auto tResidualA = tIntegrator.a_value(tU, tU_Prev,
                                        tV, tV_Prev,
                                        tA, tA_Prev, tTimeStep);

  auto tResidualA_host = Kokkos::create_mirror_view( tResidualA );
  Kokkos::deep_copy(tResidualA_host, tResidualA);

  for( int iVal=0; iVal<tNumDofs; iVal++)
  {
      TEST_FLOATING_EQUALITY(tResidualA_host(iVal), tTestVal_A, 1.0e-15);
  }


  /**************************************
   Test Newmark integrator a_grad_u
   **************************************/

  auto tR_au = tIntegrator.a_grad_u(tTimeStep);
  auto tTestVal_R_au = -1.0/(tBeta*tTimeStep*tTimeStep);
  TEST_FLOATING_EQUALITY(tR_au, tTestVal_R_au, 1.0e-15);


  /**************************************
   Test Newmark integrator a_grad_u_prev
   **************************************/

  auto tR_au_prev = tIntegrator.a_grad_u_prev(tTimeStep);
  auto tTestVal_R_au_prev = 1.0/(tBeta*tTimeStep*tTimeStep);
  TEST_FLOATING_EQUALITY(tR_au_prev, tTestVal_R_au_prev, 1.0e-15);


  /**************************************
   Test Newmark integrator a_grad_v_prev
   **************************************/

  auto tR_av_prev = tIntegrator.a_grad_v_prev(tTimeStep);
  auto tTestVal_R_av_prev = 1.0/(tBeta*tTimeStep);
  TEST_FLOATING_EQUALITY(tR_av_prev, tTestVal_R_av_prev, 1.0e-15);


  /**************************************
   Test Newmark integrator a_grad_a_prev
   **************************************/

  auto tR_aa_prev = tIntegrator.a_grad_a_prev(tTimeStep);
  auto tTestVal_R_aa_prev = 1.0/(2.0*tBeta) - 1.0;
  TEST_FLOATING_EQUALITY(tR_aa_prev, tTestVal_R_aa_prev, 1.0e-15);


}

TEUCHOS_UNIT_TEST( TransientMechanicsResidualTests, 3D_ScalarFunction )
{
  // create input for transient mechanics residual
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                   \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>  \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>           \n"
    "  <ParameterList name='Hyperbolic'>                                    \n"
    "    <ParameterList name='Penalty Function'>                            \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>           \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>      \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>              \n"
    "    </ParameterList>                                                   \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Spatial Model'>                                     \n"
    "    <ParameterList name='Domains'>                                         \n"
    "      <ParameterList name='Design Volume'>                                 \n"
    "        <Parameter name='Element Block' type='string' value='body'/>       \n"
    "        <Parameter name='Material Model' type='string' value='6061-T6 Aluminum'/>\n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Criteria'>                                      \n"
    "    <ParameterList name='Internal Energy'>                             \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>   \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/> \n"
    "      <ParameterList name='Penalty Function'>                          \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>            \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>    \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>         \n"
    "      </ParameterList>                                                 \n"
    "    </ParameterList>                                                   \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Material Models'>                                \n"
    "    <ParameterList name='6061-T6 Aluminum'>                              \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                    \n"
    "        <Parameter name='Mass Density' type='double' value='2.7'/>       \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.36'/>   \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='68.0e10'/>\n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Time Integration'>                              \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>        \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>        \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>         \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-7'/>         \n"
    "  </ParameterList>                                                     \n"
    "</ParameterList>                                                       \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);
  int tNumDofs = cSpaceDim*tMesh->nverts();

  Plato::DataMap tDataMap;
  std::string tMyFunction("Internal Energy");
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(cSpaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tInputParams);
  Plato::Hyperbolic::PhysicsScalarFunction<::Plato::Hyperbolic::Mechanics<cSpaceDim>>
    tScalarFunction(tSpatialModel, tDataMap, *tInputParams, tMyFunction);

  auto tTimeStep = tInputParams->sublist("Time Integration").get<Plato::Scalar>("Time Step");
  auto tNumSteps = tInputParams->sublist("Time Integration").get<int>("Number Time Steps");

  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(1.0, tControl);

  Plato::ScalarMultiVector tU("Displacement", tNumSteps, tNumDofs);
  Plato::ScalarMultiVector tV("Velocity",     tNumSteps, tNumDofs);
  Plato::ScalarMultiVector tA("Acceleration", tNumSteps, tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumDofs), LAMBDA_EXPRESSION(int aDofOrdinal)
  {
    for(int i=0; i<tNumSteps; i++)
    {
      tU(i, aDofOrdinal) = 1.0*i*aDofOrdinal * 1.0e-7;
      tV(i, aDofOrdinal) = 2.0*i*aDofOrdinal * 1.0e-7;
      tA(i, aDofOrdinal) = 3.0*i*aDofOrdinal * 1.0e-7;
    }
  }, "initial data");


  /**************************************
   Test ScalarFunction value
   **************************************/
  Plato::Solutions tSolution;
  tSolution.set("State", tU);
  tSolution.set("StateDot", tV);
  tSolution.set("StateDotDot", tA);
  auto tValue = tScalarFunction.value(tSolution, tControl, tTimeStep);

  TEST_FLOATING_EQUALITY(tValue, 67.9660714285714391280635, 1.0e-15);


  /**************************************
   Test ScalarFunction gradient wrt U
   **************************************/

  int tStepIndex = 1;
  auto tObjGradU = tScalarFunction.gradient_u(tSolution, tControl, tStepIndex, tTimeStep);

  auto tObjGradU_Host = Kokkos::create_mirror_view( tObjGradU );
  Kokkos::deep_copy( tObjGradU_Host, tObjGradU );

  std::vector<Plato::Scalar> tObjGradU_gold = {
   -2.82321428571428544819355e6, -1.51071428571428544819355e6, -1.69821428571428544819355e6,
   -3.20357142857142910361290e6, -1.85357142857142840512097e6, -899999.999999999883584678,
   -667857.142857142724096775,   -667857.142857142724096775,    585714.285714285448193550,
   -4.04464285714285727590322e6, -937500.000000000349245965,   -1.26964285714285681024194e6
  };

  for(int iNode=0; iNode<int(tObjGradU_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(tObjGradU_Host[iNode], tObjGradU_gold[iNode], 1e-15);
  }


  /**************************************
   Test ScalarFunction gradient wrt X
   **************************************/

  auto tObjGradX = tScalarFunction.gradient_x(tSolution, tControl, tTimeStep);

  auto tObjGradX_Host = Kokkos::create_mirror_view( tObjGradX );
  Kokkos::deep_copy( tObjGradX_Host, tObjGradX );

  std::vector<Plato::Scalar> tObjGradX_gold = {
    39.9182142857142849834418, -23.1942857142857121743873, -10.7260714285714300331165,
    37.2053571428571459023260, -27.1317857142857121743873,  3.57428571428571517643036,
    1.28035714285714163906960, -6.81964285714285622930220,  11.3892857142857124586044,
    17.6250000000000035527137,  11.7717857142857162955352, -4.50535714285714217197665
  };

  for(int iNode=0; iNode<int(tObjGradX_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(tObjGradX_Host[iNode], tObjGradX_gold[iNode], 1e-14);
  }


  /**************************************
   Test ScalarFunction gradient wrt Z
   **************************************/

  auto tObjGradZ = tScalarFunction.gradient_z(tSolution, tControl, tTimeStep);

  auto tObjGradZ_Host = Kokkos::create_mirror_view( tObjGradZ );
  Kokkos::deep_copy( tObjGradZ_Host, tObjGradZ );

  std::vector<Plato::Scalar> tObjGradZ_gold = {
    5.13160714285714192328669, 5.46964285714285658457356, 1.36741071428571392409879,
    4.77241071428571395074414, 1.63017857142857103269762, 0.780267857142857179653106,
    1.33767857142857149455040, 0.41678571428571425938614, 2.24089285714285724182560,
    4.35857142857142854097674, 1.51017857142857114816081, 0.332410714285714226079449
  };

  for(int iNode=0; iNode<int(tObjGradZ_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(tObjGradZ_Host[iNode], tObjGradZ_gold[iNode], 1e-15);
  }

}

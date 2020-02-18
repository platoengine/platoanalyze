/*!
  These unit tests are for the transient dynamics formulation
*/

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Strain.hpp"
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
#include "ScalarFunctionBase.hpp"
#include "InterpolateFromNodal.hpp"
#include "PlatoAbstractProblem.hpp"
#include "LinearElasticMaterial.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "hyperbolic/VectorFunctionHyperbolic.hpp"
#include "hyperbolic/Newmark.hpp"
#include "hyperbolic/InertialContent.hpp"
#include "hyperbolic/ElastomechanicsResidual.hpp"
#include "hyperbolic/HyperbolicMechanics.hpp"
#include "hyperbolic/HyperbolicProblem.hpp"


TEUCHOS_UNIT_TEST( TransientDynamicsProblemTests, 3D )
{ 
  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

  // create mesh sets
  //
  Omega_h::Assoc tAssoc;
  tAssoc[Omega_h::NODE_SET] = tMesh->class_sets;
  tAssoc[Omega_h::SIDE_SET] = tMesh->class_sets;
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // create input for transient mechanics problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <Parameter name='PDE Constraint' type='string' value='Hyperbolic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>              \n"
    "  <ParameterList name='Hyperbolic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Model'>                                   \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "      <Parameter name='Mass Density' type='double' value='2700.0'/>       \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.270'/>     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Time Integration'>                                 \n"
    "    <Parameter name='Newmark Gamma' type='double' value='0.5'/>           \n"
    "    <Parameter name='Newmark Beta' type='double' value='0.25'/>           \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>            \n"
    "    <Parameter name='Time Step' type='double' value='1.0e-3'/>            \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                     \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>            \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>     \n"
    "      <Parameter name='Values' type='Array(double)' value='{0, 1e3, 0}'/> \n"
    "      <Parameter name='Sides'  type='string'        value='x+'/>          \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                          \n"
  );

  auto* tHyperbolicProblem = 
    new Plato::HyperbolicProblem<Plato::Hyperbolic::Mechanics<cSpaceDim>>
    (*tMesh, tMeshSets, *tInputParams);

  TEST_ASSERT(tHyperbolicProblem != nullptr);

  int tNumDofs = cSpaceDim*tMesh->nverts();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::fill(1.0, tControl);

  // test solution 
  //
  auto tState = tHyperbolicProblem->solution(tControl);

  // TODO:  test tState


  // test objectiveValue
  //
  auto tObjectiveValue = tHyperbolicProblem->objectiveValue(tControl);

  // TODO: test tObjectiveValue


  // test objectiveValue
  //
  tObjectiveValue = tHyperbolicProblem->objectiveValue(tControl, tState);

  // TODO: test tObjectiveValue


  // test objectiveGradient
  //
  auto tObjectiveGradient = tHyperbolicProblem->objectiveGradient(tControl);

  // TODO: test tObjectiveGradient


  // test objectiveGradient
  //
  tObjectiveGradient = tHyperbolicProblem->objectiveGradient(tControl, tState);

  // TODO: test tObjectiveGradient


  // test objectiveGradientX
  //
  auto tObjectiveGradientX = tHyperbolicProblem->objectiveGradientX(tControl);

  // TODO: test tObjectiveGradientX


  // test objectiveGradientX
  //
  tObjectiveGradientX = tHyperbolicProblem->objectiveGradientX(tControl, tState);

  // TODO: test tObjectiveGradientX



  // test constraintValue
  //
  auto tConstraintValue = tHyperbolicProblem->constraintValue(tControl);

  // TODO: test tConstraintValue


  // test constraintValue
  //
  tConstraintValue = tHyperbolicProblem->constraintValue(tControl, tState);

  // TODO: test tConstraintValue


  // test constraintGradient
  //
  auto tConstraintGradient = tHyperbolicProblem->constraintGradient(tControl);

  // TODO: test tConstraintGradient


  // test constraintGradient
  //
  tConstraintGradient = tHyperbolicProblem->constraintGradient(tControl, tState);

  // TODO: test tConstraintGradient


  // test constraintGradientX
  //
  auto tConstraintGradientX = tHyperbolicProblem->constraintGradientX(tControl);

  // TODO: test tConstraintGradientX


  // test constraintGradientX
  //
  tConstraintGradientX = tHyperbolicProblem->constraintGradientX(tControl, tState);

  // TODO: test tConstraintGradientX

}




TEUCHOS_UNIT_TEST( TransientMechanicsResidualTests, 3D )
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
    "      <Parameter name='Type' type='string' value='SIMP'/>              \n"
    "    </ParameterList>                                                   \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Material Model'>                                \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                    \n"
    "      <Parameter name='Mass Density' type='double' value='1.0'/>       \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>    \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>  \n"
    "    </ParameterList>                                                   \n"
    "  </ParameterList>                                                     \n"
    "  <ParameterList name='Time Integration'>                              \n"
    "    <Parameter name='Number Time Steps' type='int' value='2'/>         \n"
    "    <Parameter name='Time Step' type='double' value='1.0'/>            \n"
    "  </ParameterList>                                                     \n"
    "</ParameterList>                                                       \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  Plato::VectorFunctionHyperbolic<::Plato::Hyperbolic::Mechanics<cSpaceDim>>
    tVectorFunction(*tMesh, tMeshSets, tDataMap, *tInputParams, 
                   tInputParams->get<std::string>("PDE Constraint"));

  int tNumDofs = cSpaceDim*tMesh->nverts();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::fill(1.0, tControl);

  // create mesh based displacement from host data
  //
  std::vector<Plato::Scalar> u_host( cSpaceDim*tMesh->nverts() );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  auto tU = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);

  Plato::ScalarVector tV("Velocity", tNumDofs);
  Plato::fill(0.0, tV);

  Plato::ScalarVector tA("Acceleration", tNumDofs);
  Plato::fill(0.0, tA);

  // compute and test value
  //
  auto tTimeStep = tInputParams->sublist("Time Integration").get<Plato::Scalar>("Time Step");
  auto tResidual = tVectorFunction.value(tU, tV, tA, tControl, tTimeStep);

  // TODO: test Residual
  auto tResidual_Host = Kokkos::create_mirror_view( tResidual );
  Kokkos::deep_copy( tResidual_Host, tResidual );

  std::vector<Plato::Scalar> tResidual_gold = {
    -1903.846153846153,  -894.2307692307692,-1038.461538461538,
    -2062.499999999999, -1024.038461538461,  -692.3076923076922,
     -379.8076923076920, -379.8076923076922,  182.6923076923077,
  };

  for(int iNode=0; iNode<int(tResidual_gold.size()); iNode++){
    if(tResidual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(tResidual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tResidual_Host[iNode], tResidual_gold[iNode], 1e-13);
    }
  }


  auto tJacobian = tVectorFunction.gradient_u(tU, tV, tA, tControl, tTimeStep);
  auto jac_entries = tJacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
    3.52564102564102504e+05, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 3.52564102564102563e+05, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 3.52564102564102563e+05,

   -6.41025641025641016e+04, 3.20512820512820508e+04, 0.00000000000000000e+00,
    4.80769230769230708e+04,-2.24358974358974316e+05, 4.80769230769230708e+04,
    0.00000000000000000e+00, 3.20512820512820508e+04,-6.41025641025641016e+04,
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }
 
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
 
  auto tJacobianA = tVectorFunction.gradient_a(tU, tV, tA, tControl, tTimeStep);
  auto jacA_entries = tJacobianA->entries();
  auto jacA_entriesHost = Kokkos::create_mirror_view( jacA_entries );
  Kokkos::deep_copy(jacA_entriesHost, jacA_entries);

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

}






#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "elliptic/updated_lagrangian/Problem.hpp"


TEUCHOS_UNIT_TEST( EllipticUpdLagProblemTests, 3D )
{
  // create test mesh
  //
  constexpr int cMeshWidth=4;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

  // create mesh sets
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(cSpaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                   \n"
    "  <Parameter name='PDE Constraint' type='string' value='Updated Lagrangian Elliptic'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                           \n"
    "  <ParameterList name='Updated Lagrangian Elliptic'>                                   \n"
    "    <ParameterList name='Penalty Function'>                                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                              \n"
    "      <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
    "      <Parameter name='Minimum Value' type='double' value='1e-5'/>                     \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Spatial Model'>                                                 \n"
    "    <ParameterList name='Domains'>                                                     \n"
    "      <ParameterList name='Design Volume'>                                             \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                   \n"
    "        <Parameter name='Material Model' type='string' value='316 Stainless'/>         \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Criteria'>                                                      \n"
    "    <ParameterList name='Internal Energy'>                                             \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>                   \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/> \n"
    "      <ParameterList name='Penalty Function'>                                          \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                            \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                         \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>                    \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Material Models'>                                               \n"
    "    <ParameterList name='316 Stainless'>                                               \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                  \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.30'/>                 \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.00e10'/>              \n"
    "        <Parameter  name='e11' type='double' value='0.0'/>                             \n"
    "        <Parameter  name='e22' type='double' value='0.0'/>                             \n"
    "        <Parameter  name='e33' type='double' value='-3.0e-6'/>                         \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Sequence'>                                                      \n"
    "    <Parameter name='Sequence Type' type='string' value='Explicit'/>                   \n"
    "    <ParameterList name='Steps'>                                                       \n"
    "      <ParameterList name='Layer 1'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='0.25'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "      <ParameterList name='Layer 2'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='0.50'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "      <ParameterList name='Layer 3'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='0.75'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "      <ParameterList name='Layer 4'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='1.00'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                                \n"
    "    <ParameterList  name='X Fixed Displacement'>                                       \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='0'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Y Fixed Displacement'>                                       \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='1'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Z Fixed Displacement'>                                       \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='2'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "</ParameterList>                                                                       \n"
  );

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  using PhysicsType = Plato::Elliptic::UpdatedLagrangian::Mechanics<cSpaceDim>;
  auto* tProblem = new Plato::Elliptic::UpdatedLagrangian::Problem<PhysicsType> (*tMesh, tMeshSets, *tInputParams, tMachine);

  TEST_ASSERT(tProblem != nullptr);

  int tNumDofs = cSpaceDim*tMesh->nverts();
  Plato::ScalarVector tControl("control", tNumDofs);
  Plato::blas1::fill(1.0, tControl);

  auto tSolution = tProblem->solution(tControl);

  /*****************************************************
   Test Problem::criterionValue(aControl);
   *****************************************************/

  auto tCriterionValue = tProblem->criterionValue(tControl, "Internal Energy");
  Plato::Scalar tCriterionValue_gold = 5.43649521380863686677761e-9;

  TEST_FLOATING_EQUALITY( tCriterionValue, tCriterionValue_gold, 1e-13);


  /*****************************************************
   Test Problem::criterionGradient(aControl);
   *****************************************************/

  auto tCriterionGradient = tProblem->criterionGradient(tControl, "Internal Energy");

  auto tCriterionGradient_Host = Kokkos::create_mirror(tCriterionGradient);
  Kokkos::deep_copy(tCriterionGradient_Host, tCriterionGradient);

  Plato::Scalar tTolerance = 1e-12;
  std::vector<Plato::Scalar> tCriterionGradient_Gold = {-0.927379, -0.46369, -0.927379, -0.46369};
  for(Plato::OrdinalType tIndex = 0; tIndex < tCriterionGradient_Gold.size(); tIndex++)
  {
      TEST_FLOATING_EQUALITY(tCriterionGradient_Host(tIndex), tCriterionGradient_Gold[tIndex], tTolerance);
  }


  /*****************************************************
   Call Problem::criterionGradientX(aControl);
   *****************************************************/

  auto tCriterionGradientX = tProblem->criterionGradientX(tControl, "Internal Energy");

  auto tCriterionGradientX_Host = Kokkos::create_mirror(tCriterionGradientX);
  Kokkos::deep_copy(tCriterionGradientX_Host, tCriterionGradientX);

  std::vector<Plato::Scalar> tCriterionGradientX_Gold = {-0.927379, -0.46369, -0.927379, -0.46369};
  for(Plato::OrdinalType tIndex = 0; tIndex < tCriterionGradientX_Gold.size(); tIndex++)
  {
      TEST_FLOATING_EQUALITY(tCriterionGradientX_Host(tIndex), tCriterionGradientX_Gold[tIndex], tTolerance);
  }



}


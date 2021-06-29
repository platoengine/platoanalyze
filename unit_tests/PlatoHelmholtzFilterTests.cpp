#include "PlatoTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "helmholtz/Helmholtz.hpp"
#include "helmholtz/VectorFunction.hpp"
#include "helmholtz/SimplexHelmholtz.hpp"
#include "helmholtz/Problem.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "BLAS1.hpp"
#include "PlatoMathHelpers.hpp"

#ifdef HAVE_AMGX
#include <alg/AmgXSparseLinearProblem.hpp>
#endif

#include <fenv.h>
#include <memory>

template <typename DataType>
void print_view(const Plato::ScalarVectorT<DataType> & aView)
{
    auto tView_host = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView_host, aView);
    std::cout << '\n';
    for (unsigned int i = 0; i < aView.extent(0); ++i)
    {
        std::cout << tView_host(i) << '\n';
    }
}

// print full matrix entries
void PrintFullMatrix(const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrix)
{
    auto tNumRows = aInMatrix->numRows();
    auto tNumCols = aInMatrix->numCols();

    auto tFullMat = ::PlatoUtestHelpers::toFull(aInMatrix);

    printf("\n Full matrix entries: \n");
    for (auto iRow = 0; iRow < tNumRows; iRow++)
    {
        for (auto iCol = 0; iCol < tNumCols; iCol++)
        {
            printf("%f ",tFullMat[iRow][iCol]);
        }
        printf("\n");
    
    }
}

/******************************************************************************/
/*!
  \brief test parsing of length scale parameter

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(HelmholtzFilterTests, LengthScaleKeywordError)
{
  // create test mesh
  //
  constexpr int meshWidth=20;
  constexpr int spaceDim=1;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;

  // set parameters
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/> \n"
    "  <ParameterList name='Length Scale'>                                    \n"
    "    <Parameter name='LengthScale' type='double' value='0.10'/>              \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                        \n"
  );

  // get mesh sets
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // create PDE
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  TEST_THROW(Plato::Helmholtz::VectorFunction<SimplexPhysics> vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint")), std::runtime_error);
}

/******************************************************************************/
/*!
  \brief test parsing Helmholtz problem

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(HelmholtzFilterTests, HelmholtzProblemError)
{
  // create test mesh
  //
  constexpr int meshWidth=4;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);
  
  // create mesh based density
  //
  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;
  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);
  Plato::ScalarVector testControl("test density", tNumDofs);
  Kokkos::deep_copy(testControl, 1.0);

  // set parameters
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/> \n"
    "  <Parameter name='Physics' type='string' value='Helmholtz Filter'/> \n"
    "  <ParameterList name='Length Scale'>                                    \n"
    "    <Parameter name='Length Scale' type='double' value='0.10'/>              \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                        \n"
  );

  // get mesh sets
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // get machine
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  // construct problem
  auto tProblem = Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<spaceDim>>(*tMesh, tMeshSets, *tParamList, tMachine);

  // perform necessary operations
  auto tSolution = tProblem.solution(control);
  Plato::ScalarVector tFilteredControl = Kokkos::subview(tSolution.get("State"), 0, Kokkos::ALL());
  Kokkos::deep_copy(testControl, tFilteredControl);

  std::string tDummyString = "Helmholtz gradient";
  Plato::ScalarVector tGradient = tProblem.criterionGradient(control,tDummyString);

  

}

/******************************************************************************/
/*!
  \brief homogeneous Helmholtz problem

  Construct a 2D Helmholtz filter problem with uniform unfiltered density 
  and solve. Test passes if filtered density values match unfiltered.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, Helmholtz2DUniformFieldTest )
{
  // create test mesh
  //
  constexpr int meshWidth=8;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // create PDE
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/> \n"
    "  <ParameterList name='Length Scale'>                                    \n"
    "    <Parameter name='Length Scale' type='double' value='0.10'/>              \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Linear Solver'>                              \n"
    "    <Parameter name='Solver Stack' type='string' value='Epetra'/>   \n"
    "    <Parameter name='Display Iterations' type='int' value='1'/>     \n"
    "    <Parameter name='Iterations' type='int' value='50'/>            \n"
    "    <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                        \n"
  );

  // get mesh sets
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // create PDE
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);
  Plato::Helmholtz::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver Stack' type='string' value='Epetra'/>   \n"
    "  <Parameter name='Display Iterations' type='int' value='1'/>     \n"
    "  <Parameter name='Iterations' type='int' value='50'/>            \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(*tMesh, tMachine, tNumDofsPerNode);
  
  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  /* std::cout << "Unfiltered Density" << std::endl; */
  /* print_view(control); */
  /* std::cout << '\n'; */

  /* std::cout << "Solution Field" << std::endl; */
  /* print_view(statesView); */
  /* std::cout << '\n'; */


  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test that filtered density field is still 1
  //
  for(int iDof=0; iDof<tNumDofs; iDof++){
    TEST_FLOATING_EQUALITY(stateView_host(iDof), 1.0, 1.0e-14);
  }

}

/******************************************************************************/
/*!
  \brief 1D Helmholtz problem

  Construct a 1D Helmholtz filter problem and solve. 
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, Helmholtz1DTest )
{
  // create test mesh
  //
  constexpr int meshWidth=140;
  constexpr int spaceDim=1;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);
  control(61) = 0.0;
  control(62) = 0.0;
  control(63) = 0.0;
  control(64) = 0.0;
  
  control(67) = 0.0;
  control(68) = 0.0;
  control(69) = 0.0;
  control(70) = 0.0;

  control(80) = 0.0;
  control(81) = 0.0;
  control(82) = 0.0;
  control(83) = 0.0;
  control(84) = 0.0;
  control(85) = 0.0;
  control(86) = 0.0;
  control(87) = 0.0;
  control(88) = 0.0;
  control(89) = 0.0;

  control(90) = 0.0;
  control(91) = 0.0;
  control(92) = 0.0;
  control(93) = 0.0;
  control(94) = 0.0;
  control(95) = 0.0;
  control(96) = 0.0;
  control(97) = 0.0;
  control(98) = 0.0;
  control(99) = 0.0;

  control(100) = 0.0;
  control(101) = 0.0;
  control(102) = 0.0;
  control(103) = 0.0;
  control(104) = 0.0;
  control(105) = 0.0;
  control(106) = 0.0;
  control(107) = 0.0;
  control(108) = 0.0;
  control(109) = 0.0;

  control(110) = 0.0;
  control(111) = 0.0;
  control(112) = 0.0;
  control(113) = 0.0;
  control(114) = 0.0;
  control(115) = 0.0;
  control(116) = 0.0;
  control(117) = 0.0;
  control(118) = 0.0;
  control(119) = 0.0;

  control(120) = 0.0;
  control(121) = 0.0;
  control(122) = 0.0;
  control(123) = 0.0;
  control(124) = 0.0;
  control(125) = 0.0;
  control(126) = 0.0;
  control(127) = 0.0;
  control(128) = 0.0;
  control(129) = 0.0;

  control(130) = 0.0;
  control(131) = 0.0;
  control(132) = 0.0;
  control(133) = 0.0;
  control(134) = 0.0;
  control(135) = 0.0;
  control(136) = 0.0;
  control(137) = 0.0;
  control(138) = 0.0;
  control(139) = 0.0;

  control(140) = 0.0;

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // create PDE
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/> \n"
    "  <ParameterList name='Length Scale'>                                    \n"
    "    <Parameter name='Length Scale' type='double' value='0.007143'/>              \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                        \n"
  );

  // get mesh sets
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // create PDE
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);
  Plato::Helmholtz::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver Stack' type='string' value='Epetra'/>   \n"
    "  <Parameter name='Display Iterations' type='int' value='1'/>     \n"
    "  <Parameter name='Iterations' type='int' value='50'/>            \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(*tMesh, tMachine, tNumDofsPerNode);
  
  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  /* std::cout << "Unfiltered Density" << std::endl; */
  /* print_view(control); */
  /* std::cout << '\n'; */

  std::cout << "Solution Field" << std::endl;
  print_view(statesView);
  std::cout << '\n';


  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test that filtered density field is still 1
  //
  /* for(int iDof=0; iDof<tNumDofs; iDof++){ */
  /*   TEST_FLOATING_EQUALITY(stateView_host(iDof), 1.0, 1.0e-14); */
  /* } */

}

/******************************************************************************/
/*!
  \brief test Helmholtz gradient

  Construct a 2D Helmholtz gradient problem.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, Helmholtz2DGradientTest )
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=1;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::HelmholtzFilter<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // create PDE
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                      \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Helmholtz Filter'/> \n"
    "  <ParameterList name='Length Scale'>                                    \n"
    "    <Parameter name='Length Scale' type='double' value='0.10'/>              \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                        \n"
  );

  // get mesh sets
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // create PDE
  Plato::DataMap tDataMap;
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);
  Plato::Helmholtz::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute residual
  //
  Plato::ScalarVector gradientState("gradient state", tNumDofs);
  Kokkos::deep_copy(gradientState, 0.0);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);
  
  // compute derivative wrt control (mass matrix)
  //
  auto tPartialPDE_WRT_Control = vectorFunction.gradient_z(state, control);


  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver Stack' type='string' value='Epetra'/>   \n"
    "  <Parameter name='Display Iterations' type='int' value='1'/>     \n"
    "  <Parameter name='Iterations' type='int' value='50'/>            \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(*tMesh, tMachine, tNumDofsPerNode);
  
  // solve linear system
  //
  Plato::blas1::scale(-1.0, control);
  tSolver->solve(*jacobian, state, control);
  
  // compute full gradient
  //
  Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Control, state, gradientState);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, gradientState);

  std::cout << "Unfiltered Density" << std::endl;
  print_view(control);
  std::cout << '\n';

  /* std::cout << "Solution Field" << std::endl; */
  /* print_view(statesView); */
  /* std::cout << '\n'; */

  std::cout << "Mass matrix " << std::endl;
  PrintFullMatrix(tPartialPDE_WRT_Control);
  
  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test that filtered density field is still 1
  //
  /* for(int iDof=0; iDof<tNumDofs; iDof++){ */
  /*   TEST_FLOATING_EQUALITY(stateView_host(iDof), 1.0, 1.0e-14); */
  /* } */

}

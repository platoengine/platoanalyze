#include "PlatoTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "helmholtz/Helmholtz.hpp"
#include "helmholtz/VectorFunction.hpp"
#include "helmholtz/SimplexHelmholtz.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "BLAS1.hpp"

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
  \brief 2D Helmholtz problem

  Construct a linear system and solve it with the old AmgX interface, the new
  AmgX interface, and the Epetra interface.  Test passes if all solutions are
  the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HelmholtzFilterTests, Helmholtz2D )
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
    "    <Parameter name='Length Scale' type='double' value='0.50'/>              \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                      \n"
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

  std::cout << "Residual values" << std::endl;
  print_view(residual);
  std::cout << '\n';

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  std::cout << "Jacobian values" << std::endl;
  PrintFullMatrix(jacobian);
  std::cout << '\n';

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

  std::cout << "Unfiltered Density" << std::endl;
  print_view(control);
  std::cout << '\n';

  std::cout << "Solution Field" << std::endl;
  print_view(statesView);
  std::cout << '\n';


  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test that filtered density field is still 1
  //
  TEST_FLOATING_EQUALITY(stateView_host(10), 1.0, 1.0e-8);

}


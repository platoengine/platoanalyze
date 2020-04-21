#include "PlatoTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "Mechanics.hpp"
#include "EssentialBCs.hpp"
#include "VectorFunction.hpp"
#include "ApplyConstraints.hpp"
#include "SimplexMechanics.hpp"
#include "LinearElasticMaterial.hpp"
#include "alg/PlatoSolverFactory.hpp"

#ifdef HAVE_AMGX
#include <alg/AmgXSparseLinearProblem.hpp>
#endif


#include <fenv.h>
#include <memory>

namespace Plato {
namespace Devel {

template <typename ClassT>
using rcp = std::shared_ptr<ClassT>;


std::vector<std::vector<Plato::Scalar>>
toFull(rcp<Epetra_VbrMatrix> aInMatrix)
{
    int tNumMatrixRows = aInMatrix->NumGlobalRows();

    std::vector<std::vector<Plato::Scalar>>
        tRetMatrix(tNumMatrixRows, std::vector<Plato::Scalar>(tNumMatrixRows, 0.0));

    for(int iMatrixRow=0; iMatrixRow<tNumMatrixRows; iMatrixRow++)
    {
        int tNumEntriesThisRow = 0;
        aInMatrix->NumMyRowEntries(iMatrixRow, tNumEntriesThisRow);
        int tNumEntriesFound = 0;
        std::vector<Plato::Scalar> tVals(tNumEntriesThisRow,0);
        std::vector<int> tInds(tNumEntriesThisRow,0);
        aInMatrix->ExtractMyRowCopy(iMatrixRow, tNumEntriesThisRow, tNumEntriesFound, tVals.data(), tInds.data());
        for(int iEntry=0; iEntry<tNumEntriesFound; iEntry++)
        {
            tRetMatrix[iMatrixRow][tInds[iEntry]] = tVals[iEntry];
        }
    }
    return tRetMatrix;
}
} // end namespace Devel
} // end namespace Plato

/******************************************************************************/
/*!
  \brief Test matrix conversion

  Create an EpetraSystem then convert a 2D elasticity jacobian from a
  Plato::CrsMatrix<int> to an Epetra_VbrMatrix.  Then, convert both to a full
  matrix and compare entries.  Test passes if entries are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, MatrixConversion )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>             \n"
    "  <ParameterList name='Elliptic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList name='Material Model'>                                 \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;

  Plato::VectorFunction<SimplexPhysics>
    vectorFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(*mesh, tMachine, tNumDofsPerNode);

  auto tEpetra_VbrMatrix = tSystem.fromMatrix(*jacobian);

  auto tFullEpetra = Plato::Devel::toFull(tEpetra_VbrMatrix);
  auto tFullPlato  = PlatoUtestHelpers::toFull(jacobian);

  for(int iRow=0; iRow<tFullEpetra.size(); iRow++)
  {
      for(int iCol=0; iCol<tFullEpetra[iRow].size(); iCol++)
      {
          TEST_FLOATING_EQUALITY(tFullEpetra[iRow][iCol], tFullPlato[iRow][iCol], 1.0e-15);
      }
  }
}


/******************************************************************************/
/*!
  \brief 2D Elastic problem

  Construct a linear system and solve it with the old AmgX interface, the new
  AmgX interface, and the Epetra interface.  Test passes if all solutions are
  the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, Elastic2D )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=8;
  constexpr int spaceDim=2;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>             \n"
    "  <ParameterList name='Elliptic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList name='Material Model'>                                 \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                   \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>          \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>   \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0}'/>  \n"
    "      <Parameter name='Sides'  type='string'        value='Load'/>      \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                 \n"
    "    <ParameterList  name='X Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  Omega_h::Read<Omega_h::I8> tMarksLoad = Omega_h::mark_class_closure(mesh.get(), Omega_h::EDGE, Omega_h::EDGE, 5 /* class id */);
  tMeshSets[Omega_h::SIDE_SET]["Load"] = Omega_h::collect_marked(tMarksLoad);

  Omega_h::Read<Omega_h::I8> tMarksFix = Omega_h::mark_class_closure(mesh.get(), Omega_h::EDGE, Omega_h::EDGE, 3 /* class id */);
  tMeshSets[Omega_h::NODE_SET]["Fix"] = Omega_h::collect_marked(tMarksFix);


  Plato::VectorFunction<SimplexPhysics>
    vectorFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // compute and test constraint value
  //
  auto residual = vectorFunction.value(state, control);

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  // parse constraints
  //
  Plato::LocalOrdinalVector mBcDofs;
  Plato::ScalarVector mBcValues;
  Plato::EssentialBCs<SimplexPhysics>
      tEssentialBoundaryConditions(params->sublist("Essential Boundary Conditions",false));
  tEssentialBoundaryConditions.get(tMeshSets, mBcDofs, mBcValues);
  Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(jacobian, residual, mBcDofs, mBcValues);

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);


#ifdef HAVE_AMGX
  // *** use old AmgX solver interface *** //
  {
    using AmgXLinearProblem = Plato::AmgXSparseLinearProblem< Plato::OrdinalType, SimplexPhysics::mNumDofsPerNode>;
    auto tConfigString = Plato::get_config_string();
    auto tSolver = Teuchos::rcp(new AmgXLinearProblem(*jacobian, state, residual, tConfigString));
    tSolver->solve();
    tSolver = Teuchos::null;
  }
  Plato::ScalarVector stateOldAmgX("state", tNumDofs);
  Kokkos::deep_copy(stateOldAmgX, state);



  // *** use new AmgX solver interface *** //
  //
  Kokkos::deep_copy(state, 0.0);
  {
    Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Linear Solver'>                                     \n"
      "  <Parameter name='Solver' type='string' value='AmgX'/>                  \n"
      "  <Parameter name='Configuration File' type='string' value='amgx.json'/> \n"
      "</ParameterList>                                                         \n"
    );

    Plato::SolverFactory tSolverFactory(*tSolverParams);

    auto tSolver = tSolverFactory.create(*mesh, tMachine, tNumDofsPerNode);

    tSolver->solve(*jacobian, state, residual);
  }
  Plato::ScalarVector stateNewAmgX("state", tNumDofs);
  Kokkos::deep_copy(stateNewAmgX, state);
#endif


  // *** use Epetra solver interface *** //
  //
  Kokkos::deep_copy(state, 0.0);
  {
    Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Linear Solver'>                              \n"
      "  <Parameter name='Solver' type='string' value='AztecOO'/>        \n"
      "  <Parameter name='Display Iterations' type='int' value='0'/>     \n"
      "  <Parameter name='Iterations' type='int' value='50'/>            \n"
      "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
      "</ParameterList>                                                  \n"
    );

    Plato::SolverFactory tSolverFactory(*tSolverParams);

    auto tSolver = tSolverFactory.create(*mesh, tMachine, tNumDofsPerNode);

    tSolver->solve(*jacobian, state, residual);
  }
  Plato::ScalarVector stateEpetra("state", tNumDofs);
  Kokkos::deep_copy(stateEpetra, state);




#ifdef HAVE_AMGX
  // compare solutions
  //
  auto stateOldAmgX_host = Kokkos::create_mirror_view(stateOldAmgX);
  Kokkos::deep_copy(stateOldAmgX_host, stateOldAmgX);

  auto stateNewAmgX_host = Kokkos::create_mirror_view(stateNewAmgX);
  Kokkos::deep_copy(stateNewAmgX_host, stateNewAmgX);

  auto stateEpetra_host = Kokkos::create_mirror_view(stateEpetra);
  Kokkos::deep_copy(stateEpetra_host, stateEpetra);


  int tLength = stateOldAmgX_host.size();
  for(int i=0; i<tLength; i++){
      if( stateOldAmgX_host(i) > 1e-18 )
      {
          TEST_FLOATING_EQUALITY(stateOldAmgX_host(i), stateNewAmgX_host(i), 1.0e-15);
          TEST_FLOATING_EQUALITY(stateOldAmgX_host(i), stateEpetra_host(i), 1.0e-12);
      }
  }
#endif




}

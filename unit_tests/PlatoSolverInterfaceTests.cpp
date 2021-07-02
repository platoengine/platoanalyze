#include "PlatoTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "Mechanics.hpp"
#include "EssentialBCs.hpp"
#include "elliptic/VectorFunction.hpp"
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

/******************************************************************************//**
 * \brief get view from device
 *
 * \param[in] aView data on device
 * @returns Mirror on host
**********************************************************************************/
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
TEUCHOS_UNIT_TEST( SolverInterfaceTests, MatrixConversionEpetra )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

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

  // create material model
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
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>               \n"
    "  <ParameterList name='Elliptic'>                                         \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                          \n"
  );

  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::Elliptic::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(*tMesh, tMachine, tNumDofsPerNode);

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
  \brief Test vector conversion

  Create an EpetraSystem then convert a Plato::ScalarVector to an Epetra_Vector.
  Test passes if entries of both vectors are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionToEpetraVector )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(*mesh, tMachine, tNumDofsPerNode);

  Plato::ScalarVector tTestVector("test vector", tNumDofs);

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumDofs), LAMBDA_EXPRESSION(int vectorIndex)
  {
    tTestVector(vectorIndex) = (double) vectorIndex;
  }, "fill vector");

  auto tConvertedVector = tSystem.fromVector(tTestVector);

  auto tTestVectorHostMirror = Kokkos::create_mirror_view(tTestVector);

  Kokkos::deep_copy(tTestVectorHostMirror,tTestVector);

  for(int i = 0; i < tNumDofs; ++i)
  {
    TEST_FLOATING_EQUALITY(tTestVectorHostMirror(i), (*tConvertedVector)[i], 1.0e-15);
  }
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Provide input of incorrect size. Test passes if std::domain_error is thrown.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionToEpetraVector_invalidInput )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(*mesh, tMachine, tNumDofsPerNode);

  Plato::ScalarVector tTestVector("test vector", tNumDofs+1);

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumDofs), LAMBDA_EXPRESSION(int vectorIndex)
  {
    tTestVector(vectorIndex) = (double) vectorIndex;
  }, "fill vector");

  TEST_THROW(tSystem.fromVector(tTestVector),std::domain_error);
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Create an EpetraSystem then convert an Epetra_Vector to a Plato::ScalarVector.
  Test passes if entries of both vectors are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionFromEpetraVector )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(*mesh, tMachine, tNumDofsPerNode);
  
  auto tTestVector = std::make_shared<Epetra_Vector>(*(tSystem.getMap()));

  tTestVector->Random();

  Plato::ScalarVector tConvertedVector("converted vector", tNumDofs);

  tSystem.toVector(tConvertedVector, tTestVector);

  auto tConvertedVectorHostMirror = Kokkos::create_mirror_view(tConvertedVector);

  Kokkos::deep_copy(tConvertedVectorHostMirror,tConvertedVector);

  for(int i = 0; i < tNumDofs; ++i)
  {
    TEST_FLOATING_EQUALITY((*tTestVector)[i], tConvertedVectorHostMirror(i), 1.0e-15);
  }
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Provide input of incorrect size. Test passes if std::domain_error is thrown.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionFromEpetraVector_invalidInput )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(*mesh, tMachine, tNumDofsPerNode);

  auto tBogusMap = std::make_shared<Epetra_BlockMap>(tNumNodes+1, tNumDofsPerNode, 0, *(tMachine.epetraComm));
  auto tTestVector = std::make_shared<Epetra_Vector>(*tBogusMap);

  Plato::ScalarVector tConvertedVector("converted vector", tNumDofs+tNumDofsPerNode);

  TEST_THROW(tSystem.toVector(tConvertedVector,tTestVector),std::domain_error);
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Provide output ScalarVector of incorrect size. Test passes if std::range_error is thrown.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionFromEpetraVector_invalidOutputContainerProvided )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::EpetraSystem tSystem(*mesh, tMachine, tNumDofsPerNode);

  auto tTestVector = std::make_shared<Epetra_Vector>(*(tSystem.getMap()));

  Plato::ScalarVector tConvertedVector("converted vector", tNumDofs+tNumDofsPerNode);

  TEST_THROW(tSystem.toVector(tConvertedVector,tTestVector),std::range_error);
}


#ifdef PLATO_TPETRA

/******************************************************************************/
/*!
  \brief Test matrix conversion

  Create an TpetraSystem then convert a 2D elasticity jacobian from a
  Plato::CrsMatrix<int> to an Tpetra_Matrix.  Then, convert both to a full
  matrix and compare entries.  Test passes if entries are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, MatrixConversionTpetra )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

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

  // create material model
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
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>               \n"
    "  <ParameterList name='Elliptic'>                                         \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                          \n"
  );

  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::Elliptic::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::TpetraSystem tSystem(*tMesh, tMachine, tNumDofsPerNode);

  auto tTpetra_Matrix = tSystem.fromMatrix(*jacobian);

  auto tFullPlato  = PlatoUtestHelpers::toFull(jacobian);

  for(int iRow=0; iRow<tFullPlato.size(); iRow++)
  {
    size_t tNumEntriesInRow = tTpetra_Matrix->getNumEntriesInGlobalRow(iRow);
    Teuchos::Array<Plato::Scalar> tRowValues(tNumEntriesInRow);
    Teuchos::Array<Plato::OrdinalType> tColumnIndices(tNumEntriesInRow);
    tTpetra_Matrix->getGlobalRowCopy(iRow, tColumnIndices(), tRowValues(), tNumEntriesInRow);

    std::vector<Plato::Scalar> tTpetraRowValues(tFullPlato[iRow].size(), 0.0);
    for(size_t i = 0; i < tNumEntriesInRow; ++i)
    {
      tTpetraRowValues[tColumnIndices[i]] = tRowValues[i];
    }

    for(int iCol=0; iCol<tFullPlato[iRow].size(); iCol++)
    {
        TEST_FLOATING_EQUALITY(tTpetraRowValues[iCol], tFullPlato[iRow][iCol], 1.0e-15);
    }
  }
}


/******************************************************************************/
/*!
  \brief Test matrix conversion mismatch

  Create an TpetraSystem and a 2D elasticity jacobian Plato::CrsMatrix<int>
  with different sizes. Try to conver the jacobian to a Tpetra_Matrix.
  Test passes if a std::domain_error is thrown
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, MatrixConversionTpetra_wrongSize )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

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

  // create material model
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
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>               \n"
    "  <ParameterList name='Elliptic'>                                         \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Models'>                                   \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "</ParameterList>                                                          \n"
  );

  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::Elliptic::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.gradient_u(state, control);

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);


  constexpr int tBogusMeshWidth=3;
  auto tBogusMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, tBogusMeshWidth);

  Plato::TpetraSystem tSystem(*tBogusMesh, tMachine, tNumDofsPerNode);

  TEST_THROW(tSystem.fromMatrix(*jacobian),std::domain_error);
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Create a TpetraSystem then convert a Plato::ScalarVector to a Tpetra_MultiVector.
  Test passes if entries of both vectors are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionToTpetraVector )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::TpetraSystem tSystem(*tMesh, tMachine, tNumDofsPerNode);

  Plato::ScalarVector tTestVector("test vector", tNumDofs);

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumDofs), LAMBDA_EXPRESSION(int vectorIndex)
  {
    tTestVector(vectorIndex) = (double) vectorIndex;
  }, "fill vector");

  auto tConvertedVector = tSystem.fromVector(tTestVector);

  auto tTestVectorHostMirror = Kokkos::create_mirror_view(tTestVector);
  Kokkos::deep_copy(tTestVectorHostMirror,tTestVector);

  auto tConvertedVectorDeviceView2D = tConvertedVector->getLocalViewDevice();
  auto tConvertedVectorDeviceView1D = Kokkos::subview(tConvertedVectorDeviceView2D,Kokkos::ALL(), 0);
  auto tConvertedVectorHostMirror = Kokkos::create_mirror_view(tConvertedVectorDeviceView1D);
  Kokkos::deep_copy(tConvertedVectorHostMirror,tConvertedVectorDeviceView1D);

  for(int i = 0; i < tNumDofs; ++i)
  {
    TEST_FLOATING_EQUALITY(tTestVectorHostMirror(i), tConvertedVectorHostMirror(i), 1.0e-15);
  }
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Provide input of incorrect size. Test passes if std::domain_error is thrown.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionToTpetraVector_invalidInput )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::TpetraSystem tSystem(*tMesh, tMachine, tNumDofsPerNode);

  Plato::ScalarVector tTestVector("test vector", tNumDofs+1);

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumDofs), LAMBDA_EXPRESSION(int vectorIndex)
  {
    tTestVector(vectorIndex) = (double) vectorIndex;
  }, "fill vector");

  TEST_THROW(tSystem.fromVector(tTestVector),std::domain_error);
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Create an TpetraSystem then convert a Tpetra_MultiVector to a Plato::ScalarVector.
  Test passes if entries of both vectors are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionFromTpetraVector )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::TpetraSystem tSystem(*tMesh, tMachine, tNumDofsPerNode);

  auto tTestVector = Teuchos::rcp(new Plato::Tpetra_MultiVector(tSystem.getMap(),1));

  tTestVector->randomize();

  Plato::ScalarVector tConvertedVector("converted vector", tNumDofs);

  tSystem.toVector(tConvertedVector,tTestVector);

  auto tConvertedVectorHostMirror = Kokkos::create_mirror_view(tConvertedVector);
  Kokkos::deep_copy(tConvertedVectorHostMirror,tConvertedVector);

  auto tTestVectorDeviceView2D = tTestVector->getLocalViewDevice();
  auto tTestVectorDeviceView1D = Kokkos::subview(tTestVectorDeviceView2D, Kokkos::ALL(), 0);
  auto tTestVectorHostMirror = Kokkos::create_mirror_view(tTestVectorDeviceView1D); 
  Kokkos::deep_copy(tTestVectorHostMirror,tTestVectorDeviceView1D);

  for(int i = 0; i < tNumDofs; ++i)
  {
    TEST_FLOATING_EQUALITY(tTestVectorHostMirror(i), tConvertedVectorHostMirror(i), 1.0e-15);
  }
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Provide input of incorrect size. Test passes if std::domain_error is thrown.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionFromTpetraVector_invalidInput )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::TpetraSystem tSystem(*tMesh, tMachine, tNumDofsPerNode);

  auto tBogusMap = Teuchos::rcp(new Plato::Tpetra_Map(tNumDofs+1, 0, tMachine.teuchosComm));

  auto tTestVector = Teuchos::rcp(new Plato::Tpetra_MultiVector(tBogusMap,1));

  tTestVector->randomize();

  Plato::ScalarVector tConvertedVector("converted vector", tNumDofs+1);

  TEST_THROW(tSystem.toVector(tConvertedVector,tTestVector), std::domain_error);
}

/******************************************************************************/
/*!
  \brief Test vector conversion

  Provide output ScalarVector of incorrect size. Test passes if std::range_error is thrown.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, VectorConversionFromTpetraVector_invalidOutputContainerProvided )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Plato::TpetraSystem tSystem(*tMesh, tMachine, tNumDofsPerNode);

  auto tTestVector = Teuchos::rcp(new Plato::Tpetra_MultiVector(tSystem.getMap(),1));

  tTestVector->randomize();

  Plato::ScalarVector tConvertedVector("converted vector", tNumDofs+1);

  TEST_THROW(tSystem.toVector(tConvertedVector,tTestVector), std::domain_error);
}

#endif


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
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

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

  // create material model
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
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>               \n"
    "  <ParameterList name='Elliptic'>                                         \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
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
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
    "      <Parameter  name='Index'    type='long long'    value='0'/>       \n"
#else
    "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
#endif
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
    "      <Parameter  name='Index'    type='long long'    value='1'/>       \n"
#else
    "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
#endif
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Omega_h::Read<Omega_h::I8> tMarksLoad = Omega_h::mark_class_closure(tMesh.get(), Omega_h::EDGE, Omega_h::EDGE, 5 /* class id */);
  tMeshSets[Omega_h::SIDE_SET]["Load"] = Omega_h::collect_marked(tMarksLoad);

  Omega_h::Read<Omega_h::I8> tMarksFix = Omega_h::mark_class_closure(tMesh.get(), Omega_h::EDGE, Omega_h::EDGE, 3 /* class id */);
  tMeshSets[Omega_h::NODE_SET]["Fix"] = Omega_h::collect_marked(tMarksFix);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::Elliptic::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

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
      tEssentialBoundaryConditions(tParamList->sublist("Essential Boundary Conditions",false), tMeshSets);
  tEssentialBoundaryConditions.get(mBcDofs, mBcValues);
  Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(jacobian, residual, mBcDofs, mBcValues);

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);


#ifdef HAVE_AMGX
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

   auto tSolver = tSolverFactory.create(*tMesh, tMachine, tNumDofsPerNode);

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
      "  <Parameter name='Solver Stack' type='string' value='Epetra'/>   \n"
      "  <Parameter name='Display Iterations' type='int' value='0'/>     \n"
      "  <Parameter name='Iterations' type='int' value='50'/>            \n"
      "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
      "</ParameterList>                                                  \n"
    );

    Plato::SolverFactory tSolverFactory(*tSolverParams);

    auto tSolver = tSolverFactory.create(*tMesh, tMachine, tNumDofsPerNode);

    tSolver->solve(*jacobian, state, residual);
  }
  Plato::ScalarVector stateEpetra("state", tNumDofs);
  Kokkos::deep_copy(stateEpetra, state);

  auto stateEpetra_host = Kokkos::create_mirror_view(stateEpetra);
  Kokkos::deep_copy(stateEpetra_host, stateEpetra);
  
#ifdef PLATO_TPETRA
  // *** use Tpetra solver interface *** //
  //
  Kokkos::deep_copy(state, 0.0);
  {
    Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Linear Solver'>                              \n"
      "  <Parameter name='Solver Stack' type='string' value='Tpetra'/>   \n"
      "  <Parameter name='Solver Package' type='string' value='Belos'/>  \n"
      "  <Parameter name='Display Iterations' type='int' value='0'/>     \n"
      "  <Parameter name='Iterations' type='int' value='50'/>            \n"
      "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
      "</ParameterList>                                                  \n"
    );

    Plato::SolverFactory tSolverFactory(*tSolverParams);

    auto tSolver = tSolverFactory.create(*tMesh, tMachine, tNumDofsPerNode);

    tSolver->solve(*jacobian, state, residual);
  }
  Plato::ScalarVector stateTpetra("state", tNumDofs);
  Kokkos::deep_copy(stateTpetra, state);

  auto stateTpetra_host = Kokkos::create_mirror_view(stateTpetra);
  Kokkos::deep_copy(stateTpetra_host, stateTpetra);

  // *** use Tpetra solver interface with MueLu preconditioner*** //
  //
  Kokkos::deep_copy(state, 0.0);
  {
    Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Linear Solver'>                              \n"
      "  <Parameter name='Solver Stack' type='string' value='Tpetra'/>   \n"
      "  <Parameter name='Solver Package' type='string' value='Belos'/>  \n"
      "  <Parameter name='Solver' type='string' value='GMRES'/>                       \n"
      "  <ParameterList name='Solver Options'>                                        \n"
      "    <Parameter name='Maximum Iterations' type='int' value='500'/>              \n"
      "    <Parameter name='Convergence Tolerance' type='double' value='1e-14'/>      \n"
      "  </ParameterList>                                                             \n"
      "  <Parameter name='Preconditioner Package' type='string' value='MueLu'/>       \n"
      "  <ParameterList name='Preconditioner Options'>                                \n"
      /***MueLu intput parameter list goes here*****************************************/
      "    <Parameter name='verbosity' type='string' value='low'/>                    \n"
      /*********************************************************************************/
      "  </ParameterList>                                                             \n"
      "</ParameterList>                                                               \n"
    );

    Plato::SolverFactory tSolverFactory(*tSolverParams);

    auto tSolver = tSolverFactory.create(*tMesh, tMachine, tNumDofsPerNode);

    tSolver->solve(*jacobian, state, residual);
  }
  Plato::ScalarVector stateTpetraWithMueLuPreconditioner("state", tNumDofs);
  Kokkos::deep_copy(stateTpetraWithMueLuPreconditioner, state);

  auto stateTpetraWithMueLuPreconditioner_host = Kokkos::create_mirror_view(stateTpetraWithMueLuPreconditioner);
  Kokkos::deep_copy(stateTpetraWithMueLuPreconditioner_host, stateTpetraWithMueLuPreconditioner);

  // *** use Tpetra solver interface with Amesos2 Direct Solver*** //
  //
  Kokkos::deep_copy(state, 0.0);
  {
    Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Linear Solver'>                               \n"
      "  <Parameter name='Solver Stack' type='string' value='Tpetra'/>    \n"
      "  <Parameter name='Solver Package' type='string' value='Amesos2'/> \n"
      "  <Parameter name='Solver' type='string' value='Superlu'/>         \n"
      "</ParameterList>                                                   \n"
    );

    Plato::SolverFactory tSolverFactory(*tSolverParams);

    auto tSolver = tSolverFactory.create(*tMesh, tMachine, tNumDofsPerNode);

    tSolver->solve(*jacobian, state, residual);
  }
  Plato::ScalarVector stateTpetraWithAmesos2DirectSolver("state", tNumDofs);
  Kokkos::deep_copy(stateTpetraWithAmesos2DirectSolver, state);

  auto stateTpetraWithAmesos2DirectSolver_host = Kokkos::create_mirror_view(stateTpetraWithAmesos2DirectSolver);
  Kokkos::deep_copy(stateTpetraWithAmesos2DirectSolver_host, stateTpetraWithAmesos2DirectSolver);
#endif

  // compare solutions
  int tLength = stateEpetra_host.size();

#ifdef PLATO_TPETRA
  for(int i=0; i<tLength; i++)
  {
      if( stateEpetra_host(i) > 1e-18 || stateTpetra_host(i) > 1e-18)
      {
          TEST_FLOATING_EQUALITY(stateTpetra_host(i), stateEpetra_host(i), 1.0e-12);
          TEST_FLOATING_EQUALITY(stateTpetra_host(i), stateTpetraWithMueLuPreconditioner_host(i), 1.0e-11);
          TEST_FLOATING_EQUALITY(stateTpetra_host(i), stateTpetraWithAmesos2DirectSolver_host(i), 1.0e-11);
      }
  }
#endif

#ifdef HAVE_AMGX
  auto stateNewAmgX_host = Kokkos::create_mirror_view(stateNewAmgX);
  Kokkos::deep_copy(stateNewAmgX_host, stateNewAmgX);

  for(int i=0; i<tLength; i++){
      if( stateNewAmgX_host(i) > 1e-18 )
      {
          TEST_FLOATING_EQUALITY(stateNewAmgX_host(i), stateEpetra_host(i), 1.0e-12);
#ifdef PLATO_TPETRA
          TEST_FLOATING_EQUALITY(stateNewAmgX_host(i), stateTpetra_host(i), 1.0e-12);
#endif
      }
  }
#endif

}


#ifdef PLATO_TPETRA
/******************************************************************************/
/*!
  \brief Tpetra Linear Solver will accept direct parameterlist input for
  tpetra solver and preconditioner

  Create solver with generic plato inputs and another with specific
  parameterlist inputs for a Tpetra solver and preconditioner, then solve.
  Test passes if both systems give the same solution.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, TpetraSolver_accept_parameterlist_input )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=8;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

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

  // create material model
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
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>       \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>               \n"
    "  <ParameterList name='Elliptic'>                                         \n"
    "    <ParameterList name='Penalty Function'>                               \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>              \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
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
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
    "      <Parameter  name='Index'    type='long long'    value='0'/>       \n"
#else
    "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
#endif
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
    "      <Parameter  name='Index'    type='long long'    value='1'/>       \n"
#else
    "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
#endif
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Omega_h::Read<Omega_h::I8> tMarksLoad = Omega_h::mark_class_closure(tMesh.get(), Omega_h::EDGE, Omega_h::EDGE, 5 /* class id */);
  tMeshSets[Omega_h::SIDE_SET]["Load"] = Omega_h::collect_marked(tMarksLoad);

  Omega_h::Read<Omega_h::I8> tMarksFix = Omega_h::mark_class_closure(tMesh.get(), Omega_h::EDGE, Omega_h::EDGE, 3 /* class id */);
  tMeshSets[Omega_h::NODE_SET]["Fix"] = Omega_h::collect_marked(tMarksFix);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::Elliptic::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));

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
      tEssentialBoundaryConditions(tParamList->sublist("Essential Boundary Conditions",false),tMeshSets);
  tEssentialBoundaryConditions.get(mBcDofs, mBcValues);
  Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(jacobian, residual, mBcDofs, mBcValues);

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  // *** use Tpetra solver interface *** //
  //
  Kokkos::deep_copy(state, 0.0);
  {
    Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Linear Solver'>                                           \n"
      "  <Parameter name='Solver Stack' type='string' value='Tpetra' />               \n"
      "  <Parameter name='Solver Package' type='string' value='Belos'/>               \n"
      "  <Parameter name='Solver' type='string' value='Pseudoblock CG'/>              \n"
      "  <ParameterList name='Solver Options'>                                        \n"
      "    <Parameter name='Maximum Iterations' type='int' value='500'/>              \n"
      "    <Parameter name='Convergence Tolerance' type='double' value='1e-14'/>      \n"
      "  </ParameterList>                                                             \n"
      "  <Parameter name='Preconditioner Package' type='string' value='IFpack2'/>     \n"
      "  <Parameter name='Preconditioner Type' type='string' value='ILUT'/>           \n"
      "  <ParameterList name='Preconditioner Options'>                                \n"
      /***IFpack2 intput parameter list goes here***************************************/
      "    <Parameter name='fact: ilut level-of-fill' type='double' value='2.0'/>     \n"
      "    <Parameter name='fact: drop tolerance' type='double' value='0.0'/>         \n"
      "    <Parameter name='fact: absolute threshold' type='double' value='0.1'/>     \n"
      /*********************************************************************************/
      "  </ParameterList>                                                             \n"
      "</ParameterList>                                                               \n"
    );

    Plato::SolverFactory tSolverFactory(*tSolverParams);

    auto tSolver = tSolverFactory.create(*tMesh, tMachine, tNumDofsPerNode);

    tSolver->solve(*jacobian, state, residual);
  }

  Plato::ScalarVector stateTpetra_CG("state", tNumDofs);
  Kokkos::deep_copy(stateTpetra_CG, state);

  auto stateTpetra_CG_host = Kokkos::create_mirror_view(stateTpetra_CG);
  Kokkos::deep_copy(stateTpetra_CG_host, stateTpetra_CG);


  // *** use Tpetra solver interface *** //
  //
  Kokkos::deep_copy(state, 0.0);
  {
    Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Linear Solver'>                              \n"
      "  <Parameter name='Solver Stack' type='string' value='Tpetra' />  \n"
      "  <Parameter name='Solver Package' type='string' value='Belos'/>  \n"
      "  <Parameter name='Display Iterations' type='int' value='0'/>     \n"
      "  <Parameter name='Iterations' type='int' value='50'/>            \n"
      "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
      "</ParameterList>                                                  \n"
    );

    Plato::SolverFactory tSolverFactory(*tSolverParams);

    auto tSolver = tSolverFactory.create(*tMesh, tMachine, tNumDofsPerNode);

    tSolver->solve(*jacobian, state, residual);
  }

  Plato::ScalarVector stateTpetra_default("state", tNumDofs);
  Kokkos::deep_copy(stateTpetra_default, state);

  auto stateTpetra_default_host = Kokkos::create_mirror_view(stateTpetra_default);
  Kokkos::deep_copy(stateTpetra_default_host, stateTpetra_default);
    
  // compare solutions
  

  int tLength = stateTpetra_CG_host.size();

  for(int i=0; i<tLength; i++)
  {
      if( stateTpetra_CG_host(i) > 1e-18 || stateTpetra_default_host(i) > 1e-18)
      {
          TEST_FLOATING_EQUALITY(stateTpetra_CG_host(i), stateTpetra_default_host(i), 1.0e-12);
      }
  }
}

/******************************************************************************/
/*!
  \brief Test valid input parameterlist specifying solver and preconditioner
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, TpetraSolver_valid_input )
{
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                                           \n"
    "  <Parameter name='Solver Stack' type='string' value='Tpetra' />               \n"
    "  <Parameter name='Solver Package' type='string' value='Belos'/>               \n"
    "  <Parameter name='Solver' type='string' value='Pseudoblock CG'/>              \n"
    "  <ParameterList name='Solver Options'>                                        \n"
    "    <Parameter name='Maximum Iterations' type='int' value='50'/>               \n"
    "    <Parameter name='Convergence Tolerance' type='double' value='1e-14'/>      \n"
    "  </ParameterList>                                                             \n"
    "  <Parameter name='Preconditioner Package' type='string' value='IFpack2'/>     \n"
    "  <Parameter name='Preconditioner Type' type='string' value='ILUT'/>           \n"
    "  <ParameterList name='Preconditioner Options'>                                \n"
    /***IFpack2 intput parameter list goes here***************************************/
    "    <Parameter name='fact: ilut level-of-fill' type='double' value='2.0'/>     \n"
    "    <Parameter name='fact: drop tolerance' type='double' value='0.0'/>         \n"
    "    <Parameter name='fact: absolute threshold' type='double' value='0.1'/>     \n"
    /*********************************************************************************/
    "  </ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
  );

  Plato::SolverFactory tSolverFactory(*tSolverParams);
}

/******************************************************************************/
/*!
  \brief Test invalid input parameterlist
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, TpetraSolver_invalid_solver_package )
{
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;
  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                                           \n"
    "  <Parameter name='Solver Stack' type='string' value='Tpetra' />               \n"
    "  <Parameter name='Solver Package' type='string' value='Muelu'/>               \n"
    "  <ParameterList name='Solver Options'>                                        \n"
    "    <Parameter name='Maximum Iterations' type='int' value='50'/>               \n"
    "    <Parameter name='Convergence Tolerance' type='double' value='1e-14'/>      \n"
    "  </ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
  );

  Plato::SolverFactory tSolverFactory(*tSolverParams);
  TEST_THROW(tSolverFactory.create(*tMesh, tMachine, tNumDofsPerNode),std::invalid_argument);
}
#endif

#include "TpetraLinearSolver.hpp"
#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <Ifpack2_Factory.hpp>
#include <MueLu.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Xpetra_CrsMatrix.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>

namespace Plato {
/******************************************************************************//**
 * @brief get view from device
 *
 * @param[in] aView data on device
 * @returns Mirror on host
**********************************************************************************/
template <typename ViewType>
typename ViewType::HostMirror
get(ViewType aView)
{
    using RetType = typename ViewType::HostMirror;
    RetType tView = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView, aView);
    return tView;
}

/******************************************************************************//**
 * @brief Abstract system interface

   This class contains the node and dof map information and permits persistence
   of this information between solutions.
**********************************************************************************/
TpetraSystem::TpetraSystem(
    Omega_h::Mesh& aMesh,
    Comm::Machine  aMachine,
    int            aDofsPerNode
)
{
    mComm = aMachine.teuchosComm;

    int tNumNodes = aMesh.nverts();
    int tNumDofs = tNumNodes*aDofsPerNode;

    mMap = Teuchos::rcp( new Tpetra_Map(tNumDofs, 0, mComm));

}

// void copyRowDataToTpetraMatrix(const Plato::OrdinalType aGlobalRowIndex,
//                                const Kokkos::View<Plato::OrdinalType, Plato::MemSpace>& aGlobalColumnIndicesView,
//                                const Kokkos::View<Plato::Scalar, Plato::MemSpace>& aGlobalColumnValuesView,
//                                Teuchos::RCP<Tpetra_Matrix>& aRetVal)
// {
//   const Kokkos::View<const Plato::OrdinalType*,
//         Plato::MemSpace,
//         Kokkos::MemoryUnmanaged>
//           tGlobalColumnIndicesViewUnmanaged(aGlobalColumnIndicesView.data(), aGlobalColumnIndicesView.extent(0));
//   const Kokkos::View<const Plato::Scalar*,
//         Plato::MemSpace,
//         Kokkos::MemoryUnmanaged>
//           tGlobalColumnValuesViewUnmanaged(aGlobalColumnValuesView.data(), aGlobalColumnValuesView.extent(0));

//   Tpetra::project2nd<Plato::Scalar,Plato::Scalar> tProject2nd;

//   aRetVal->transformGlobalValues<Tpetra::project2nd<Plato::Scalar,Plato::Scalar>,Plato::MemSpace>
//     (aGlobalRowIndex,tGlobalColumnIndicesViewUnmanaged,tGlobalColumnValuesViewUnmanaged,tProject2nd);
// }

// void copyRow(Plato::OrdinalType iLocalRowIndex,
//                   Plato::OrdinalType iBlockRowIndex,
//                   Plato::OrdinalType iColMapEntryIndex,
//                   const Plato::CrsMatrix<Plato::OrdinalType> aInMatrix,
//                   Teuchos::RCP<Tpetra_Matrix>& aRetVal)
// {

//   auto tNumColsPerBlock = aInMatrix.numColsPerBlock();
//   auto tNumRowsPerBlock = aInMatrix.numRowsPerBlock();
//   auto tBlockSize = tNumRowsPerBlock*tNumColsPerBlock;

//   auto tColMap = aInMatrix.columnIndices();
//   auto tValues = aInMatrix.entries();

//   Kokkos::View<Plato::OrdinalType, Plato::MemSpace> tGlobalColumnIndicesView("columnIndices",tNumColsPerBlock);
//   Kokkos::View<Plato::Scalar, Plato::MemSpace> tGlobalColumnValuesView("values",tNumColsPerBlock);

//   Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumColsPerBlock), LAMBDA_EXPRESSION(int iLocalColIndex)
//   {
//     Plato::OrdinalType tBlockColIndex = tColMap(iColMapEntryIndex);
//     Plato::OrdinalType tGlobalColIndex = tBlockColIndex * tNumColsPerBlock + iLocalColIndex;
//     Plato::OrdinalType tSparseIndex = iColMapEntryIndex * tBlockSize + iLocalRowIndex * tNumColsPerBlock + iLocalColIndex;

//     tGlobalColumnIndicesView(iLocalColIndex) = (int) tGlobalColIndex;
//     tGlobalColumnValuesView(iLocalColIndex) = (double) tValues(tSparseIndex);
//   }, "copy row to Tpetra_Matrix");

//   const Plato::OrdinalType tGlobalRowIndex = iBlockRowIndex * tNumRowsPerBlock + iLocalRowIndex;

//   copyRowDataToTpetraMatrix(tGlobalRowIndex,tGlobalColumnIndicesView,tGlobalColumnValuesView, aRetVal);
// }

// void copyBlock(Plato::OrdinalType iBlockRowIndex,
//                   Plato::OrdinalType iColMapEntryIndex,
//                   const Plato::CrsMatrix<Plato::OrdinalType> aInMatrix,
//                   Teuchos::RCP<Tpetra_Matrix>& aRetVal)
// {
//   auto tNumRowsPerBlock = aInMatrix.numRowsPerBlock();

//   for(Plato::OrdinalType iLocalRowIndex=0; iLocalRowIndex<tNumRowsPerBlock; iLocalRowIndex++)
//     copyRow(iLocalRowIndex, iBlockRowIndex, iColMapEntryIndex, aInMatrix, aRetVal);
// }

// void copyBlocksInBlockRow(Plato::OrdinalType iBlockRowIndex,
//                              Kokkos::View<Plato::OrdinalType*, MemSpace>::HostMirror aRowMap,
//                              const Plato::CrsMatrix<Plato::OrdinalType> aInMatrix,
//                              Teuchos::RCP<Tpetra_Matrix>& aRetVal)
// {
//   auto tFrom = aRowMap(iBlockRowIndex);
//   auto tTo   = aRowMap(iBlockRowIndex+1);
//   for(auto iColMapEntryIndex=tFrom; iColMapEntryIndex<tTo; iColMapEntryIndex++)
//     copyBlock(iBlockRowIndex, iColMapEntryIndex, aInMatrix, aRetVal);
// }

// /******************************************************************************//**
//  * @brief Convert from Plato::CrsMatrix<Plato::OrdinalType> to Tpetra_Matrix
// **********************************************************************************/
// Teuchos::RCP<Tpetra_Matrix>
// TpetraSystem::fromMatrix(const Plato::CrsMatrix<Plato::OrdinalType> aInMatrix) const
// {
//   auto tRowMap = get(aInMatrix.rowMap());

//   checkInputMatrixSize(aInMatrix,tRowMap);

//   auto tRetVal = Teuchos::rcp(new Tpetra_Matrix(mMap, 0));

//   auto tNumBlockRows = tRowMap.extent(0)-1;

//   for(Plato::OrdinalType iBlockRowIndex=0; iBlockRowIndex<tNumBlockRows; iBlockRowIndex++)
//     copyBlocksInBlockRow(iBlockRowIndex,tRowMap,aInMatrix,tRetVal);

//   tRetVal->fillComplete();

//   return tRetVal;
// }

// /******************************************************************************//**
//  * @brief Check if intput Plato::CrsMatrix is consistent with TpetraSystem map 
// **********************************************************************************/
// void TpetraSystem::checkInputMatrixSize(const Plato::CrsMatrix<Plato::OrdinalType> aInMatrix,
//       Kokkos::View<Plato::OrdinalType*, MemSpace>::HostMirror aRowMap) const
// {
//   auto tTemp = Teuchos::rcp(new Tpetra_Matrix(mMap, 0));

//   auto tNumRowsPerBlock = aInMatrix.numRowsPerBlock();
//   auto tNumBlockRows = aRowMap.extent(0)-1;

//   size_t tCrsMatrixGlobalNumRows = tNumBlockRows * tNumRowsPerBlock;
//   size_t tTpetraGlobalNumRows = tTemp->getGlobalNumRows();
//   if(tCrsMatrixGlobalNumRows != tTpetraGlobalNumRows)
//     throw std::domain_error("Input Plato::CrsMatrix size does not match TpetraSystem map.\n");
// }

/******************************************************************************//**
 * @brief Convert from Plato::CrsMatrix<Plato::OrdinalType> to Tpetra_Matrix
**********************************************************************************/
Teuchos::RCP<Tpetra_Matrix>
TpetraSystem::fromMatrix(Plato::CrsMatrix<Plato::OrdinalType> aInMatrix) const
{
  auto tRetVal = Teuchos::rcp(new Tpetra_Matrix(mMap, 0));

  auto tNumRowsPerBlock = aInMatrix.numRowsPerBlock();
  auto tNumColsPerBlock = aInMatrix.numColsPerBlock();
  auto tBlockSize = tNumRowsPerBlock*tNumColsPerBlock;

  auto tRowMap = get(aInMatrix.rowMap());
  auto tColMap = get(aInMatrix.columnIndices());
  auto tValues = get(aInMatrix.entries());

  auto tNumRows = tRowMap.extent(0)-1;
  size_t tCrsMatrixGlobalNumRows = tNumRows * tNumRowsPerBlock;
  size_t tTpetraGlobalNumRows = tRetVal->getGlobalNumRows();
  if(tCrsMatrixGlobalNumRows != tTpetraGlobalNumRows)
    throw std::domain_error("Input Plato::CrsMatrix size does not match TpetraSystem map.\n");

  for(Plato::OrdinalType iRowIndex=0; iRowIndex<tNumRows; iRowIndex++)
  {
      auto tFrom = tRowMap(iRowIndex);
      auto tTo   = tRowMap(iRowIndex+1);
      for(auto iColMapEntryIndex=tFrom; iColMapEntryIndex<tTo; iColMapEntryIndex++)
      {
          auto tBlockColIndex = tColMap(iColMapEntryIndex);
          for(Plato::OrdinalType iLocalRowIndex=0; iLocalRowIndex<tNumRowsPerBlock; iLocalRowIndex++)
          {
              auto tRowIndex = iRowIndex * tNumRowsPerBlock + iLocalRowIndex;
              std::vector<Plato::OrdinalType> tGlobalColumnIndices;
              std::vector<Plato::Scalar> tGlobalColumnValues;
              for(Plato::OrdinalType iLocalColIndex=0; iLocalColIndex<tNumColsPerBlock; iLocalColIndex++)
              {
                  auto tColIndex = tBlockColIndex * tNumColsPerBlock + iLocalColIndex;
                  auto tSparseIndex = iColMapEntryIndex * tBlockSize + iLocalRowIndex * tNumColsPerBlock + iLocalColIndex;
                  tGlobalColumnIndices.push_back(tColIndex);
                  tGlobalColumnValues.push_back(tValues[tSparseIndex]);
              }
              Teuchos::ArrayView<const Plato::OrdinalType> tGlobalColumnIndicesView(tGlobalColumnIndices);
              Teuchos::ArrayView<const Plato::Scalar> tGlobalColumnValuesView(tGlobalColumnValues);
              tRetVal->insertGlobalValues(tRowIndex,tGlobalColumnIndicesView,tGlobalColumnValuesView);
          }
      }
  }

  tRetVal->fillComplete();

  return tRetVal;
}

/******************************************************************************//**
 * @brief Convert from ScalarVector to Tpetra_MultiVector
**********************************************************************************/
Teuchos::RCP<Tpetra_MultiVector>
TpetraSystem::fromVector(const Plato::ScalarVector tInVector) const
{
  auto tOutVector = Teuchos::rcp(new Tpetra_MultiVector(mMap, 1));
  if(tInVector.extent(0) != tOutVector->getLocalLength())
    throw std::domain_error("ScalarVector size does not match TpetraSystem map\n");

  auto tOutVectorDeviceView2D = tOutVector->getLocalViewDevice();
  auto tOutVectorDeviceView1D = Kokkos::subview(tOutVectorDeviceView2D,Kokkos::ALL(), 0);

  Kokkos::deep_copy(tOutVectorDeviceView1D,tInVector);

  return tOutVector;
}

/******************************************************************************//**
 * @brief Convert from Tpetra_MultiVector to ScalarVector
**********************************************************************************/
void 
TpetraSystem::toVector(Plato::ScalarVector& tOutVector, const Teuchos::RCP<Tpetra_MultiVector> tInVector) const
{
    auto tLength = tInVector->getLocalLength();
    auto tTemp = Teuchos::rcp(new Tpetra_MultiVector(mMap, 1));
    if(tLength != tTemp->getLocalLength())
      throw std::domain_error("Tpetra_MultiVector map does not match TpetraSystem map.");

    if(tOutVector.extent(0) != tTemp->getLocalLength())
      throw std::range_error("ScalarVector does not match TpetraSystem map.");

    auto tInVectorDeviceView2D = tInVector->getLocalViewDevice();
    auto tInVectorDeviceView1D = Kokkos::subview(tInVectorDeviceView2D,Kokkos::ALL(), 0);

    Kokkos::deep_copy(tOutVector,tInVectorDeviceView1D);
}

void getPrecondTypeAndParameters (std::string& precondType, Teuchos::ParameterList& pl)
{
  precondType = "ILUT";

  const double fillLevel = 2.0;
  const double dropTol = 0.0;
  const double absThreshold = 0.1;

  pl.set ("fact: ilut level-of-fill", fillLevel);
  pl.set ("fact: drop tolerance", dropTol);
  pl.set ("fact: absolute threshold", absThreshold);
}

template<class TpetraMatrixType>
Teuchos::RCP<Tpetra::Operator<typename TpetraMatrixType::scalar_type,
                              typename TpetraMatrixType::local_ordinal_type,
                              typename TpetraMatrixType::global_ordinal_type,
                              typename TpetraMatrixType::node_type> >
createIFpack2Preconditioner (const Teuchos::RCP<const TpetraMatrixType>& A,
                      const std::string& precondType,
                      const Teuchos::ParameterList& plist)
{
  typedef typename TpetraMatrixType::scalar_type scalar_type;
  typedef typename TpetraMatrixType::local_ordinal_type local_ordinal_type;
  typedef typename TpetraMatrixType::global_ordinal_type global_ordinal_type;
  typedef typename TpetraMatrixType::node_type node_type;

  typedef Ifpack2::Preconditioner<scalar_type, local_ordinal_type, 
                                  global_ordinal_type, node_type> prec_type;

  Teuchos::RCP<prec_type> prec;
  Ifpack2::Factory factory;
  prec = factory.create (precondType, A);
  prec->setParameters (plist);

  prec->initialize();
  prec->compute();

  return prec;
}


/******************************************************************************//**
 * @brief TpetraLinearSolver constructor

 This constructor takes an Omega_h::Mesh and creates a new TpetraSystem.
**********************************************************************************/
TpetraLinearSolver::TpetraLinearSolver(
    const Teuchos::ParameterList& aSolverParams,
    Omega_h::Mesh&          aMesh,
    Comm::Machine           aMachine,
    int                     aDofsPerNode
) : mSystem(Teuchos::rcp( new TpetraSystem(aMesh, aMachine, aDofsPerNode)))
{
  if(aSolverParams.isType<std::string>("Solver Package"))
    mSolverPackage = aSolverParams.get<std::string>("Solver Package");
  else
    mSolverPackage = "Belos";

  if(aSolverParams.isType<std::string>("Solver"))
    mSolver = aSolverParams.get<std::string>("Solver");
  else if (mSolverPackage == "Belos")
    mSolver = "GMRES";
  else
    throw std::invalid_argument("Solver not specified in input parameter list.\n");

  if(aSolverParams.isType<Teuchos::ParameterList>("Solver Options"))
    mSolverOptions = aSolverParams.get<Teuchos::ParameterList>("Solver Options");
  else if(mSolverPackage == "Belos")
  {
    int tMaxIterations;
    if(aSolverParams.isType<int>("Iterations"))
      tMaxIterations = aSolverParams.get<int>("Iterations");
    else
      tMaxIterations = 300;

    double tTolerance;
    if(aSolverParams.isType<int>("Tolerance"))
      tTolerance = aSolverParams.get<double>("Tolerance");
    else
      tTolerance = 1e-14;

    mSolverOptions.set ("Maximum Iterations", tMaxIterations);
    mSolverOptions.set ("Convergence Tolerance", tTolerance);
  }

  if(aSolverParams.isType<std::string>("Preconditioner Package"))
    mPreconditionerPackage = aSolverParams.get<std::string>("Preconditioner Package");
  else
    mPreconditionerPackage = "IFpack2";

  if(aSolverParams.isType<std::string>("Preconditioner Type"))
    mPreconditionerType = aSolverParams.get<std::string>("Preconditioner Type");
  else if(mPreconditionerPackage == "IFpack2")
    mPreconditionerType = "ILUT";

  if(aSolverParams.isType<Teuchos::ParameterList>("Preconditioner Options"))
    mPreconditionerOptions = aSolverParams.get<Teuchos::ParameterList>("Preconditioner Options");
}

template<class MV, class OP>
void
TpetraLinearSolver::belosSolve (Teuchos::RCP<const OP> A, Teuchos::RCP<MV> X, Teuchos::RCP<const MV> B, Teuchos::RCP<const OP> M) 
{
  using scalar_type = typename MV::scalar_type;
  Teuchos::RCP<Teuchos::ParameterList> tSolverOptions = Teuchos::rcp(new Teuchos::ParameterList(mSolverOptions));
  Belos::SolverFactory<scalar_type, MV, OP> factory;
  Teuchos::RCP<Belos::SolverManager<scalar_type, MV, OP> > solver = 
    factory.create (mSolver, tSolverOptions);

  typedef Belos::LinearProblem<scalar_type, MV, OP> problem_type;
  Teuchos::RCP<problem_type> problem = 
    Teuchos::rcp (new problem_type(A, X, B));

  problem->setRightPrec(M);
  
  problem->setProblem();
  solver->setProblem (problem);

  Belos::ReturnType result = solver->solve();

  // Ask the solver how many iterations the last solve() took.
  const int numIters = solver->getNumIters();

  const double tTolerance = solver->achievedTol();
  if (result == Belos::Converged) {
    //std::cout << "The Belos solve took " << numIters << " iteration(s) to reach "
    //  "a relative residual tolerance of " << tTolerance << "." << std::endl;
  } else {
    std::cout << "The Belos solve took " << numIters << " iteration(s), but did not converge. Achieved tolerance = "
              << tTolerance << "." << std::endl;
  }
}

/******************************************************************************//**
 * @brief Solve the linear system
**********************************************************************************/
void
TpetraLinearSolver::solve(
    Plato::CrsMatrix<Plato::OrdinalType> aA,
    Plato::ScalarVector   aX,
    Plato::ScalarVector   aB
)
{
  Teuchos::RCP<Tpetra_Matrix> A = mSystem->fromMatrix(aA);
  Teuchos::RCP<Tpetra_MultiVector> X = mSystem->fromVector(aX);
  Teuchos::RCP<Tpetra_MultiVector> B = mSystem->fromVector(aB);

  Teuchos::RCP<Tpetra_Operator> M;

  if(mPreconditionerPackage == "IFpack2")
    M = createIFpack2Preconditioner<Tpetra_Matrix> (A, mPreconditionerType, mPreconditionerOptions);
  else if(mPreconditionerPackage == "MueLu")
    M = MueLu::CreateTpetraPreconditioner(static_cast<Teuchos::RCP<Tpetra_Operator>>(A), mPreconditionerOptions);
  else
  {
    std::string tInvalid_solver = "Preconditioner Package " + mPreconditionerPackage + " is not currently a valid option\n";
    throw std::invalid_argument(tInvalid_solver);
  }

  if(mSolverPackage == "Belos")
    belosSolve<Tpetra_MultiVector, Tpetra_Operator> (A, X, B, M);
  else
  {
    std::string tInvalid_solver = "Solver Package " + mSolverPackage + " is not currently a valid option\n";
    throw std::invalid_argument(tInvalid_solver);
  }
  mSystem->toVector(aX,X);
}

} // end namespace Plato

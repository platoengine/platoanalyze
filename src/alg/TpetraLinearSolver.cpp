#include "TpetraLinearSolver.hpp"
#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <Ifpack2_Factory.hpp>

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
    int            aNumNodes,
    Comm::Machine  aMachine,
    int            aDofsPerNode
) {
    mComm = aMachine.teuchosComm;

    int tNumDofs = aNumNodes*aDofsPerNode;

    mMap = Teuchos::rcp( new Tpetra_Map(tNumDofs, 0, mComm));

}

/******************************************************************************//**
 * @brief Convert from Plato::CrsMatrix<int> to Tpetra_Matrix
**********************************************************************************/
Teuchos::RCP<Tpetra_Matrix>
TpetraSystem::fromMatrix(Plato::CrsMatrix<int> aInMatrix) const
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
TpetraSystem::fromVector(Plato::ScalarVector tInVector) const
{
  auto tRetVal = Teuchos::rcp(new Tpetra_MultiVector(mMap, 1));
  if(tInVector.extent(0) != tRetVal->getLocalLength())
    throw std::domain_error("ScalarVector size does not match TpetraSystem map\n");

  auto tRetValHostView2D = tRetVal->getLocalViewHost();
  auto tRetValHostView1D = Kokkos::subview(tRetValHostView2D,Kokkos::ALL(), 0);

  // copy to host from device
  Kokkos::deep_copy(tRetValHostView1D, tInVector);

  return tRetVal;
}

/******************************************************************************//**
 * @brief Convert from Tpetra_MultiVector to ScalarVector
**********************************************************************************/
void 
TpetraSystem::toVector(Plato::ScalarVector tOutVector, Teuchos::RCP<Tpetra_MultiVector> tInVector) const
{
    auto tLength = tInVector->getLocalLength();
    auto tTemp = Teuchos::rcp(new Tpetra_MultiVector(mMap, 1));
    if(tLength != tTemp->getLocalLength())
      throw std::domain_error("Tpetra_MultiVector map does not match TpetraSystem map.");

    if(tOutVector.extent(0) != tTemp->getLocalLength())
      throw std::range_error("ScalarVector does not match TpetraSystem map.");

    auto tInVectorHostView2D = tInVector->getLocalViewHost();
    auto tInVectorHostView1D = Kokkos::subview(tInVectorHostView2D,Kokkos::ALL(), 0);

    Kokkos::deep_copy(tOutVector, tInVectorHostView1D);
}

/******************************************************************************//**
 * @brief TpetraLinearSolver constructor

 This constructor takes an Omega_h::Mesh and creates a new System.
**********************************************************************************/
TpetraLinearSolver::TpetraLinearSolver(
    const Teuchos::ParameterList& aSolverParams,
    int                     aNumNodes,
    Comm::Machine           aMachine,
    int                     aDofsPerNode
) :
    mSolverParams(aSolverParams),
    mSystem(Teuchos::rcp( new TpetraSystem(aNumNodes, aMachine, aDofsPerNode)))
{
    if(mSolverParams.isType<int>("Iterations"))
    {
        mIterations = mSolverParams.get<int>("Iterations");
    }
    else
    {
        mIterations = 100;
    }

    if(mSolverParams.isType<double>("Tolerance"))
    {
        mTolerance = mSolverParams.get<double>("Tolerance");
    }
    else
    {
        mTolerance = 1e-6;
    }
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
createPreconditioner (const Teuchos::RCP<const TpetraMatrixType>& A,
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

template<class MV, class OP>
void
TpetraLinearSolver::belosSolve (std::ostream& out, Teuchos::RCP<const OP> A, Teuchos::RCP<MV> X, Teuchos::RCP<const MV> B, Teuchos::RCP<const OP> M) 
{
  // Make an empty new parameter list.
  Teuchos::RCP<Teuchos::ParameterList> solverParams = Teuchos::parameterList();
  int tMaxIterations = mSolverParams.get<int>("Iterations");
  double tTolerance = mSolverParams.get<double>("Tolerance");

  solverParams->set ("Num Blocks", 40);
  solverParams->set ("Maximum Iterations", tMaxIterations);
  solverParams->set ("Convergence Tolerance", tTolerance);

  Belos::SolverFactory<Plato::Scalar, MV, OP> factory;
  Teuchos::RCP<Belos::SolverManager<Plato::Scalar, MV, OP> > solver = 
    factory.create ("GMRES", solverParams);

  typedef Belos::LinearProblem<Plato::Scalar, MV, OP> problem_type;
  Teuchos::RCP<problem_type> problem = 
    Teuchos::rcp (new problem_type (A, X, B));

  // problem->setRightPrec (M);
  
  problem->setProblem ();
  solver->setProblem (problem);

  Belos::ReturnType result = solver->solve();

  // Ask the solver how many iterations the last solve() took.
  const int numIters = solver->getNumIters();

  if (result == Belos::Converged) {
    std::cout << "The Belos solve took " << numIters << " iteration(s) to reach "
      "a relative residual tolerance of " << tTolerance << "." << std::endl;
  } else {
    std::cout << "The Belos solve took " << numIters << " iteration(s), but did not reach "
      "a relative residual tolerance of " << tTolerance << "." << std::endl;
  }
}

/******************************************************************************//**
 * @brief Solve the linear system
**********************************************************************************/
void
TpetraLinearSolver::solve(
    Plato::CrsMatrix<int> aA,
    Plato::ScalarVector   aX,
    Plato::ScalarVector   aB
)
{
  Teuchos::RCP<const Tpetra_Matrix> A = mSystem->fromMatrix(aA);
  Teuchos::RCP<Tpetra_MultiVector> X = mSystem->fromVector(aX);
  Teuchos::RCP<Tpetra_MultiVector> B = mSystem->fromVector(aB);

  std::string tSolverType = mSolverParams.get<std::string>("Solver");
  if(tSolverType == "Belos")
  {
    std::string precondType;
    Teuchos::ParameterList plist;
    getPrecondTypeAndParameters (precondType, plist);

    Teuchos::RCP<Tpetra_Operator> M = createPreconditioner<Tpetra_Matrix> (A, precondType, plist);

    belosSolve<Tpetra_MultiVector, Tpetra_Operator> (std::cout, A, X, B, M);
  }
  else
  {
    std::string tInvalid_solver = "Solver type " + tSolverType + " is not a valid option\n";
    throw std::invalid_argument(tInvalid_solver);
  }
  mSystem->toVector(aX,X);
}

} // end namespace Plato

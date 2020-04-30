#include "TpetraLinearSolver.hpp"
#include <BelosTpetraAdapter.hpp>
#include <BelosSolverFactory.hpp>
#include <Amesos2.hpp>
// #include <supermatrix.h>

namespace Plato {
  using Tpetra_Map = Tpetra::Map<Plato::OrdinalType, Plato::OrdinalType>;
  using Tpetra_Vector = Tpetra::Vector<Plato::Scalar, Plato::OrdinalType, Plato::OrdinalType>;
  using Tpetra_Matrix = Tpetra::CrsMatrix<Plato::Scalar, Plato::OrdinalType, Plato::OrdinalType>;

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
) {
    mComm = aMachine.teuchosComm;

    int tNumNodes = aMesh.nverts();
    int tNumDofs = tNumNodes*aDofsPerNode;

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
 * @brief Convert from ScalarVector to Tpetra_Vector
**********************************************************************************/
Teuchos::RCP<Tpetra_Vector>
TpetraSystem::fromVector(Plato::ScalarVector tInVector) const
{
  auto tRetVal = Teuchos::rcp(new Tpetra_Vector(mMap));
  if(tInVector.extent(0) != tRetVal->getLocalLength())
    throw std::domain_error("ScalarVector size does not match TpetraSystem map\n");

  auto tRetValHostView2D = tRetVal->getLocalViewHost();
  auto tRetValHostView1D = Kokkos::subview(tRetValHostView2D,Kokkos::ALL(), 0);

  // copy to host from device
  Kokkos::deep_copy(tRetValHostView1D, tInVector);

  return tRetVal;
}

/******************************************************************************//**
 * @brief Convert from Tpetra_Vector to ScalarVector
**********************************************************************************/
void 
TpetraSystem::toVector(Plato::ScalarVector tOutVector, Teuchos::RCP<Tpetra_Vector> tInVector) const
{
    auto tLength = tInVector->getLocalLength();
    auto tTemp = Teuchos::rcp(new Tpetra_Vector(mMap));
    if(tLength != tTemp->getLocalLength())
      throw std::domain_error("Tpetra_Vector map does not match TpetraSystem map.");

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
    Omega_h::Mesh&          aMesh,
    Comm::Machine           aMachine,
    int                     aDofsPerNode
) :
    mSolverParams(aSolverParams),
    mSystem(Teuchos::rcp( new TpetraSystem(aMesh, aMachine, aDofsPerNode)))
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

template<class MV, class OP>
void
belosSolve (std::ostream& out, MV& X, const MV& B, const OP& A) 
{
  using Teuchos::ParameterList;
  using Teuchos::parameterList;
  using Teuchos::RCP; 
  using Teuchos::rcp;
  using Teuchos::rcpFromRef; // Make a "weak" RCP from a reference.
  typedef typename MV::scalar_type scalar_type;

  // Make an empty new parameter list.
  RCP<ParameterList> solverParams = parameterList();

  // Set some GMRES parameters.
  //
  // "Num Blocks" = Maximum number of Krylov vectors to store.  This
  // is also the restart length.  "Block" here refers to the ability
  // of this particular solver (and many other Belos solvers) to solve
  // multiple linear systems at a time, even though we may only be
  // solving one linear system in this example.
  //
  // "Maximum Iterations": Maximum total number of iterations,
  // including restarts.
  //
  // "Convergence Tolerance": By default, this is the relative
  // residual 2-norm, although you can change the meaning of the
  // convergence tolerance using other parameters.
  solverParams->set ("Num Blocks", 40);
  solverParams->set ("Maximum Iterations", 400);
  solverParams->set ("Convergence Tolerance", 1.0e-8);

  // Create the GMRES solver using a "factory" and 
  // the list of solver parameters created above.
  Belos::SolverFactory<scalar_type, MV, OP> factory;
  RCP<Belos::SolverManager<scalar_type, MV, OP> > solver = 
    factory.create ("GMRES", solverParams);

  // Create a LinearProblem struct with the problem to solve.
  // A, X, B, and M are passed by (smart) pointer, not copied.
  typedef Belos::LinearProblem<scalar_type, MV, OP> problem_type;
  RCP<problem_type> problem = 
    rcp (new problem_type (rcpFromRef (A), rcpFromRef (X), rcpFromRef (B)));
  // You don't have to call this if you don't have a preconditioner.
  // If M is null, then Belos won't use a (right) preconditioner.
  // problem->setRightPrec (M);
  // Tell the LinearProblem to make itself ready to solve.
  problem->setProblem ();

  // Tell the solver what problem you want to solve.
  solver->setProblem (problem);

  // Attempt to solve the linear system.  result == Belos::Converged 
  // means that it was solved to the desired tolerance.  This call 
  // overwrites X with the computed approximate solution.
  Belos::ReturnType result = solver->solve();

  // Ask the solver how many iterations the last solve() took.
  const int numIters = solver->getNumIters();

  if (result == Belos::Converged) {
    std::cout << "The Belos solve took " << numIters << " iteration(s) to reach "
      "a relative residual tolerance of " << 1.0e-8 << "." << std::endl;
  } else {
    std::cout << "The Belos solve took " << numIters << " iteration(s), but did not reach "
      "a relative residual tolerance of " << 1.0e-8 << "." << std::endl;
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
  typedef Tpetra::CrsMatrix<Plato::Scalar, Plato::OrdinalType, Plato::OrdinalType> matrix_type;
  typedef Tpetra::Operator<Plato::Scalar, Plato::OrdinalType, Plato::OrdinalType> op_type;
  typedef Tpetra::MultiVector<Plato::Scalar, Plato::OrdinalType, Plato::OrdinalType> vec_type;
  typedef Tpetra::Map<Plato::Scalar, Plato::OrdinalType, Plato::OrdinalType> map_type;


  Teuchos::RCP<const matrix_type> A = mSystem->fromMatrix(aA);

  Teuchos::RCP<vec_type> X = mSystem->fromVector(aX); // Set to zeros by default.
  Teuchos::RCP<vec_type> B = mSystem->fromVector(aB);

  // Solve the linear system using Belos.
  belosSolve<vec_type, op_type> (std::cout, *X, *B, *A);
  // mSystem->toVector(aX,X);
}

// /******************************************************************************//**
//  * @brief Setup the AztecOO solver
// **********************************************************************************/
// void
// EpetraLinearSolver::setupSolver(AztecOO& aSolver)
// {
//     int tDisplayIterations = 0;
//     if(mSolverParams.isType<int>("Display Iterations"))
//     {
//         tDisplayIterations = mSolverParams.get<int>("Display Iterations");
//     }

//     aSolver.SetAztecOption(AZ_output, tDisplayIterations);

//     // defaults (TODO: add options)
//     aSolver.SetAztecOption(AZ_precond, AZ_ilu);
//     aSolver.SetAztecOption(AZ_subdomain_solve, AZ_ilu);
//     aSolver.SetAztecOption(AZ_precond, AZ_dom_decomp);
//     aSolver.SetAztecOption(AZ_scaling, AZ_row_sum);
//     aSolver.SetAztecOption(AZ_solver, AZ_gmres);
// }

} // end namespace Plato

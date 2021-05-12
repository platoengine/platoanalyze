#include "EpetraLinearSolver.hpp"
#include "PlatoUtilities.hpp"

namespace Plato {

/******************************************************************************//**
 * \brief Abstract system interface

   This class contains the node and dof map information and permits persistence
   of this information between solutions.
**********************************************************************************/
EpetraSystem::EpetraSystem(
    Omega_h::Mesh& aMesh,
    Comm::Machine  aMachine,
    int            aDofsPerNode
):  mMatrixConversionTimer(Teuchos::TimeMonitor::getNewTimer("Analyze: Matrix Conversion")),
    mVectorConversionTimer(Teuchos::TimeMonitor::getNewTimer("Analyze: Vector Conversion"))
{
    mComm = aMachine.epetraComm;

    int tNumNodes = aMesh.nverts();
    mBlockRowMap = std::make_shared<Epetra_BlockMap>(tNumNodes, aDofsPerNode, 0, *mComm);

}

/******************************************************************************//**
 * \brief Convert from Plato::CrsMatrix<Plato::OrdinalType> to Epetra_VbrMatrix
**********************************************************************************/
rcp<Epetra_VbrMatrix>
EpetraSystem::fromMatrix(Plato::CrsMatrix<Plato::OrdinalType> tInMatrix) const
{ 
    Teuchos::TimeMonitor LocalTimer(*mMatrixConversionTimer);

    auto tRowMap_host = Kokkos::create_mirror_view(tInMatrix.rowMap());
    auto tNumRowsPerBlock = tInMatrix.numRowsPerBlock();
    auto tNumBlocks = tRowMap_host.extent(0)-1;

    int tNumRows = mBlockRowMap->NumMyElements();
    std::vector<int> tNumEntries(tNumRows, 0);
    Kokkos::deep_copy(tRowMap_host, tInMatrix.rowMap());
    int tMaxNumEntries = 0;
    for(int iRow=0; iRow<tNumRows; iRow++)
    {
        tNumEntries[iRow] = tRowMap_host[iRow+1] - tRowMap_host[iRow];
        if(tNumEntries[iRow] > tMaxNumEntries) tMaxNumEntries = tNumEntries[iRow];
    }
    auto tNumColsPerBlock = tInMatrix.numColsPerBlock();

    auto tRetVal = std::make_shared<Epetra_VbrMatrix>(Copy, *mBlockRowMap, tNumEntries.data());

    auto tColMap_host = Kokkos::create_mirror_view(tInMatrix.columnIndices());
    Kokkos::deep_copy(tColMap_host, tInMatrix.columnIndices());

    auto tEntries_host = Kokkos::create_mirror_view(tInMatrix.entries());
    Kokkos::deep_copy(tEntries_host, tInMatrix.entries());

    auto tNumEntriesPerBlock = tNumColsPerBlock*tNumRowsPerBlock;

    std::vector<int> tColIndices(tMaxNumEntries,0);

    Epetra_SerialDenseMatrix tBlockEntry(tNumRowsPerBlock, tNumColsPerBlock);
    for(int iRow=0; iRow<tNumRows; iRow++)
    {

        auto tNumEntries = tRowMap_host(iRow+1) - tRowMap_host(iRow);
        auto tBegin = tRowMap_host(iRow);
        for(int i=0; i<tNumEntries; i++) tColIndices[i] = tColMap_host(tBegin++);
        tRetVal->BeginInsertGlobalValues(iRow, tNumEntries, tColIndices.data());
        for(int iEntryOrd=tRowMap_host(iRow); iEntryOrd<tRowMap_host(iRow+1); iEntryOrd++)
        {
            auto tBlockEntryOrd = iEntryOrd*tNumEntriesPerBlock;
            for(int j=0; j<tNumRowsPerBlock; j++)
            {
                for(int k=0; k<tNumColsPerBlock; k++)
                {
                    tBlockEntry(j,k) = tEntries_host(tBlockEntryOrd+j*tNumColsPerBlock+k);
                }
            }
            tRetVal->SubmitBlockEntry(tBlockEntry);
        }
        tRetVal->EndSubmitEntries();
    }

    tRetVal->FillComplete();

    return tRetVal;
}

/******************************************************************************//**
 * \brief Convert from ScalarVector to Epetra_Vector
**********************************************************************************/
rcp<Epetra_Vector>
EpetraSystem::fromVector(Plato::ScalarVector tInVector) const
{
    Teuchos::TimeMonitor LocalTimer(*mVectorConversionTimer);

    auto tRetVal = std::make_shared<Epetra_Vector>(*mBlockRowMap);
    if(tInVector.extent(0) != tRetVal->MyLength())
      throw std::domain_error("ScalarVector size does not match EpetraSystem map\n");

    Plato::Scalar* tRetValData;
    tRetVal->ExtractView(&tRetValData);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      tDataHostView(tRetValData, tInVector.extent(0));

    // copy to host from device
    Kokkos::deep_copy(tDataHostView, tInVector);

    return tRetVal;
}
/******************************************************************************//**
 * \brief Convert from Epetra_Vector to ScalarVector
**********************************************************************************/
void 
EpetraSystem::toVector(Plato::ScalarVector tOutVector, rcp<Epetra_Vector> tInVector) const
{
    Teuchos::TimeMonitor LocalTimer(*mVectorConversionTimer);

    auto tLength = tInVector->MyLength();
    auto tTemp = std::make_shared<Epetra_Vector>(*mBlockRowMap);
    if(tLength != tTemp->MyLength())
      throw std::domain_error("Epetra_Vector map does not match EpetraSystem map.");
    if(tOutVector.extent(0) != tTemp->MyLength())
      throw std::range_error("ScalarVector does not match EpetraSystem map.");
    Plato::Scalar* tInData;
    tInVector->ExtractView(&tInData);
    Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
      tInVector_host(tInData, tLength);
    Kokkos::deep_copy(tOutVector, tInVector_host);
}

/******************************************************************************//**
 * \brief EpetraLinearSolver constructor

 This constructor takes an Omega_h::Mesh and creates a new System.
**********************************************************************************/
EpetraLinearSolver::EpetraLinearSolver(
    const Teuchos::ParameterList& aSolverParams,
    Omega_h::Mesh&          aMesh,
    Comm::Machine           aMachine,
    int                     aDofsPerNode
) :
    mSolverParams(aSolverParams),
    mSystem(std::make_shared<EpetraSystem>(aMesh, aMachine, aDofsPerNode)),
    mLinearSolverTimer(Teuchos::TimeMonitor::getNewTimer("Analyze: Epetra Linear Solve"))
{
    mIterations = 1000;
    if(mSolverParams.isType<int>("Iterations"))
        mIterations = mSolverParams.get<int>("Iterations");

    mTolerance = 1e-14;
    if(mSolverParams.isType<double>("Tolerance"))
        mTolerance = mSolverParams.get<double>("Tolerance");
    
    mDisplayIterations = 0;
    if(mSolverParams.isType<int>("Display Iterations"))
        mDisplayIterations = mSolverParams.get<int>("Display Iterations");
}

/******************************************************************************//**
 * \brief Solve the linear system
**********************************************************************************/
void
EpetraLinearSolver::solve(
    Plato::CrsMatrix<Plato::OrdinalType> aA,
    Plato::ScalarVector   aX,
    Plato::ScalarVector   aB
) {
    auto tMatrix = mSystem->fromMatrix(aA);
    auto tSolution = mSystem->fromVector(aX);
    auto tForcing = mSystem->fromVector(aB);

    Epetra_VbrRowMatrix tVbrRowMatrix(tMatrix.get());
    Epetra_LinearProblem tProblem(&tVbrRowMatrix, tSolution.get(), tForcing.get());
    AztecOO tSolver(tProblem);

    setupSolver(tSolver);

    mLinearSolverTimer->start();
    tSolver.Iterate( mIterations, mTolerance );
    mLinearSolverTimer->stop(); mLinearSolverTimer->incrementNumCalls();
    
    const double* tSolverStatus = tSolver.GetAztecStatus();
    if (tSolverStatus[AZ_why] == AZ_normal)
    {
        // Do nothing
    }
    else if (tSolverStatus[AZ_why] == AZ_loss)
    {
        printf("Epetra Warning: Loss of numerical precision occured during linear solve.\n");
    }
    else if (tSolverStatus[AZ_why] == AZ_ill_cond)
    {
        printf("Epetra Warning: Hessenberg matrix in GMRES is ill-conditioned.\n");
    }
    else if (tSolverStatus[AZ_why] == AZ_maxits)
    {
        const std::string tWarningMessage = std::string("Epetra Warning: Specified maximum iterations (")
              + std::to_string(mIterations) + ") reached without convergence to requested tolerance ("
              + std::to_string(mTolerance) + ").";
        std::cout << tWarningMessage << std::endl;
    }
    else if (tSolverStatus[AZ_why] == AZ_param)
    {
        THROWERR("Epetra Error: User requested option not available.")
    }
    else if (tSolverStatus[AZ_why] == AZ_breakdown)
    {
        THROWERR("Epetra Error: Numerical breakdown occured during linear solve.")
    }

    if (mDisplayIterations > 0)
        printf("Epetra Lin. Solve %5.1f second(s), %4.0f iteration(s), %7.1e achieved relative tolerance (%7.1e requested)\n",
                tSolverStatus[AZ_solve_time], tSolverStatus[AZ_its], tSolverStatus[AZ_scaled_r], mTolerance);

    mSystem->toVector(aX, tSolution);
}

/******************************************************************************//**
 * \brief Setup the AztecOO solver
**********************************************************************************/
void
EpetraLinearSolver::setupSolver(AztecOO& aSolver)
{
    std::string tSolverType = "gmres";
    if(mSolverParams.isType<std::string>("Solver"))
        tSolverType = mSolverParams.get<std::string>("Solver");
    std::string tLowerSolverType = Plato::tolower(tSolverType);

    aSolver.SetAztecOption(AZ_output, 0); // Previously used tDisplayIterations here but now perform our own output.
    aSolver.SetAztecOption(AZ_subdomain_solve, AZ_ilu);
    aSolver.SetAztecOption(AZ_precond, AZ_dom_decomp);
    if (tLowerSolverType == "gmres")
    {
        aSolver.SetAztecOption(AZ_kspace,  mIterations);
        aSolver.SetAztecOption(AZ_scaling, AZ_row_sum);
        aSolver.SetAztecOption(AZ_solver,  AZ_gmres);
    }
    else if (tLowerSolverType == "cg")
    {
        aSolver.SetAztecOption(AZ_type_overlap, AZ_symmetric);
        aSolver.SetAztecOption(AZ_overlap, 0);
        aSolver.SetAztecOption(AZ_solver, AZ_cg);
    }
    else if (tLowerSolverType == "bicgstab")
    {
        aSolver.SetAztecOption(AZ_solver, AZ_bicgstab);
    }
    else
    {
        const std::string tErrorMessage = std::string("Epetra Error: Specified solver '")
              + tSolverType + "' not implemented. Current options are 'gmres', 'cg', and 'bicgstab'";
        THROWERR(tErrorMessage)
    }
}

} // end namespace Plato

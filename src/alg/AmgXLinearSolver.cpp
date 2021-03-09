#ifdef HAVE_AMGX
#include "alg/AmgXLinearSolver.hpp"
#include "AnalyzeMacros.hpp"
#include <amgx_c.h>
#include <string>
#include <fstream>
#include <mpi.h>

namespace Plato {

/******************************************************************************//**
 * @brief Load AmgX configuration string from file
**********************************************************************************/
std::string
AmgXLinearSolver::loadConfigString(std::string aConfigFile)
{
  std::string configString;

  std::ifstream infile;
  infile.open(aConfigFile, std::ifstream::in);
  if(infile){
    std::string line;
    std::stringstream config;
    while (std::getline(infile, line)){
      std::istringstream iss(line);
      config << iss.str();
    }
    configString = config.str();
  } else {
    std::cout << "*** Warning" << std::endl;
    std::cout << "*** AmgX configuration file (" << aConfigFile << ") not found." << std::endl;
    std::cout << "*** Using default settings." << std::endl;

    configString = \
    "{\
        \"config_version\": 2,\
        \"determinism_flag\": 1,\
        \"solver\": {\
            \"preconditioner\": {\
                \"print_grid_stats\": 1,\
                \"algorithm\": \"AGGREGATION\",\
                \"print_vis_data\": 0,\
                \"solver\": \"AMG\",\
                \"smoother\": {\
                    \"relaxation_factor\": 0.8,\
                    \"scope\": \"jacobi\",\
                    \"solver\": \"BLOCK_JACOBI\",\
                    \"monitor_residual\": 0,\
                    \"print_solve_stats\": 0\
                 },\
                \"print_solve_stats\": 0,\
                \"presweeps\": 0,\
                \"selector\": \"SIZE_8\",\
                \"coarse_solver\": \"NOSOLVER\",\
                \"max_iters\": 1,\
                \"monitor_residual\": 0,\
                \"store_res_history\": 0,\
                \"scope\": \"amg\",\
                \"max_levels\": 100,\
                \"postsweeps\": 4,\
                \"cycle\": \"V\"\
             },\
            \"solver\": \"PBICGSTAB\",\
            \"print_solve_stats\": 0,\
            \"obtain_timings\": 1,\
            \"max_iters\": 1000,\
            \"monitor_residual\": 1,\
            \"convergence\": \"ABSOLUTE\",\
            \"scope\": \"main\",\
            \"tolerance\": 1.0e-14,\
            \"norm\": \"L2\"\
         }\
    }";
  }

  return configString;
}

/******************************************************************************//**
 * @brief AmgXLinearSolver constructor
**********************************************************************************/
AmgXLinearSolver::AmgXLinearSolver(
    const Teuchos::ParameterList& aSolverParams,
    int aDofsPerNode
) :
    mDofsPerNode(aDofsPerNode),
    mDivergenceIsFatal(false)
{
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_initialize_plugins());
    AMGX_SAFE_CALL(AMGX_install_signal_handler());

    std::string tConfigFile("amgx.json");
    if(aSolverParams.isType<std::string>("Configuration File"))
    {
        tConfigFile = aSolverParams.get<std::string>("Configuration File");
    }
    auto tConfigString = loadConfigString(tConfigFile);
    AMGX_config_create(&mConfigHandle, tConfigString.c_str());

    if(aSolverParams.isType<bool>("Divergence is Fatal"))
    {
        mDivergenceIsFatal = aSolverParams.get<bool>("Divergence is Fatal");
    }

    // everything currently assumes exactly one MPI rank.
    MPI_Comm mpi_comm = MPI_COMM_SELF;
    int ndevices = 1;
    int devices[1];
    //it is critical to specify the current device, which is not always zero
    cudaGetDevice(&devices[0]);
    AMGX_resources_create(
        &mResources, mConfigHandle, &mpi_comm, ndevices, devices);

    AMGX_matrix_create(&mMatrixHandle,   mResources, AMGX_mode_dDDI);
    AMGX_vector_create(&mForcingHandle,  mResources, AMGX_mode_dDDI);
    AMGX_vector_create(&mSolutionHandle, mResources, AMGX_mode_dDDI);
    AMGX_solver_create(&mSolverHandle,   mResources, AMGX_mode_dDDI, mConfigHandle);
}

/******************************************************************************//**
 * @brief AmgXLinearSolver constructor
**********************************************************************************/
void
AmgXLinearSolver::solve(
    Plato::CrsMatrix<int> aA,
    Plato::ScalarVector   aX,
    Plato::ScalarVector   aB
) {

#ifndef NDEBUG
    check_inputs(aA, aX, aB);
#endif

    mSolution = aX;
    auto N = aX.size();
    auto nnz = aA.columnIndices().size();

    const int *row_map = aA.rowMap().data();
    const int *col_map = aA.columnIndices().data();
    const void *data   = aA.entries().data();
    const void *diag   = nullptr; // no exterior diagonal
    AMGX_matrix_upload_all(mMatrixHandle, N/mDofsPerNode, nnz, mDofsPerNode, mDofsPerNode, row_map, col_map, data, diag);

    AMGX_vector_upload(mForcingHandle, aB.size()/mDofsPerNode, mDofsPerNode, aB.data());
    AMGX_vector_upload(mSolutionHandle, aX.size()/mDofsPerNode, mDofsPerNode, aX.data());

    AMGX_solver_setup(mSolverHandle, mMatrixHandle);

    int err = cudaDeviceSynchronize();
    assert(err == cudaSuccess);
    auto solverErr = AMGX_solver_solve(mSolverHandle, mForcingHandle, mSolutionHandle);
    AMGX_SOLVE_STATUS tStatus;
    AMGX_solver_get_status(mSolverHandle, &tStatus);
    if (tStatus == AMGX_SOLVE_FAILED)
    {
        THROWERR("AMGX Solver Failed!");
    }
    else if (tStatus == AMGX_SOLVE_DIVERGED)
    {
        if (mDivergenceIsFatal)
        {
            THROWERR("AMGX Solver Diverged!");
        }
        else
        {
            WARNING("AMGX Solver Diverged!");
        }
    }
    AMGX_vector_download(mSolutionHandle, mSolution.data());
}

/******************************************************************************//**
 * @brief AmgXLinearSolver constructor
**********************************************************************************/
AmgXLinearSolver::
~AmgXLinearSolver()
{
    AMGX_solver_destroy    (mSolverHandle);
    AMGX_matrix_destroy    (mMatrixHandle);
    AMGX_vector_destroy    (mForcingHandle);
    AMGX_vector_destroy    (mSolutionHandle);
    AMGX_resources_destroy (mResources);

    AMGX_SAFE_CALL(AMGX_config_destroy(mConfigHandle));
    AMGX_SAFE_CALL(AMGX_finalize_plugins());
    AMGX_SAFE_CALL(AMGX_finalize());
}

/******************************************************************************//**
 * @brief sanity check for solve() arguments
**********************************************************************************/
void
AmgXLinearSolver::
check_inputs(
    const Plato::CrsMatrix<int> A,
    Plato::ScalarVector x,
    const Plato::ScalarVector b
) {
    auto ndofs = int(x.extent(0));
    assert(int(b.extent(0)) == ndofs);
    assert(ndofs % mDofsPerNode == 0);
    auto nblocks = ndofs / mDofsPerNode;
    auto row_map = A.rowMap();
    assert(int(row_map.extent(0)) == nblocks + 1);
    auto col_inds = A.columnIndices();
    auto nnz = int(col_inds.extent(0));
    assert(int(A.entries().extent(0)) == nnz * mDofsPerNode * mDofsPerNode);
    assert(cudaSuccess == cudaDeviceSynchronize());
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, nblocks), KOKKOS_LAMBDA(int i) {
        auto begin = row_map(i);
        assert(0 <= begin);
        auto end = row_map(i + 1);
        assert(begin <= end);
        if (i == nblocks - 1) assert(end == nnz);
        else assert(end < nnz);
        for (int ij = begin; ij < end; ++ij)
        {
            auto j = col_inds(ij);
            assert(0 <= j);
            assert(j < nblocks);
        }
    }, "check_inputs");
    assert(cudaSuccess == cudaDeviceSynchronize());
}

} // end namespace Plato
#endif // HAVE_AMGX

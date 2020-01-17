//
//  AmgXSparseLinearProblem.hpp
//  
//
//  Created by Roberts, Nathan V on 8/8/17.
//
//
#ifndef AMGX_SPARSE_LINEAR_PROBLEM_HPP
#define AMGX_SPARSE_LINEAR_PROBLEM_HPP

#include <Teuchos_Comm.hpp>
#include <Teuchos_RCP.hpp>

#ifdef HAVE_MPI
#include <Teuchos_DefaultMpiComm.hpp>
#endif

#include "alg/CrsLinearProblem.hpp"
#include "alg/AmgXConfigs.hpp"
#include <PlatoTypes.hpp>

#include <amgx_c.h>
#include <sstream>
#include <fstream>

#include <cassert>

namespace Plato
{

inline std::string get_config_string(bool aUseAbsoluteTolerance = false, Plato::OrdinalType aMaxIters = 1000)
{
    std::string tConfigString;

    Plato::Scalar tTolerance = 1e-12;
    Plato::OrdinalType tMaxIters = aMaxIters;

    std::ifstream tInputFile;
    tInputFile.open("amgx.json", std::ifstream::in);
    if(tInputFile)
    {
        std::string tLine;
        std::stringstream tConfig;
        while(std::getline(tInputFile, tLine))
        {
            std::istringstream tInputStringStream(tLine);
            tConfig << tInputStringStream.str();
        }
        tConfigString = tConfig.str();
    }
    else
    {
        tConfigString = Plato::configurationString("pcg_noprec", tTolerance, tMaxIters, aUseAbsoluteTolerance);
    }

    return tConfigString;
}
// function get_config_string

template<class Ordinal, Plato::OrdinalType BlockSize = 1>
class AmgXSparseLinearProblem : public CrsLinearProblem<Ordinal>
{

public:
    typedef Kokkos::View<Scalar*, MemSpace> Vector;
private:
    typedef Plato::OrdinalType RowMapEntryType;
    typedef CrsMatrix<Ordinal> Matrix;

    typedef Ordinal LocalOrdinal;
    typedef Ordinal GlobalOrdinal;

    AMGX_matrix_handle mMatrix;
    AMGX_vector_handle mRHS;
    AMGX_vector_handle mLHS;
    AMGX_resources_handle mResources;
    AMGX_solver_handle mSolver;
    AMGX_config_handle mSolverConfigurations;

    bool mHaveInitialized = false;

    Vector mSolution; // will want to copy here (from mLHS) in solve()...

public:
    static void initializeAMGX()
    {
        AMGX_SAFE_CALL(AMGX_initialize());
        AMGX_SAFE_CALL(AMGX_initialize_plugins());
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
    }

    AmgXSparseLinearProblem(const Matrix aA, Vector aX, const Vector aB,
                            std::string const& aSolverConfigString = configurationString("pcg_noprec")) :
            CrsLinearProblem<Ordinal>(aA, aX, aB)
    {
        check_inputs(aA, aX, aB);

        initializeAMGX();
        AMGX_config_create(&mSolverConfigurations, aSolverConfigString.c_str());
        // everything currently assumes exactly one MPI rank.
        MPI_Comm tMPI_COMM = MPI_COMM_SELF;
        Plato::OrdinalType tNumDevices = 1;
        Plato::OrdinalType tDevices[1];
        //it is critical to specify the current device, which is not always zero
        cudaGetDevice(&tDevices[0]);
        AMGX_resources_create(&mResources, mSolverConfigurations, &tMPI_COMM, tNumDevices, tDevices);
        AMGX_matrix_create(&mMatrix, mResources, AMGX_mode_dDDI);
        AMGX_vector_create(&mRHS, mResources, AMGX_mode_dDDI);
        AMGX_vector_create(&mLHS, mResources, AMGX_mode_dDDI);

        mSolution = aX;
        Ordinal tLocalNumVars = aX.size();
        Ordinal tNumNonZero = aA.columnIndices().size();

        AMGX_solver_create(&mSolver, mResources, AMGX_mode_dDDI, mSolverConfigurations);

        // This seems to do the right thing whether the data is on device or host. In our case it is on the device.
        this->uploadMatrix(aA, tLocalNumVars);
        this->uploadLeftHandSide(aX);
        this->uploadRightHandSide(aB);
    }

    void initializePreconditioner()
    {
        Kokkos::Profiling::pushRegion("AMGX_solver_setup");
        AMGX_solver_setup(mSolver, mMatrix);
        Kokkos::Profiling::popRegion();
    }

    void initializeSolver() // TODO: add mechanism for setting options
    {
        this->initializePreconditioner();
    }

    void uploadLeftHandSide(const Vector aLHS)
    {
        AMGX_vector_upload(mLHS, aLHS.size() / BlockSize, BlockSize, aLHS.data());
    }

    void setMaxIters(Plato::OrdinalType aMaxCGIters)
    {
        // NOTE: this does not work; we're getting the format wrong, somehow
        std::ostringstream tConfigStr;
        tConfigStr << "config_version=2" << std::endl;
        tConfigStr << "solver:max_iters=" << aMaxCGIters << std::endl;
        AMGX_config_add_parameters(&mSolverConfigurations, tConfigStr.str().c_str());
    }

    void uploadRightHandSide(const Vector aRHS)
    {
        AMGX_vector_upload(mRHS, aRHS.size() / BlockSize, BlockSize, aRHS.data());
    }

    void uploadMatrix(const Matrix & aMatrix, const Ordinal & aNumEquations)
    {
        const void *tData = aMatrix.entries().data();
        const void *tDiagData = nullptr; // no exterior diagonal
        const Plato::OrdinalType *tRowPtrs = aMatrix.rowMap().data();
        const Plato::OrdinalType *tColIndices = aMatrix.columnIndices().data();
        const Ordinal tNumNonZeros = aMatrix.columnIndices().size();
        AMGX_matrix_upload_all(mMatrix,
                               aNumEquations / BlockSize,
                               tNumNonZeros,
                               BlockSize,
                               BlockSize,
                               tRowPtrs,
                               tColIndices,
                               tData,
                               tDiagData);
    }

    void setTolerance(Plato::Scalar aTolerance)
    {
        // NOTE: this does not work; we're getting the format wrong, somehow
        std::ostringstream tConfigStr;
        tConfigStr << "config_version=2" << std::endl;
        tConfigStr << "solver:tolerance=" << aTolerance << std::endl;
        AMGX_config_add_parameters(&mSolverConfigurations, tConfigStr.str().c_str());
    }

    Plato::OrdinalType solve()
    {
        using namespace std;

        if(!mHaveInitialized)
        {
            this->initializeSolver();
            mHaveInitialized = true;
        }
        Plato::OrdinalType tErrorMsg = cudaDeviceSynchronize();
        assert(tErrorMsg == cudaSuccess);
        Kokkos::Profiling::pushRegion("AMGX_solver_solve");
        auto tSolverErr = AMGX_solver_solve(mSolver, mRHS, mLHS);
        Kokkos::Profiling::popRegion();
        Kokkos::Profiling::pushRegion("AMGX_vector_download");
        AMGX_vector_download(mLHS, mSolution.data());
        Kokkos::Profiling::popRegion();
        return tSolverErr;
    }

    ~AmgXSparseLinearProblem()
    {
        AMGX_solver_destroy(mSolver);
        AMGX_matrix_destroy(mMatrix);
        AMGX_vector_destroy(mRHS);
        AMGX_vector_destroy(mLHS);
        AMGX_resources_destroy(mResources);

        AMGX_SAFE_CALL(AMGX_config_destroy(mSolverConfigurations));

        AMGX_SAFE_CALL(AMGX_finalize_plugins());
        AMGX_SAFE_CALL(AMGX_finalize());
    }

    static void check_inputs(const Matrix aMatrix, Vector aLHS, const Vector aRHS)
    {
        auto tNumDofs = Plato::OrdinalType(aLHS.extent(0));
        assert(Plato::OrdinalType(aRHS.extent(0)) == tNumDofs);
        assert(tNumDofs % BlockSize == 0);
        auto tNumBlocks = tNumDofs / BlockSize;
        auto tRowMap = aMatrix.rowMap();
        assert(Plato::OrdinalType(tRowMap.extent(0)) == tNumBlocks + 1);
        auto tColIndices = aMatrix.columnIndices();
        auto tNumNonZero = Plato::OrdinalType(tColIndices.extent(0));
        assert(Plato::OrdinalType(aMatrix.entries().extent(0)) == tNumNonZero * BlockSize * BlockSize);
        assert(cudaSuccess == cudaDeviceSynchronize());
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumBlocks), KOKKOS_LAMBDA(Plato::OrdinalType aBlockIndex)
        {
            auto tBegin = tRowMap(aBlockIndex);
            assert(0 <= tBegin);
            auto tEnd = tRowMap(aBlockIndex + 1);
            assert(tBegin <= tEnd);
            if (aBlockIndex == tNumBlocks - 1) assert(tEnd == tNumNonZero);
            else assert(tEnd < tNumNonZero);
            for (Plato::OrdinalType tIJ = tBegin; tIJ < tEnd; ++tIJ)
            {
                auto tJ = tColIndices(tIJ);
                assert(0 <= tJ);
                assert(tJ < tNumBlocks);
            }
        }, "check_inputs");
        assert(cudaSuccess == cudaDeviceSynchronize());
    }

};
}

#endif /* AmgXSparseLinearProblem_h */

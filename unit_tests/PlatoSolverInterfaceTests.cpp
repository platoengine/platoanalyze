#include "PlatoTestHelpers.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_teuchos.hpp"

#include <AztecOO.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_MpiComm.h>
#include <Epetra_VbrMatrix.h>
#include <Epetra_VbrRowMatrix.h>
#include <Epetra_LinearProblem.h>

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>

#include "AnalyzeMacros.hpp"
#include <alg/ParallelComm.hpp>
#include "Simp.hpp"
#include "EssentialBCs.hpp"
#include "ScalarProduct.hpp"
#include "SimplexFadTypes.hpp"
#include "WorksetBase.hpp"
#include "VectorFunction.hpp"
#include "PhysicsScalarFunction.hpp"
#include "StateValues.hpp"
#include "ApplyConstraints.hpp"
#include "SimplexMechanics.hpp"
#include "Mechanics.hpp"
#include "ComputedField.hpp"
#include "ImplicitFunctors.hpp"
#include "LinearElasticMaterial.hpp"

#ifdef HAVE_AMGX
#include <alg/AmgXSparseLinearProblem.hpp>
#endif

#include <fenv.h>
#include <memory>

/******************************************************************************//**
 Questions:
  1.  The linear solver will need a comm.  Should it be global or an argument.  It's
      currently an argument.
  2.  What use cases do we want to support?
      -- solve with arguments, solver.solve(A, x, b)
      -- set matrix then solve, solver.set(A, b); solver.solve(x);
**********************************************************************************/




/******************************************************************************//**
 * @brief

 * @param [in]
 * @return
**********************************************************************************/

namespace Plato {
namespace Devel {

template <typename ClassT>
using rcp = std::shared_ptr<ClassT>;

/******************************************************************************//**
 * @brief Abstract solver interface

  Note that the solve() function takes 'native' matrix and vector types.  A next
  step would be to adopt generic matrix and vector interfaces that we can wrap
  around Epetra types, Tpetra types, Kokkos view-based types, etc.


**********************************************************************************/
class AbstractSolver
{
  public:
    virtual void solve(
        Plato::CrsMatrix<int> aA,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB
    ) = 0;
};

/******************************************************************************//**
 * @brief Abstract system interface

   This class contains the node and dof map information and permits persistence
   of this information between solutions.
**********************************************************************************/
class EpetraSystem
{
    rcp<Epetra_BlockMap> mBlockRowMap;
    rcp<Epetra_Comm>     mComm;

  public:
    EpetraSystem(
        Omega_h::Mesh& aMesh,
        Comm::Machine  aMachine,
        int            aDofsPerNode
    ) {
        mComm = aMachine.epetraComm;

        int tNumNodes = aMesh.nverts();
        mBlockRowMap = std::make_shared<Epetra_BlockMap>(tNumNodes, aDofsPerNode, 0, *mComm);

    }

    /******************************************************************************//**
     * @brief Convert from Plato::CrsMatrix<int> to Epetra_VbrMatrix
    **********************************************************************************/
    rcp<Epetra_VbrMatrix> fromMatrix(Plato::CrsMatrix<int> tInMatrix) const
    {
        int tNumRows = mBlockRowMap->NumMyElements();
        std::vector<int> tNumEntries(tNumRows, 0);
        auto tRowMap_host = Kokkos::create_mirror_view(tInMatrix.rowMap());
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

        auto tNumRowsPerBlock = tInMatrix.numRowsPerBlock();
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
     * @brief Convert from ScalarVector to Epetra_Vector
    **********************************************************************************/
    rcp<Epetra_Vector> fromVector(Plato::ScalarVector tInVector) const
    {
        auto tRetVal = std::make_shared<Epetra_Vector>(*mBlockRowMap);

        Plato::Scalar* tRetValData;
        tRetVal->ExtractView(&tRetValData);
        Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
          tDataHostView(tRetValData, tInVector.extent(0));

        // copy to host from device
        Kokkos::deep_copy(tDataHostView, tInVector);

        return tRetVal;
    }
    /******************************************************************************//**
     * @brief Convert from Epetra_Vector to ScalarVector
    **********************************************************************************/
    void toVector(Plato::ScalarVector tOutVector, rcp<Epetra_Vector> tInVector) const
    {
        Plato::Scalar* tInData;
        tInVector->ExtractView(&tInData);
        auto tLength = tInVector->MyLength();
        Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
          tInVector_host(tInData, tLength);
        Kokkos::deep_copy(tOutVector, tInVector_host);
    }
};


#ifdef HAVE_AMGX
#include <amgx_c.h>
/******************************************************************************//**
 * @brief Concrete AmgXLinearSolver
**********************************************************************************/
class AmgXLinearSolver : public AbstractSolver
{
  private:

    AMGX_matrix_handle    mMatrixHandle;
    AMGX_vector_handle    mForcingHandle;
    AMGX_vector_handle    mSolutionHandle;
    AMGX_solver_handle    mSolverHandle;
    AMGX_config_handle    mConfigHandle;
    AMGX_resources_handle mResources;

    int mDofsPerNode;

    Plato::ScalarVector mSolution;

    static void initializeAMGX()
    {
        AMGX_SAFE_CALL(AMGX_initialize());
        AMGX_SAFE_CALL(AMGX_initialize_plugins());
        AMGX_SAFE_CALL(AMGX_install_signal_handler());
    }

    static std::string loadConfigString(std::string aConfigFile)
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
      }
      return configString;
    }

  public:
    AmgXLinearSolver(
        const Teuchos::ParameterList& aSolverParams,
        int aDofsPerNode
    ) : mDofsPerNode(aDofsPerNode)
    {
      initializeAMGX();

      std::string tConfigFile("amgx.json");
      if(aSolverParams.isType<std::string>("Configuration File"))
      {
          tConfigFile = aSolverParams.get<std::string>("Configuration File");
      }
      auto tConfigString = loadConfigString(tConfigFile);
      AMGX_config_create(&mConfigHandle, tConfigString.c_str());

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

    void solve(
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
      AMGX_vector_download(mSolutionHandle, mSolution.data());
    }

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

    void check_inputs(const Plato::CrsMatrix<int> A, Plato::ScalarVector x, const Plato::ScalarVector b)
    {
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
        for (int ij = begin; ij < end; ++ij) {
          auto j = col_inds(ij);
          assert(0 <= j);
          assert(j < nblocks);
        }
      }, "check_inputs");
      assert(cudaSuccess == cudaDeviceSynchronize());
    }
};
#endif // HAVE_AMGX

/******************************************************************************//**
 * @brief Concrete EpetraLinearSolver
**********************************************************************************/
class EpetraLinearSolver : public AbstractSolver
{
    rcp<EpetraSystem> mSystem;

    const Teuchos::ParameterList& mSolverParams;

    int mIterations;
    Plato::Scalar mTolerance;

  public:
    /******************************************************************************//**
     * @brief EpetraLinearSolver constructor

     This constructor takes an Omega_h::Mesh and creates a new System.
    **********************************************************************************/
    EpetraLinearSolver(
        const Teuchos::ParameterList& aSolverParams,
        Omega_h::Mesh&          aMesh,
        Comm::Machine           aMachine,
        int                     aDofsPerNode
    ) :
        mSolverParams(aSolverParams),
        mSystem(std::make_shared<EpetraSystem>(aMesh, aMachine, aDofsPerNode))
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

    /******************************************************************************//**
     * @brief Solve the linear system
    **********************************************************************************/
    void solve(
        Plato::CrsMatrix<int> aA,
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

        tSolver.Iterate( mIterations, mTolerance );

        mSystem->toVector(aX, tSolution);
    }

    void setupSolver(AztecOO& aSolver)
    {
        int tDisplayIterations = 0;
        if(mSolverParams.isType<int>("Display Iterations"))
        {
            tDisplayIterations = mSolverParams.get<int>("Display Iterations");
        }

        aSolver.SetAztecOption(AZ_output, tDisplayIterations);

        // defaults (TODO: add options)
        aSolver.SetAztecOption(AZ_precond, AZ_ilu);
        aSolver.SetAztecOption(AZ_subdomain_solve, AZ_ilu);
        aSolver.SetAztecOption(AZ_precond, AZ_dom_decomp);
        aSolver.SetAztecOption(AZ_scaling, AZ_row_sum);
        aSolver.SetAztecOption(AZ_solver, AZ_gmres);
    }
};


/******************************************************************************//**
 * @brief Solver factory for AbstractSolvers
**********************************************************************************/
class SolverFactory
{
    const Teuchos::ParameterList& mSolverParams;

  public:
    SolverFactory(
        Teuchos::ParameterList& aSolverParams
    ) : mSolverParams(aSolverParams) { }

    rcp<AbstractSolver>
    create(
        Omega_h::Mesh&          aMesh,
        Comm::Machine           aMachine,
        int                     aDofsPerNode
    )
    {
        std::string tSolverType;
        if(mSolverParams.isType<std::string>("Solver"))
        {
            tSolverType = mSolverParams.get<std::string>("Solver");
        }
        else
        {
#ifdef HAVE_AMGX
            tSolverType = "AmgX";
#else
            tSolverType = "AztecOO";
#endif
        }

        if(tSolverType == "AztecOO")
        {
            return std::make_shared<Plato::Devel::EpetraLinearSolver>(mSolverParams, aMesh, aMachine, aDofsPerNode);
        }
        else
        if(tSolverType == "AmgX")
        {
#ifdef HAVE_AMGX
            return std::make_shared<Plato::Devel::AmgXLinearSolver>(mSolverParams, aDofsPerNode);
#else
            THROWERR("Not compiled with AmgX");
#endif
        }
        THROWERR("Requested solver type not found");
    }
};


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

  Plato::Devel::EpetraSystem tSystem(*mesh, tMachine, tNumDofsPerNode);

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
    auto tConfigString = AmgXLinearProblem::getConfigString();
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

    Plato::Devel::SolverFactory tSolverFactory(*tSolverParams);

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

    Plato::Devel::SolverFactory tSolverFactory(*tSolverParams);

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

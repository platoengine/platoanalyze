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

#include <alg/CrsLinearProblem.hpp>
#include <alg/ParallelComm.hpp>
#include "Simp.hpp"
#include "ScalarProduct.hpp"
#include "SimplexFadTypes.hpp"
#include "WorksetBase.hpp"
#include "VectorFunction.hpp"
#include "PhysicsScalarFunction.hpp"
#include "StateValues.hpp"
#include "ApplyConstraints.hpp"
#include "SimplexThermomechanics.hpp"
#include "Thermomechanics.hpp"
#include "ComputedField.hpp"
#include "ImplicitFunctors.hpp"
#include "LinearThermoelasticMaterial.hpp"

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
     * @brief Convert from abstract Matrix to MatrixT template type
    **********************************************************************************/
    rcp<Epetra_VbrMatrix> fromMatrix(Plato::CrsMatrix<int> tInMatrix) const
    {
        int tNumRows = mBlockRowMap->NumMyElements();
        std::vector<int> tNumEntries(tNumRows, 0);
        auto tRowMap_host = Kokkos::create_mirror_view(tInMatrix.rowMap());
        Kokkos::deep_copy(tRowMap_host, tInMatrix.rowMap());
        for(int iRow=0; iRow<tNumRows; iRow++)
        {
            tNumEntries[iRow] = tRowMap_host[iRow+1] - tRowMap_host[iRow];
        }

        auto tRetVal = std::make_shared<Epetra_VbrMatrix>(Copy, *mBlockRowMap, tNumEntries.data());

        auto tColMap_host = Kokkos::create_mirror_view(tInMatrix.columnIndices());
        Kokkos::deep_copy(tColMap_host, tInMatrix.columnIndices());

        auto tEntries_host = Kokkos::create_mirror_view(tInMatrix.entries());
        Kokkos::deep_copy(tEntries_host, tInMatrix.entries());

        auto tNumColsPerBlock = tInMatrix.numColsPerBlock();
        auto tNumRowsPerBlock = tInMatrix.numRowsPerBlock();
        auto tNumEntriesPerBlock = tNumColsPerBlock*tNumRowsPerBlock;
        bool tSumInto = false;

        for(int iRow=0; iRow<tNumRows; iRow++)
        {
            for(int iEntryOrd=tRowMap_host(iRow); iEntryOrd<tRowMap_host(iRow+1); iEntryOrd++)
            {
                auto iCol = tColMap_host(iEntryOrd);
                auto tVals = &(tEntries_host(iEntryOrd*tNumEntriesPerBlock));
                tRetVal->DirectSubmitBlockEntry(iRow, iCol, tVals, tNumColsPerBlock,
                                                tNumRowsPerBlock, tNumColsPerBlock, tSumInto);
            }
        }

        tRetVal->FillComplete();

        return tRetVal;
    }

    /******************************************************************************//**
     * @brief Convert from abstract Vector to VectorT template type
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

    virtual Epetra_Map dofRowMap()         const {}
    virtual Epetra_Map dofOverlapRowMap()  const {}
    virtual Epetra_Map nodeRowMap()        const {}
    virtual Epetra_Map nodeOverlapRowMap() const {}
};


/******************************************************************************//**
 * @brief Concrete EpetraLinearSolver
**********************************************************************************/
class EpetraLinearSolver : public AbstractSolver
{
    rcp<EpetraSystem>         mSystem;

  public:
    /******************************************************************************//**
     * @brief EpetraLinearSolver constructor
    
     This constructor takes an abstract Mesh and creates a new System.
    **********************************************************************************/
    EpetraLinearSolver(
        Teuchos::ParameterList& aSolverPparams,
        Omega_h::Mesh&          aMesh,
        Comm::Machine           aMachine,
        int                     aDofsPerNode
    ) :
        mSystem(std::make_shared<EpetraSystem>(aMesh, aMachine, aDofsPerNode))
    {}

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

        tSolver.Iterate( 10, 1e-8 );
    }
};



}
}


/******************************************************************************/
/*! 
  \brief 3D Thermoelastic problem
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( SolverInterfaceTests, Thermoelastic3D )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  // create mesh based density from host data
  //
  int tNumDofsPerNode = (spaceDim+1);
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based temperature from host data
  //
  Plato::ScalarVector state("temperature", tNumDofs);
  Kokkos::deep_copy(state, 1.0);

  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                    \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>                     \n"
    "  <Parameter name='Objective' type='string' value='My Internal Thermoelastic Energy'/>  \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                             \n"
    "  <ParameterList name='Elliptic'>                                                       \n"
    "    <ParameterList name='Penalty Function'>                                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                            \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "  <ParameterList name='My Internal Thermoelastic Energy'>                               \n"
    "    <Parameter name='Type' type='string' value='Scalar Function'/>                      \n"
    "    <Parameter name='Scalar Function Type' type='string' value='Internal Thermoelastic Energy'/>  \n"
    "    <ParameterList name='Penalty Function'>                                             \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                               \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "  <ParameterList name='Material Model'>                                                 \n"
    "    <ParameterList name='Isotropic Linear Thermoelastic'>                               \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>                  \n"
    "      <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e-5'/>   \n"
    "      <Parameter  name='Thermal Conductivity Coefficient' type='double' value='910.0'/> \n"
    "      <Parameter  name='Reference Temperature' type='double' value='0.0'/>              \n"
    "    </ParameterList>                                                                    \n"
    "  </ParameterList>                                                                      \n"
    "  <ParameterList name='Linear Solver'>                                                  \n"
    "  </ParameterList>                                                                      \n"
    "</ParameterList>                                                                        \n"
  );

  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  Plato::VectorFunction<::Plato::Thermomechanics<spaceDim>>
    vectorFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // compute and test constraint value
  //
  auto residual = vectorFunction.value(state, control);

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.gradient_u(state, control);


  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  auto tSolverParams = params->sublist("Linear Solver");
  Plato::Devel::EpetraLinearSolver tSolver(tSolverParams, *mesh, tMachine, /*dofs=*/ 4);
  tSolver.solve(*jacobian, state, residual);

}

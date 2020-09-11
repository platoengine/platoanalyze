#ifndef PLATO_EPETRA_SOLVER_HPP
#define PLATO_EPETRA_SOLVER_HPP

#include "PlatoAbstractSolver.hpp"
#include "alg/ParallelComm.hpp"

#include <Omega_h_mesh.hpp>

#include <AztecOO.h>
#include <Epetra_Map.h>
#include <Epetra_Vector.h>
#include <Epetra_MpiComm.h>
#include <Epetra_VbrMatrix.h>
#include <Epetra_VbrRowMatrix.h>
#include <Epetra_LinearProblem.h>
#include <Teuchos_ParameterList.hpp>


namespace Plato {

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
        int            aNumNodes,
        Comm::Machine  aMachine,
        int            aDofsPerNode
    );

    /******************************************************************************//**
     * @brief Convert from Plato::CrsMatrix<int> to Epetra_VbrMatrix
    **********************************************************************************/
    rcp<Epetra_VbrMatrix>
    fromMatrix(Plato::CrsMatrix<int> tInMatrix) const;

    /******************************************************************************//**
     * @brief Convert from ScalarVector to Epetra_Vector
    **********************************************************************************/
    rcp<Epetra_Vector>
    fromVector(Plato::ScalarVector tInVector) const;

    /******************************************************************************//**
     * @brief Convert from Epetra_Vector to ScalarVector
    **********************************************************************************/
    void
    toVector(Plato::ScalarVector tOutVector, rcp<Epetra_Vector> tInVector) const;

    /******************************************************************************//**
     * @brief get EpetraSystem map 
    **********************************************************************************/
    rcp<Epetra_BlockMap> getMap() const {return mBlockRowMap;}
};

/******************************************************************************//**
 * @brief Concrete EpetraLinearSolver
**********************************************************************************/
class EpetraLinearSolver : public AbstractSolver
{
    rcp<EpetraSystem> mSystem;

    Teuchos::ParameterList mSolverParams;

    int mIterations;
    Plato::Scalar mTolerance;

  public:
    /******************************************************************************//**
     * @brief EpetraLinearSolver constructor

     This constructor takes an Omega_h::Mesh and creates a new System.
    **********************************************************************************/
    EpetraLinearSolver(
        const Teuchos::ParameterList&                   aSolverParams,
        int                                             aNumNodes,
        Comm::Machine                                   aMachine,
        int                                             aDofsPerNode
    );

    /******************************************************************************//**
     * @brief EpetraLinearSolver constructor with MPCs

     This constructor takes an Omega_h::Mesh and MultipointConstraints and creates a new System.
    **********************************************************************************/
    EpetraLinearSolver(
        const Teuchos::ParameterList&                   aSolverParams,
        int                                             aNumNodes,
        Comm::Machine                                   aMachine,
        int                                             aDofsPerNode,
        std::shared_ptr<Plato::MultipointConstraints>   aMPCs
    );

    /******************************************************************************//**
     * @brief Solve the linear system
    **********************************************************************************/
    void
    innerSolve(
        Plato::CrsMatrix<int> aA,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB
    ) override;

    /******************************************************************************//**
     * @brief Setup the AztecOO solver
    **********************************************************************************/
    void
    setupSolver(AztecOO& aSolver);
};

} // end namespace Plato

#endif

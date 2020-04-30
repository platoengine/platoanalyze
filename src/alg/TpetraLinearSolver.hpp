#ifndef PLATO_TPETRA_SOLVER_HPP
#define PLATO_TPETRA_SOLVER_HPP

#include "PlatoAbstractSolver.hpp"
#include "alg/ParallelComm.hpp"

#include <Omega_h_mesh.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Comm.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_Vector.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace Plato {

  using Tpetra_Map = Tpetra::Map<Plato::OrdinalType, Plato::OrdinalType>;
  using Tpetra_Vector = Tpetra::Vector<Plato::Scalar, Plato::OrdinalType, Plato::OrdinalType>;
  using Tpetra_Matrix = Tpetra::CrsMatrix<Plato::Scalar, Plato::OrdinalType, Plato::OrdinalType>;

/******************************************************************************//**
 * @brief Abstract system interface

   This class contains the node and dof map information and permits persistence
   of this information between solutions.
**********************************************************************************/
class TpetraSystem
{
  Teuchos::RCP<Tpetra_Map> mMap;
  Teuchos::RCP<const Teuchos::Comm<int>>  mComm;

  public:
    TpetraSystem(
        Omega_h::Mesh& aMesh,
        Comm::Machine  aMachine,
        int            aDofsPerNode
    );

    /******************************************************************************//**
     * @brief Convert from Plato::CrsMatrix<int> to Epetra_VbrMatrix
    **********************************************************************************/
    Teuchos::RCP<Tpetra_Matrix>
    fromMatrix(Plato::CrsMatrix<int> tInMatrix) const;

    /******************************************************************************//**
     * @brief Convert from ScalarVector to Tpetra_Vector
    **********************************************************************************/
    Teuchos::RCP<Tpetra_Vector>
    fromVector(Plato::ScalarVector tInVector) const;

    /******************************************************************************//**
     * @brief Convert from Tpetra_Vector to ScalarVector
    **********************************************************************************/
    void
    toVector(Plato::ScalarVector tOutVector, Teuchos::RCP<Tpetra_Vector> tInVector) const;

    /******************************************************************************//**
     * @brief get TpetraSystem map 
    **********************************************************************************/
    Teuchos::RCP<Tpetra_Map> getMap() const {return mMap;}
};

/******************************************************************************//**
 * @brief Concrete TpetraLinearSolver
**********************************************************************************/
class TpetraLinearSolver : public AbstractSolver
{
    Teuchos::RCP<TpetraSystem> mSystem;

    Teuchos::ParameterList mSolverParams;

    int mIterations;
    Plato::Scalar mTolerance;

  public:
    /******************************************************************************//**
     * @brief TpetraLinearSolver constructor

     This constructor takes an Omega_h::Mesh and creates a new System.
    **********************************************************************************/
    TpetraLinearSolver(
        const Teuchos::ParameterList& aSolverParams,
        Omega_h::Mesh&          aMesh,
        Comm::Machine           aMachine,
        int                     aDofsPerNode
    );

    /******************************************************************************//**
     * @brief Solve the linear system
    **********************************************************************************/
    void
    solve(
        Plato::CrsMatrix<int> aA,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB
    );

    // /******************************************************************************//**
    //  * @brief Setup the AztecOO solver
    // **********************************************************************************/
    // void
    // setupSolver(AztecOO& aSolver);
};

} // end namespace Plato

#endif

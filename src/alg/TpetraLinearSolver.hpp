#ifndef PLATO_TPETRA_SOLVER_HPP
#define PLATO_TPETRA_SOLVER_HPP

#include "PlatoAbstractSolver.hpp"
#include "alg/ParallelComm.hpp"

#include <Omega_h_mesh.hpp>

#include <Teuchos_ParameterList.hpp>
#include <Teuchos_Comm.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_MultiVector.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace Plato {

  using Tpetra_Map = Tpetra::Map<int, Plato::OrdinalType>;
  using Tpetra_MultiVector = Tpetra::MultiVector<Plato::Scalar, int, Plato::OrdinalType>;
  using Tpetra_Matrix = Tpetra::CrsMatrix<Plato::Scalar, int, Plato::OrdinalType>;
  using Tpetra_Operator = Tpetra::Operator<Plato::Scalar, int, Plato::OrdinalType>;

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
     * @brief Convert from Plato::CrsMatrix<int> to Tpetra_Matrix
    **********************************************************************************/
    Teuchos::RCP<Tpetra_Matrix>
    fromMatrix(const Plato::CrsMatrix<Plato::OrdinalType> tInMatrix) const;

    /******************************************************************************//**
     * @brief Convert from ScalarVector to Tpetra_MultiVector
    **********************************************************************************/
    Teuchos::RCP<Tpetra_MultiVector>
    fromVector(const Plato::ScalarVector tInVector) const;

    /******************************************************************************//**
     * @brief Convert from Tpetra_MultiVector to ScalarVector
    **********************************************************************************/
    void
    toVector(Plato::ScalarVector& tOutVector, const Teuchos::RCP<Tpetra_MultiVector> tInVector) const;

    /******************************************************************************//**
     * @brief get TpetraSystem map 
    **********************************************************************************/
    Teuchos::RCP<Tpetra_Map> getMap() const {return mMap;}

  private:
      void checkInputMatrixSize(const Plato::CrsMatrix<Plato::OrdinalType> aInMatrix,
               Kokkos::View<Plato::OrdinalType*, MemSpace>::HostMirror aRowMap) const;
};

/******************************************************************************//**
 * @brief Concrete TpetraLinearSolver
**********************************************************************************/
class TpetraLinearSolver : public AbstractSolver
{
    Teuchos::RCP<TpetraSystem> mSystem;

    std::string mSolverPackage;
    std::string mSolver;
    std::string mPreconditionerPackage;
    std::string mPreconditionerType;

    Teuchos::ParameterList mSolverOptions;
    Teuchos::ParameterList mPreconditionerOptions;

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
        Plato::CrsMatrix<Plato::OrdinalType> aA,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB
    );

    /******************************************************************************//**
     * @brief Setup the Belos solver
    **********************************************************************************/
    template<class MV, class OP>
    void
    belosSolve (std::ostream& out, Teuchos::RCP<const OP> A, Teuchos::RCP<MV> X, Teuchos::RCP<const MV> B, Teuchos::RCP<const OP> M);
};

} // end namespace Plato

#endif

#ifndef PLATO_ABSTRACT_SOLVER_HPP
#define PLATO_ABSTRACT_SOLVER_HPP
#include <memory>

#include "PlatoStaticsTypes.hpp"
#include "MultipointConstraints.hpp"
#include "PlatoMathHelpers.hpp"
#include "BLAS1.hpp"

namespace Plato {

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
  protected:
    std::shared_ptr<Plato::MultipointConstraints>   mSystemMPCs;

    AbstractSolver() : mSystemMPCs(nullptr) {}

    virtual void innerSolve(
        Plato::CrsMatrix<int> aA,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB
    ) = 0;

  public:
    AbstractSolver(std::shared_ptr<Plato::MultipointConstraints> aMPCs) : mSystemMPCs(aMPCs) {}

    void solve(
        Plato::CrsMatrix<int> aAf,
        /* Plato::CrsMatrix<int> aA, */
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB)
    {
        Teuchos::RCP<Plato::CrsMatrixType> aA(&aAf);

        if(mSystemMPCs)
        {
            const Plato::OrdinalType tNumNodes = mSystemMPCs->getNumTotalNodes();
            const Plato::OrdinalType tNumCondensedNodes = mSystemMPCs->getNumCondensedNodes();

            Plato::OrdinalType tNumDofsPerNode = mSystemMPCs->getNumDofsPerNode();
            auto tNumDofs = tNumNodes*tNumDofsPerNode;
            auto tNumCondensedDofs = tNumCondensedNodes*tNumDofsPerNode;

            // get MPC condensation matrices and RHS
            Teuchos::RCP<Plato::CrsMatrixType> tTransformMatrix = mSystemMPCs->getTransformMatrix();
            Teuchos::RCP<Plato::CrsMatrixType> tTransformMatrixTranspose = mSystemMPCs->getTransformMatrixTranspose();
            Plato::ScalarVector tMpcRhs = mSystemMPCs->getRhsVector();

            // build condensed matrix
            auto tCondensedALeft = Teuchos::rcp( new Plato::CrsMatrixType(tNumDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode) );
            auto tCondensedA     = Teuchos::rcp( new Plato::CrsMatrixType(tNumCondensedDofs, tNumCondensedDofs, tNumDofsPerNode, tNumDofsPerNode) );
      
            /* Plato::MatrixMatrixMultiply(aA, *tTransformMatrix, *tCondensedALeft); */
            /* Plato::MatrixMatrixMultiply(*tTransformMatrixTranspose, *tCondensedALeft, *tCondensedA); */
            Plato::MatrixMatrixMultiply(aA, tTransformMatrix, tCondensedALeft);
            Plato::MatrixMatrixMultiply(tTransformMatrixTranspose, tCondensedALeft, tCondensedA);

            // build condensed vector
            Plato::ScalarVector tInnerB = aB;
            Plato::blas1::scale(-1.0, tMpcRhs);
            Plato::MatrixTimesVectorPlusVector(aA, tMpcRhs, tInnerB);
      
            Plato::ScalarVector tCondensedB("Condensed RHS Vector", tNumCondensedDofs);
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tCondensedB);
      
            Plato::MatrixTimesVectorPlusVector(tTransformMatrixTranspose, tInnerB, tCondensedB);
            /* Plato::MatrixTimesVectorPlusVector(*tTransformMatrixTranspose, tInnerB, tCondensedB); */

            // solve condensed system
            Plato::ScalarVector tCondensedX("Condensed Solution", tNumCondensedDofs);
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tCondensedX);

            this->innerSolve(*tCondensedA, tCondensedX, tCondensedB);
            
            // get full solution vector
            Plato::ScalarVector tFullX("Full State Solution", aX.extent(0));
            Plato::blas1::copy(tMpcRhs, tFullX);
            Plato::blas1::scale(-1.0, tFullX); // since tMpcRhs was scaled by -1 above, set back to original values
      
            Plato::MatrixTimesVectorPlusVector(tTransformMatrix, tCondensedX, tFullX);
            /* Plato::MatrixTimesVectorPlusVector(*tTransformMatrix, tCondensedX, tFullX); */
            Plato::blas1::axpy<Plato::ScalarVector>(1.0, tFullX, aX);
        }
        else
        {
            this->innerSolve(*aA, aX, aB);
        }
    }
};
} // end namespace Plato

#endif

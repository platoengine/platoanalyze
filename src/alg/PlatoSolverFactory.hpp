#ifndef PLATO_SOLVER_FACTORY_HPP
#define PLATO_SOLVER_FACTORY_HPP

#include "alg/AmgXLinearSolver.hpp"
#include "alg/EpetraLinearSolver.hpp"
#ifdef PLATO_TPETRA
#include "alg/TpetraLinearSolver.hpp"
#endif

namespace Plato {

/******************************************************************************//**
 * \brief Solver factory for AbstractSolvers
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
    );
};

} // end Plato namespace

#endif

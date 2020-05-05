#include "alg/PlatoSolverFactory.hpp"

namespace Plato {

/******************************************************************************//**
 * @brief Solver factory for AbstractSolvers
**********************************************************************************/
rcp<AbstractSolver>
SolverFactory::create(
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
        return std::make_shared<Plato::EpetraLinearSolver>(mSolverParams, aMesh, aMachine, aDofsPerNode);
    }
    if(tSolverType == "Belos")
    {
        return std::make_shared<Plato::TpetraLinearSolver>(mSolverParams, aMesh, aMachine, aDofsPerNode);
    }
    else if(tSolverType == "AmgX")
    {
#ifdef HAVE_AMGX
        return std::make_shared<Plato::AmgXLinearSolver>(mSolverParams, aDofsPerNode);
#else
        THROWERR("Not compiled with AmgX");
#endif
    }
    THROWERR("Requested solver type not found");
}

} // end namespace Plato

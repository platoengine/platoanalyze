#include "alg/PlatoSolverFactory.hpp"

namespace Plato {

/******************************************************************************//**
 * @brief Solver factory for AbstractSolvers
**********************************************************************************/
rcp<AbstractSolver>
SolverFactory::create(
    int                                             aNumNodes,
    Comm::Machine                                   aMachine,
    int                                             aDofsPerNode
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
        return std::make_shared<Plato::EpetraLinearSolver>(mSolverParams, aNumNodes, aMachine, aDofsPerNode);
    }
#ifdef PLATO_TPETRA
    if(tSolverType == "Belos")
    {
        return std::make_shared<Plato::TpetraLinearSolver>(mSolverParams, aNumNodes, aMachine, aDofsPerNode);
    }
#endif
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

/******************************************************************************//**
 * @brief Solver factory for AbstractSolvers with MPCs
**********************************************************************************/
rcp<AbstractSolver>
SolverFactory::create(
    int                                             aNumNodes,
    Comm::Machine                                   aMachine,
    int                                             aDofsPerNode,
    std::shared_ptr<Plato::MultipointConstraints>   aMPCs
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
        Plato::OrdinalType tNumCondensedNodes = aMPCs->getNumCondensedNodes();
        return std::make_shared<Plato::EpetraLinearSolver>(mSolverParams, tNumCondensedNodes, aMachine, aDofsPerNode, aMPCs);
    }
#ifdef PLATO_TPETRA
    if(tSolverType == "Belos")
    {
        Plato::OrdinalType tNumCondensedNodes = aMPCs->getNumCondensedNodes();
        return std::make_shared<Plato::TpetraLinearSolver>(mSolverParams, tNumCondensedNodes, aMachine, aDofsPerNode, aMPCs);
    }
#endif
    else if(tSolverType == "AmgX")
    {
#ifdef HAVE_AMGX
        return std::make_shared<Plato::AmgXLinearSolver>(mSolverParams, aDofsPerNode, aMPCs);
#else
        THROWERR("Not compiled with AmgX");
#endif
    }
    THROWERR("Requested solver type not found");
}

} // end namespace Plato

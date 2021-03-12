#include "alg/PlatoSolverFactory.hpp"

namespace Plato {

std::string determineSolverStack(const Teuchos::ParameterList& tSolverParams)
{
  std::string tSolverStack;
  if(tSolverParams.isType<std::string>("Solver Stack"))
  {
      tSolverStack = tSolverParams.get<std::string>("Solver Stack");
  }
  else
  {
#ifdef HAVE_AMGX
      tSolverStack = "AmgX";
#elif PLATO_TPETRA
      tSolverStack = "Tpetra";
#else
      tSolverStack = "Epetra";
#endif
  }

  return tSolverStack;
}

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
  std::string tSolverStack = determineSolverStack(mSolverParams);

    if(tSolverStack == "Epetra")
    {
        return std::make_shared<Plato::EpetraLinearSolver>(mSolverParams, aNumNodes, aMachine, aDofsPerNode);
    }
    else if(tSolverStack == "Tpetra")
    {
#ifdef PLATO_TPETRA
        return std::make_shared<Plato::TpetraLinearSolver>(mSolverParams, aNumNodes, aMachine, aDofsPerNode);
#else
        THROWERR("Not compiled with Tpetra");
#endif
  }
  else if(tSolverStack == "AmgX")
  {
#ifdef HAVE_AMGX
      return std::make_shared<Plato::AmgXLinearSolver>(mSolverParams, aDofsPerNode);
#else
      THROWERR("Not compiled with AmgX");
#endif
  }
  THROWERR("Requested solver stack not found");
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

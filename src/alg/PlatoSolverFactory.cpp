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
    Omega_h::Mesh&          aMesh,
    Comm::Machine           aMachine,
    int                     aDofsPerNode
)
{
  std::string tSolverStack = determineSolverStack(mSolverParams);

  if(tSolverStack == "Epetra")
  {
      return std::make_shared<Plato::EpetraLinearSolver>(mSolverParams, aMesh, aMachine, aDofsPerNode);
  }
  else if(tSolverStack == "Tpetra")
  {
#ifdef PLATO_TPETRA
      return std::make_shared<Plato::TpetraLinearSolver>(mSolverParams, aMesh, aMachine, aDofsPerNode);
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

} // end namespace Plato

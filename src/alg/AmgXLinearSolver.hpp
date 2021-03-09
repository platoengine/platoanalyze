#ifndef AMGX_LINEAR_SOLVER_HPP
#define AMGX_LINEAR_SOLVER_HPP

#ifdef HAVE_AMGX
#include "alg/PlatoAbstractSolver.hpp"

#include <Teuchos_ParameterList.hpp>

#include <amgx_c.h>

namespace Plato {

/******************************************************************************//**
 * @brief Concrete AmgXLinearSolver
**********************************************************************************/
class AmgXLinearSolver : public AbstractSolver
{
  private:

    AMGX_matrix_handle    mMatrixHandle;
    AMGX_vector_handle    mForcingHandle;
    AMGX_vector_handle    mSolutionHandle;
    AMGX_solver_handle    mSolverHandle;
    AMGX_config_handle    mConfigHandle;
    AMGX_resources_handle mResources;

    int mDofsPerNode;

    bool mDivergenceIsFatal;

    Plato::ScalarVector mSolution;

    static std::string loadConfigString(std::string aConfigFile);

  public:
    AmgXLinearSolver(
        const Teuchos::ParameterList& aSolverParams,
        int aDofsPerNode
    );

    void solve(
        Plato::CrsMatrix<int> aA,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB
    );

    ~AmgXLinearSolver();

    void check_inputs(
        const Plato::CrsMatrix<int> A,
        Plato::ScalarVector x,
        const Plato::ScalarVector b
    );
};

} // end namespace Plato

#endif // HAVE_AMGX

#endif

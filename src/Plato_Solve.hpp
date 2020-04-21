#ifndef PLATO_SOLVE_HPP
#define PLATO_SOLVE_HPP

#include <memory>

#include "PlatoMathFunctors.hpp"
#include "PlatoStaticsTypes.hpp"

#include "alg/ParallelComm.hpp"
#ifdef HAVE_AMGX
#include "alg/AmgXSparseLinearProblem.hpp"
#endif

namespace Plato {

namespace Solve {

    /******************************************************************************//**
     * \brief Solve linear system, A x = b.
     * \param [in]     a_A Matrix, A
     * \param [in/out] a_x Solution vector, x, with initial guess
     * \param [in]     a_b Forcing vector, b
     * \param [in]     aUseAbsoluteTolerance enables absolute stopping tolerance measure
    **********************************************************************************/
    template <Plato::OrdinalType NumDofsPerNode>
    void Consistent(
        Teuchos::RCP<Plato::CrsMatrixType> a_A,
        Plato::ScalarVector a_x,
        Plato::ScalarVector a_b,
        bool aUseAbsoluteTolerance = false
        )
        {
#ifdef HAVE_AMGX
              using AmgXLinearProblem = Plato::AmgXSparseLinearProblem< Plato::OrdinalType, NumDofsPerNode>;
              auto tConfigString = Plato::get_config_string(aUseAbsoluteTolerance);
              AmgXLinearProblem tSolver(*a_A, a_x, a_b, tConfigString);
              tSolver.solve();
#endif
        }

    /******************************************************************************//**
     * \brief Approximate solution for linear system, A x = b, by x = R^-1 b, where
     *        R is the row sum of A.
     * \param [in]     a_A Matrix, A
     * \param [in/out] a_x Solution vector, x, with initial guess
     * \param [in]     a_b Forcing vector, b
    **********************************************************************************/
    template <Plato::OrdinalType NumDofsPerNode>
    void RowSummed(
        Teuchos::RCP<Plato::CrsMatrixType> a_A, 
        Plato::ScalarVector a_x,
        Plato::ScalarVector a_b)
        {

            Plato::RowSum tRowSumFunctor(a_A);

            Plato::InverseWeight<NumDofsPerNode> tInverseWeight;

            Plato::ScalarVector tRowSum("row sum", a_x.extent(0));

            // a_x[i] 1.0/sum_j(a_A[i,j]) * a_b[i]
            auto tNumBlockRows = a_A->rowMap().size() - 1;
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumBlockRows), LAMBDA_EXPRESSION(const Plato::OrdinalType& aBlockRowOrdinal)
            {
                // compute row sum
                tRowSumFunctor(aBlockRowOrdinal, tRowSum);

                // apply inverse weight
                tInverseWeight(aBlockRowOrdinal, tRowSum, a_b, a_x, /*scale=*/-1.0);
                
            }, "row sum inverse");
        }

} // namespace Solve

} // namespace Plato

#endif

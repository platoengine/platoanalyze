#ifndef PLATO_ABSTRACT_SOLVER_HPP
#define PLATO_ABSTRACT_SOLVER_HPP
#include <memory>

#include "PlatoStaticsTypes.hpp"

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
  public:
    virtual void solve(
        Plato::CrsMatrix<int> aA,
        Plato::ScalarVector   aX,
        Plato::ScalarVector   aB
    ) = 0;
};
} // end namespace Plato

#endif

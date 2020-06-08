//
//  CrsLinearProblem.hpp
//
//
//  Created by Roberts, Nathan V on 6/20/17.
//
//

#pragma once

#include "alg/CrsMatrix.hpp"
#include <Kokkos_Core.hpp>

namespace Plato {

template <class Ordinal>
class CrsLinearProblem
{
 private:
  typedef Kokkos::View<Scalar *, MemSpace>          Vector;
  typedef Kokkos::View<Scalar **, Plato::Layout, MemSpace>         MultiVector;
  typedef CrsMatrix<Ordinal> Matrix;

  Matrix _A;
  Vector _x, _b;  //left-hand side (solution), right-hand side
 public:
  CrsLinearProblem(const Matrix &Aa, Vector &ex, const Vector &be)
      : _A(Aa), _x(ex), _b(be) {}
  virtual ~CrsLinearProblem(){}

  Matrix &A() { return _A; }
  Vector &b() { return _b; }
  Vector &x() { return _x; }

  virtual void initializeSolver() {}
  // concrete subclasses should know how to solve:
  virtual int solve() = 0;
};
// class CrsLinearProblem

}
// namespace Plato


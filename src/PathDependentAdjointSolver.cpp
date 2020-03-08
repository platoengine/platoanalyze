/*
 * PathDependentAdjointSolver.hpp
 *
 *  Created on: Mar 2, 2020
 */

#include "PathDependentAdjointSolver.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainPlasticity<3>>;
#endif


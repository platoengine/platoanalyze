/*
 * PathDependentAdjointSolver.hpp
 *
 *  Created on: Mar 2, 2020
 */

#include "PathDependentAdjointSolver.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainPlasticity<1>>;
template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainThermoPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainPlasticity<2>>;
template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainThermoPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainPlasticity<3>>;
template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainThermoPlasticity<3>>;
#endif


/*
 * NewtonRaphsonSolver.cpp
 *
 *  Created on: Mar 3, 2020
 */

#include "NewtonRaphsonSolver.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainPlasticity<1>>;
template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainThermoPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainPlasticity<2>>;
template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainThermoPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainPlasticity<3>>;
template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainThermoPlasticity<3>>;
#endif


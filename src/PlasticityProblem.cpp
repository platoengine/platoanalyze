/*
 * PlasticityProblem.cpp
 *
 *  Created on: Mar 3, 2020
 */

#include "PlasticityProblem.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<1>>;
template class Plato::PlasticityProblem<Plato::InfinitesimalStrainThermoPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<2>>;
template class Plato::PlasticityProblem<Plato::InfinitesimalStrainThermoPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<3>>;
template class Plato::PlasticityProblem<Plato::InfinitesimalStrainThermoPlasticity<3>>;
#endif


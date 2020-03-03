/*
 * PlasticityProblem.cpp
 *
 *  Created on: Mar 3, 2020
 */

#include "PlasticityProblem.hpp"

#ifdef PLATOANALYZE_2D
template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<3>>;
#endif


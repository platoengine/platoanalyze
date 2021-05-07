/*
 * ElastoPlasticityTest.cpp
 *
 *  Created on: Mar 3, 2020
 */

#include "BasicLocalScalarFunction.hpp"

#ifdef PLATOANALYZE_2D
template class Plato::BasicLocalScalarFunction<Plato::InfinitesimalStrainPlasticity<2>>;
template class Plato::BasicLocalScalarFunction<Plato::InfinitesimalStrainThermoPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::BasicLocalScalarFunction<Plato::InfinitesimalStrainPlasticity<3>>;
template class Plato::BasicLocalScalarFunction<Plato::InfinitesimalStrainThermoPlasticity<3>>;
#endif

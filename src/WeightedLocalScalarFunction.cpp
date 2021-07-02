/*
 * WeightedLocalScalarFunction.cpp
 *
 *  Created on: Mar 8, 2020
 */

#include "WeightedLocalScalarFunction.hpp"

#ifdef PLATOANALYZE_2D
template class Plato::WeightedLocalScalarFunction<Plato::InfinitesimalStrainPlasticity<2>>;
template class Plato::WeightedLocalScalarFunction<Plato::InfinitesimalStrainThermoPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::WeightedLocalScalarFunction<Plato::InfinitesimalStrainPlasticity<3>>;
template class Plato::WeightedLocalScalarFunction<Plato::InfinitesimalStrainThermoPlasticity<3>>;
#endif

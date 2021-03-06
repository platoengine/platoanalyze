/*
 * ElastoPlasticityTest.cpp
 *
 *  Created on: Mar 3, 2020
 */

#include "BasicLocalScalarFunction.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::BasicLocalScalarFunction<Plato::InfinitesimalStrainPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::BasicLocalScalarFunction<Plato::InfinitesimalStrainPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::BasicLocalScalarFunction<Plato::InfinitesimalStrainPlasticity<3>>;
#endif

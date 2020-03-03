/*
 * ElastoPlasticityTest.cpp
 *
 *  Created on: Mar 3, 2020
 */

#include "BasicLocalScalarFunctionInc.hpp"

#ifdef PLATOANALYZE_2D
template class Plato::BasicLocalScalarFunctionInc<Plato::InfinitesimalStrainPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::BasicLocalScalarFunctionInc<Plato::InfinitesimalStrainPlasticity<3>>;
#endif



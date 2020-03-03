/*
 * PathDependentScalarFunctionFactory.cpp
 *
 *  Created on: Mar 3, 2020
 */

#include "PathDependentScalarFunctionFactory.hpp"

#ifdef PLATOANALYZE_2D
template class Plato::PathDependentScalarFunctionFactory<Plato::InfinitesimalStrainPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::PathDependentScalarFunctionFactory<Plato::InfinitesimalStrainPlasticity<3>>;
#endif


/*
 * PathDependentScalarFunctionFactory.cpp
 *
 *  Created on: Mar 3, 2020
 */

#include "PathDependentScalarFunctionFactory.hpp"
#include "PathDependentScalarFunctionFactory_def.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::PathDependentScalarFunctionFactory<Plato::InfinitesimalStrainPlasticity<1>>;
template class Plato::PathDependentScalarFunctionFactory<Plato::InfinitesimalStrainPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::PathDependentScalarFunctionFactory<Plato::InfinitesimalStrainPlasticity<2>>;
template class Plato::PathDependentScalarFunctionFactory<Plato::InfinitesimalStrainThermoPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::PathDependentScalarFunctionFactory<Plato::InfinitesimalStrainPlasticity<3>>;
template class Plato::PathDependentScalarFunctionFactory<Plato::InfinitesimalStrainThermoPlasticity<3>>;
#endif


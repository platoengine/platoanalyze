/*
 * FluidsWeightedScalarFunction.cpp
 *
 *  Created on: Apr 7, 2021
 */

#include "FluidsWeightedScalarFunction.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Fluids::WeightedScalarFunction<Plato::IncompressibleFluids<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Fluids::WeightedScalarFunction<Plato::IncompressibleFluids<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Fluids::WeightedScalarFunction<Plato::IncompressibleFluids<3>>;
#endif

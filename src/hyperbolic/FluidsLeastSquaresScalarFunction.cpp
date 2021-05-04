/*
 * FluidsLeastSquaresScalarFunction.cpp
 *
 *  Created on: Apr 7, 2021
 */

#include "FluidsLeastSquaresScalarFunction.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Fluids::LeastSquaresScalarFunction<Plato::IncompressibleFluids<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Fluids::LeastSquaresScalarFunction<Plato::IncompressibleFluids<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Fluids::LeastSquaresScalarFunction<Plato::IncompressibleFluids<3>>;
#endif

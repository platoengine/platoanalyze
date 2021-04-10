/*
 * FluidsQuasiImplicit.cpp
 *
 *  Created on: Apr 10, 2021
 */

#include "hyperbolic/FluidsQuasiImplicit.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<3>>;
#endif

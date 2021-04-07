/*
 * FluidsCriterionFactory.cpp
 *
 *  Created on: Apr 7, 2021
 */

#include "FluidsCriterionFactory.hpp"
#include "FluidsCriterionFactory_def.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::FluidsCriterionFactory<Plato::IncompressibleFluids<1>>;
template class Plato::FluidsCriterionFactory<Plato::IncompressibleFluids<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::FluidsCriterionFactory<Plato::IncompressibleFluids<2>>;
template class Plato::FluidsCriterionFactory<Plato::IncompressibleFluids<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::FluidsCriterionFactory<Plato::IncompressibleFluids<3>>;
template class Plato::FluidsCriterionFactory<Plato::IncompressibleFluids<3>>;
#endif


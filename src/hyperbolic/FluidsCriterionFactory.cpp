/*
 * FluidsCriterionFactory.cpp
 *
 *  Created on: Apr 7, 2021
 */

#include "FluidsCriterionFactory.hpp"
#include "FluidsCriterionFactory_def.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Fluids::CriterionFactory<Plato::IncompressibleFluids<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Fluids::CriterionFactory<Plato::IncompressibleFluids<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Fluids::CriterionFactory<Plato::IncompressibleFluids<3>>;
#endif


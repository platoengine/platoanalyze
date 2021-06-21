/*
 * FluidsThermalSources.cpp
 *
 *  Created on: June 17, 2021
 */

#include "hyperbolic/FluidsThermalSources.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::ThermalSources, Plato::EnergyConservation, Plato::SimplexFluids, 1, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::ThermalSources, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::ThermalSources, Plato::EnergyConservation, Plato::SimplexFluids, 2, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::ThermalSources, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::ThermalSources, Plato::EnergyConservation, Plato::SimplexFluids, 3, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::ThermalSources, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif

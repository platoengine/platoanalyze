/*
 * AbstractVolumetricSource.cpp
 *
 *  Created on: June 16, 2021
 */

#include "AbstractVolumetricSource.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF_FLUIDS(Plato::AbstractVolumetricSource, Plato::EnergyConservation, Plato::SimplexFluids, 1, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::AbstractVolumetricSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_FLUIDS(Plato::AbstractVolumetricSource, Plato::EnergyConservation, Plato::SimplexFluids, 2, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::AbstractVolumetricSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_FLUIDS(Plato::AbstractVolumetricSource, Plato::EnergyConservation, Plato::SimplexFluids, 3, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::AbstractVolumetricSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif

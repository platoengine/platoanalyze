/*
 * AbstractScalarFunction.cpp
 *
 *  Created on: Apr 8, 2021
 */

#include "hyperbolic/AbstractScalarFunction.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::AbstractScalarFunction, Plato::MassConservation, Plato::SimplexFluids, 1, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::AbstractScalarFunction, Plato::EnergyConservation, Plato::SimplexFluids, 1, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::AbstractScalarFunction, Plato::MomentumConservation, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::AbstractScalarFunction, Plato::MassConservation, Plato::SimplexFluids, 2, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::AbstractScalarFunction, Plato::EnergyConservation, Plato::SimplexFluids, 2, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::AbstractScalarFunction, Plato::MomentumConservation, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::AbstractScalarFunction, Plato::MassConservation, Plato::SimplexFluids, 3, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::AbstractScalarFunction, Plato::EnergyConservation, Plato::SimplexFluids, 3, 1)
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::AbstractScalarFunction, Plato::MomentumConservation, Plato::SimplexFluids, 3, 1)
#endif

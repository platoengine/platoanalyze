/*
 * PressureResidual.cpp
 *
 *  Created on: Apr 7, 2021
 */

#include "hyperbolic/PressureResidual.hpp"

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::PressureResidual, Plato::MassConservation, Plato::SimplexFluids, 2, 1)
#endif

/*
 * PressureResidual.cpp
 *
 *  Created on: Apr 7, 2021
 */

#include "hyperbolic/PressureResidual.hpp"

#ifdef PLATOANALYZE_2D
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2,1>,Plato::Fluids::ResultTypes<Plato::SimplexFluids<2,1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2,1>,Plato::Fluids::GradConfigTypes<Plato::SimplexFluids<2,1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2,1>,Plato::Fluids::GradControlTypes<Plato::SimplexFluids<2,1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2,1>,Plato::Fluids::GradCurrentMassTypes<Plato::SimplexFluids<2,1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2,1>,Plato::Fluids::GradCurrentEnergyTypes<Plato::SimplexFluids<2,1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2,1>,Plato::Fluids::GradCurrentMomentumTypes<Plato::SimplexFluids<2,1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2,1>,Plato::Fluids::GradPreviousMassTypes<Plato::SimplexFluids<2,1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2,1>,Plato::Fluids::GradPreviousEnergyTypes<Plato::SimplexFluids<2,1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2,1>,Plato::Fluids::GradPreviousMomentumTypes<Plato::SimplexFluids<2,1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2,1>,Plato::Fluids::GradMomentumPredictorTypes<Plato::SimplexFluids<2,1>>>;
#endif

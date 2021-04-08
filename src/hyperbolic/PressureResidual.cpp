/*
 * PressureResidual.cpp
 *
 *  Created on: Apr 7, 2021
 */

#include "PressureResidual.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Fluids::PressureResidual<Plato::MassConservation<1>,Plato::Fluids::ResultTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<1>,Plato::Fluids::GradConfigTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<1>,Plato::Fluids::GradControlTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<1>,Plato::Fluids::GradCurrentMassTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<1>,Plato::Fluids::GradCurrentEnergyTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<1>,Plato::Fluids::GradCurrentMomentumTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<1>,Plato::Fluids::GradPreviousMassTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<1>,Plato::Fluids::GradPreviousEnergyTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<1>,Plato::Fluids::GradPreviousMomentumTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<1>,Plato::Fluids::GradMomentumPredictorTypes<Plato::SimplexFluids<1>>>;

template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<1>,Plato::Fluids::ResultTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<1>,Plato::Fluids::GradConfigTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<1>,Plato::Fluids::GradControlTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<1>,Plato::Fluids::GradCurrentMassTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<1>,Plato::Fluids::GradCurrentEnergyTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<1>,Plato::Fluids::GradCurrentMomentumTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<1>,Plato::Fluids::GradPreviousMassTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<1>,Plato::Fluids::GradPreviousEnergyTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<1>,Plato::Fluids::GradPreviousMomentumTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<1>,Plato::Fluids::GradMomentumPredictorTypes<Plato::SimplexFluids<1>>>;

template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<1>,Plato::Fluids::ResultTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<1>,Plato::Fluids::GradConfigTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<1>,Plato::Fluids::GradControlTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<1>,Plato::Fluids::GradCurrentMassTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<1>,Plato::Fluids::GradCurrentEnergyTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<1>,Plato::Fluids::GradCurrentMomentumTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<1>,Plato::Fluids::GradPreviousMassTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<1>,Plato::Fluids::GradPreviousEnergyTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<1>,Plato::Fluids::GradPreviousMomentumTypes<Plato::SimplexFluids<1>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<1>,Plato::Fluids::GradMomentumPredictorTypes<Plato::SimplexFluids<1>>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2>,Plato::Fluids::ResultTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2>,Plato::Fluids::GradConfigTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2>,Plato::Fluids::GradControlTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2>,Plato::Fluids::GradCurrentMassTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2>,Plato::Fluids::GradCurrentEnergyTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2>,Plato::Fluids::GradCurrentMomentumTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2>,Plato::Fluids::GradPreviousMassTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2>,Plato::Fluids::GradPreviousEnergyTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2>,Plato::Fluids::GradPreviousMomentumTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<2>,Plato::Fluids::GradMomentumPredictorTypes<Plato::SimplexFluids<2>>>;

template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<2>,Plato::Fluids::ResultTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<2>,Plato::Fluids::GradConfigTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<2>,Plato::Fluids::GradControlTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<2>,Plato::Fluids::GradCurrentMassTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<2>,Plato::Fluids::GradCurrentEnergyTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<2>,Plato::Fluids::GradCurrentMomentumTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<2>,Plato::Fluids::GradPreviousMassTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<2>,Plato::Fluids::GradPreviousEnergyTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<2>,Plato::Fluids::GradPreviousMomentumTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<2>,Plato::Fluids::GradMomentumPredictorTypes<Plato::SimplexFluids<2>>>;

template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<2>,Plato::Fluids::ResultTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<2>,Plato::Fluids::GradConfigTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<2>,Plato::Fluids::GradControlTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<2>,Plato::Fluids::GradCurrentMassTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<2>,Plato::Fluids::GradCurrentEnergyTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<2>,Plato::Fluids::GradCurrentMomentumTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<2>,Plato::Fluids::GradPreviousMassTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<2>,Plato::Fluids::GradPreviousEnergyTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<2>,Plato::Fluids::GradPreviousMomentumTypes<Plato::SimplexFluids<2>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<2>,Plato::Fluids::GradMomentumPredictorTypes<Plato::SimplexFluids<2>>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Fluids::PressureResidual<Plato::MassConservation<3>,Plato::Fluids::ResultTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<3>,Plato::Fluids::GradConfigTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<3>,Plato::Fluids::GradControlTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<3>,Plato::Fluids::GradCurrentMassTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<3>,Plato::Fluids::GradCurrentEnergyTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<3>,Plato::Fluids::GradCurrentMomentumTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<3>,Plato::Fluids::GradPreviousMassTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<3>,Plato::Fluids::GradPreviousEnergyTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<3>,Plato::Fluids::GradPreviousMomentumTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MassConservation<3>,Plato::Fluids::GradMomentumPredictorTypes<Plato::SimplexFluids<3>>>;

template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<3>,Plato::Fluids::ResultTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<3>,Plato::Fluids::GradConfigTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<3>,Plato::Fluids::GradControlTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<3>,Plato::Fluids::GradCurrentMassTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<3>,Plato::Fluids::GradCurrentEnergyTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<3>,Plato::Fluids::GradCurrentMomentumTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<3>,Plato::Fluids::GradPreviousMassTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<3>,Plato::Fluids::GradPreviousEnergyTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<3>,Plato::Fluids::GradPreviousMomentumTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::EnergyConservation<3>,Plato::Fluids::GradMomentumPredictorTypes<Plato::SimplexFluids<3>>>;

template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<3>,Plato::Fluids::ResultTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<3>,Plato::Fluids::GradConfigTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<3>,Plato::Fluids::GradControlTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<3>,Plato::Fluids::GradCurrentMassTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<3>,Plato::Fluids::GradCurrentEnergyTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<3>,Plato::Fluids::GradCurrentMomentumTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<3>,Plato::Fluids::GradPreviousMassTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<3>,Plato::Fluids::GradPreviousEnergyTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<3>,Plato::Fluids::GradPreviousMomentumTypes<Plato::SimplexFluids<3>>>;
template class Plato::Fluids::PressureResidual<Plato::MomentumConservation<3>,Plato::Fluids::GradMomentumPredictorTypes<Plato::SimplexFluids<3>>>;
#endif

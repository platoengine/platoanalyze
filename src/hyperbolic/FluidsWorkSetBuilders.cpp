/*
 * FluidsWorkSetBuilders.cpp
 *
 *  Created on: Apr 8, 2021
 */

#include "hyperbolic/FluidsWorkSetBuilders.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Fluids::WorkSetBuilder<Plato::MassConservation<1>>;
template class Plato::Fluids::WorkSetBuilder<Plato::EnergyConservation<1>>;
template class Plato::Fluids::WorkSetBuilder<Plato::MomentumConservation<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Fluids::WorkSetBuilder<Plato::MassConservation<2>>;
template class Plato::Fluids::WorkSetBuilder<Plato::EnergyConservation<2>>;
template class Plato::Fluids::WorkSetBuilder<Plato::MomentumConservation<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Fluids::WorkSetBuilder<Plato::MassConservation<3>>;
template class Plato::Fluids::WorkSetBuilder<Plato::EnergyConservation<3>>;
template class Plato::Fluids::WorkSetBuilder<Plato::MomentumConservation<3>>;
#endif

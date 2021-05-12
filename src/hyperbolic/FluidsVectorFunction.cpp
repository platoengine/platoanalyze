/*
 * FluidsVectorFunction.cpp
 *
 *  Created on: Apr 7, 2021
 */

#include "hyperbolic/FluidsVectorFunction.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Fluids::VectorFunction<Plato::MassConservation<1>>;
template class Plato::Fluids::VectorFunction<Plato::EnergyConservation<1>>;
template class Plato::Fluids::VectorFunction<Plato::MomentumConservation<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Fluids::VectorFunction<Plato::MassConservation<2>>;
template class Plato::Fluids::VectorFunction<Plato::EnergyConservation<2>>;
template class Plato::Fluids::VectorFunction<Plato::MomentumConservation<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Fluids::VectorFunction<Plato::MassConservation<3>>;
template class Plato::Fluids::VectorFunction<Plato::EnergyConservation<3>>;
template class Plato::Fluids::VectorFunction<Plato::MomentumConservation<3>>;
#endif

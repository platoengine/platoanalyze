/*
 * TensileEnergyDensity.cpp
 *
 */

#include "TensileEnergyDensity.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::TensileEnergyDensity<1>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::TensileEnergyDensity<2>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::TensileEnergyDensity<3>;
#endif
/*
 * GlobalVectorFunctionInc.cpp
 *
 *  Created on: Mar 1, 2020
 */

#include "GlobalVectorFunctionInc.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::GlobalVectorFunctionInc<Plato::InfinitesimalStrainPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::GlobalVectorFunctionInc<Plato::InfinitesimalStrainPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::GlobalVectorFunctionInc<Plato::InfinitesimalStrainPlasticity<3>>;
#endif


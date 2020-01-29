/*
 * ThermalVonMisesLocalMeasure.cpp
 *
 */

#include "ThermalVonMisesLocalMeasure.hpp"


#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF2(Plato::ThermalVonMisesLocalMeasure, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF2(Plato::ThermalVonMisesLocalMeasure, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF2(Plato::ThermalVonMisesLocalMeasure, Plato::SimplexThermomechanics, 3)
#endif
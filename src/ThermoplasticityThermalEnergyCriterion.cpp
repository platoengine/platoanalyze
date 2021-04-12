/*
 * ThermoplasticityThermalEnergyCriterion.cpp
 *
 *  Created on: Mar 22, 2021
 */

#include "ThermoplasticityThermalEnergyCriterion.hpp"

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_INC_VMS(Plato::ThermoplasticityThermalEnergyCriterion, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_INC_VMS(Plato::ThermoplasticityThermalEnergyCriterion, Plato::SimplexThermoPlasticity, 3)
#endif

/*
 * TotalWorkCriterion.cpp
 *
 *  Created on: Mar 22, 2021
 */

#include "TotalWorkCriterion.hpp"

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_INC_VMS(Plato::TotalWorkCriterion, Plato::SimplexPlasticity, 2)
PLATO_EXPL_DEF_INC_VMS(Plato::TotalWorkCriterion, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_INC_VMS(Plato::TotalWorkCriterion, Plato::SimplexPlasticity, 3)
PLATO_EXPL_DEF_INC_VMS(Plato::TotalWorkCriterion, Plato::SimplexThermoPlasticity, 3)
#endif

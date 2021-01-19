/*
 * PlasticWorkCriterion.cpp
 *
 *  Created on: Mar 3, 2020
 */

#include "PlasticWorkCriterion.hpp"

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_INC_VMS(Plato::PlasticWorkCriterion, Plato::SimplexPlasticity, 2)
PLATO_EXPL_DEF_INC_VMS(Plato::PlasticWorkCriterion, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_INC_VMS(Plato::PlasticWorkCriterion, Plato::SimplexPlasticity, 3)
PLATO_EXPL_DEF_INC_VMS(Plato::PlasticWorkCriterion, Plato::SimplexThermoPlasticity, 3)
#endif

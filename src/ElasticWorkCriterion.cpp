/*
 * ElasticWorkCriterion.cpp
 *
 *  Created on: Mar 7, 2020
 */

#include "ElasticWorkCriterion.hpp"

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_INC_VMS(Plato::ElasticWorkCriterion, Plato::SimplexPlasticity, 2)
PLATO_EXPL_DEF_INC_VMS(Plato::ElasticWorkCriterion, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_INC_VMS(Plato::ElasticWorkCriterion, Plato::SimplexPlasticity, 3)
PLATO_EXPL_DEF_INC_VMS(Plato::ElasticWorkCriterion, Plato::SimplexThermoPlasticity, 3)
#endif

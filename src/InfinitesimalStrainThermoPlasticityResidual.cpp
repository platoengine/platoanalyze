/*
 * InfinitesimalStrainThermoPlasticityResidual.cpp
 * 
 * Created on: Jan 20, 2021
 */

#include "InfinitesimalStrainThermoPlasticityResidual.hpp"

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_INC_VMS(Plato::InfinitesimalStrainThermoPlasticityResidual, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_INC_VMS(Plato::InfinitesimalStrainThermoPlasticityResidual, Plato::SimplexThermoPlasticity, 3)
#endif

/*
 * InfinitesimalStrainPlasticity.cpp
 * 
 * Created on: Mar 3, 2020
 */

#include "InfinitesimalStrainPlasticity.hpp"

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_INC_VMS(Plato::InfinitesimalStrainPlasticityResidual, Plato::SimplexPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_INC_VMS(Plato::InfinitesimalStrainPlasticityResidual, Plato::SimplexPlasticity, 3)
#endif

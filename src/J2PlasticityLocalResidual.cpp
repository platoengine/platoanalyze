#include "J2PlasticityLocalResidual.hpp"

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_INC_LOCAL_2(Plato::J2PlasticityLocalResidual, Plato::SimplexPlasticity, 2)
PLATO_EXPL_DEF_INC_LOCAL_2(Plato::J2PlasticityLocalResidual, Plato::SimplexThermoPlasticity, 2)
#endif
#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_INC_LOCAL_2(Plato::J2PlasticityLocalResidual, Plato::SimplexPlasticity, 3)
PLATO_EXPL_DEF_INC_LOCAL_2(Plato::J2PlasticityLocalResidual, Plato::SimplexThermoPlasticity, 3)
#endif

#include "EMStressPNorm.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATO_1D
PLATO_EXPL_DEF(Plato::EMStressPNorm, Plato::SimplexElectromechanics, 1)
#endif

#ifdef PLATO_2D
PLATO_EXPL_DEF(Plato::EMStressPNorm, Plato::SimplexElectromechanics, 2)
#endif

#ifdef PLATO_3D
PLATO_EXPL_DEF(Plato::EMStressPNorm, Plato::SimplexElectromechanics, 3)
#endif

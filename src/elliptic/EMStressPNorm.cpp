#include "elliptic/EMStressPNorm.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF(Plato::Elliptic::EMStressPNorm, Plato::SimplexElectromechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF(Plato::Elliptic::EMStressPNorm, Plato::SimplexElectromechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF(Plato::Elliptic::EMStressPNorm, Plato::SimplexElectromechanics, 3)
#endif

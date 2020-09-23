#include "elliptic/TMStressPNorm.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF(Plato::Elliptic::TMStressPNorm, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF(Plato::Elliptic::TMStressPNorm, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF(Plato::Elliptic::TMStressPNorm, Plato::SimplexThermomechanics, 3)
#endif

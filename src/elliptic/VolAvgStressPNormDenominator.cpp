#include "elliptic/VolAvgStressPNormDenominator.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF(Plato::Elliptic::VolAvgStressPNormDenominator, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF(Plato::Elliptic::VolAvgStressPNormDenominator, Plato::SimplexMechanics, 3)
#endif

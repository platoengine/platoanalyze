#include "StabilizedElastostaticEnergy.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 3)
#endif

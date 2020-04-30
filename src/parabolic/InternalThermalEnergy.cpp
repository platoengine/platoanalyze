#include "InternalThermalEnergy.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF(Plato::Parabolic::InternalThermalEnergy, Plato::SimplexThermal, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF(Plato::Parabolic::InternalThermalEnergy, Plato::SimplexThermal, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF(Plato::Parabolic::InternalThermalEnergy, Plato::SimplexThermal, 3)
#endif

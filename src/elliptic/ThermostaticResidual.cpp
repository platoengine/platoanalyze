#include "elliptic/ThermostaticResidual.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF(Plato::Elliptic::ThermostaticResidual, Plato::SimplexThermal, 1)
#endif
#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF(Plato::Elliptic::ThermostaticResidual, Plato::SimplexThermal, 2)
#endif
#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF(Plato::Elliptic::ThermostaticResidual, Plato::SimplexThermal, 3)
#endif

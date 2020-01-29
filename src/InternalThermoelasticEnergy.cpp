#include "InternalThermoelasticEnergy.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF(Plato::InternalThermoelasticEnergy,    Plato::SimplexThermomechanics, 1)
PLATO_EXPL_DEF(Plato::InternalThermoelasticEnergyInc, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF(Plato::InternalThermoelasticEnergy,    Plato::SimplexThermomechanics, 2)
PLATO_EXPL_DEF(Plato::InternalThermoelasticEnergyInc, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF(Plato::InternalThermoelasticEnergy,    Plato::SimplexThermomechanics, 3)
PLATO_EXPL_DEF(Plato::InternalThermoelasticEnergyInc, Plato::SimplexThermomechanics, 3)
#endif

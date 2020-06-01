#include "parabolic/InternalThermoelasticEnergy.hpp"
#include "parabolic/ExpInstMacros.hpp"

#ifdef PLATOANALYZE_1D
PLATO_PARABOLIC_EXPL_DEF(Plato::Parabolic::InternalThermoelasticEnergy, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_PARABOLIC_EXPL_DEF(Plato::Parabolic::InternalThermoelasticEnergy, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_PARABOLIC_EXPL_DEF(Plato::Parabolic::InternalThermoelasticEnergy, Plato::SimplexThermomechanics, 3)
#endif

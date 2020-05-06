#include "parabolic/TransientThermomechResidual.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF_INC(Plato::Parabolic::TransientThermomechResidual, Plato::SimplexThermomechanics, 1)
#endif
#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_INC(Plato::Parabolic::TransientThermomechResidual, Plato::SimplexThermomechanics, 2)
#endif
#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_INC(Plato::Parabolic::TransientThermomechResidual, Plato::SimplexThermomechanics, 3)
#endif

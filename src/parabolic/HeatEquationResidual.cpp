#include "parabolic/HeatEquationResidual.hpp"
#include "parabolic/ExpInstMacros.hpp"

#ifdef PLATOANALYZE_1D
PLATO_PARABOLIC_EXPL_DEF(Plato::Parabolic::HeatEquationResidual, Plato::SimplexThermal, 1)
#endif
#ifdef PLATOANALYZE_2D
PLATO_PARABOLIC_EXPL_DEF(Plato::Parabolic::HeatEquationResidual, Plato::SimplexThermal, 2)
#endif
#ifdef PLATOANALYZE_3D
PLATO_PARABOLIC_EXPL_DEF(Plato::Parabolic::HeatEquationResidual, Plato::SimplexThermal, 3)
#endif

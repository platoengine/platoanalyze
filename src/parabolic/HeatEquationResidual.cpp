#include "parabolic/HeatEquationResidual.hpp"
#include "ExpInstMacros.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF_INC(Plato::Parabolic::HeatEquationResidual, Plato::SimplexThermal, 1)
#endif
#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_INC(Plato::Parabolic::HeatEquationResidual, Plato::SimplexThermal, 2)
#endif
#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_INC(Plato::Parabolic::HeatEquationResidual, Plato::SimplexThermal, 3)
#endif

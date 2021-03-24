#include "YieldStressExpression.hpp"

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_INC_LOCAL_1(Plato::YieldStressExpression, Plato::SimplexPlasticity,       2)
PLATO_EXPL_DEF_INC_LOCAL_1(Plato::YieldStressExpression, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_INC_LOCAL_1(Plato::YieldStressExpression, Plato::SimplexPlasticity,       3)
PLATO_EXPL_DEF_INC_LOCAL_1(Plato::YieldStressExpression, Plato::SimplexThermoPlasticity, 3)
#endif

/*
 * LinearStressExpression.cpp
 *
 */

#include "LinearStressExpression.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF2(Plato::LinearStressExpression, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF2(Plato::LinearStressExpression, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF2(Plato::LinearStressExpression, Plato::SimplexMechanics, 3)
#endif

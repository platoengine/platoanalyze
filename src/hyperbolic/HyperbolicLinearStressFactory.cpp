/*
 * HyperbolicLinearStressFactory.cpp
 *
 */

#include "HyperbolicLinearStressFactory.hpp"

#ifdef PLATOANALYZE_1D
PLATO_HYPERBOLIC_EXPL_DEF2(Plato::Hyperbolic::HyperbolicLinearStressFactory, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_HYPERBOLIC_EXPL_DEF2(Plato::Hyperbolic::HyperbolicLinearStressFactory, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_HYPERBOLIC_EXPL_DEF2(Plato::Hyperbolic::HyperbolicLinearStressFactory, Plato::SimplexMechanics, 3)
#endif

/*
 * EllipticUpLagLinearStressExpression.cpp
 *
 */

#include "EllipticUpLagLinearStressExpression.hpp"

#ifdef PLATOANALYZE_1D
PLATO_ELLIPTIC_UPLAG_EXPL_DEF2(Plato::Elliptic::UpdatedLagrangian::EllipticUpLagLinearStressExpression, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_ELLIPTIC_UPLAG_EXPL_DEF2(Plato::Elliptic::UpdatedLagrangian::EllipticUpLagLinearStressExpression, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_ELLIPTIC_UPLAG_EXPL_DEF2(Plato::Elliptic::UpdatedLagrangian::EllipticUpLagLinearStressExpression, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 3)
#endif

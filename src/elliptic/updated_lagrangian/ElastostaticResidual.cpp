#include "elliptic/updated_lagrangian/ElastostaticResidual.hpp"
#include "elliptic/updated_lagrangian/ExpInstMacros.hpp"

#ifdef PLATOANALYZE_1D
PLATO_ELLIPTIC_UPLAG_EXPL_DEF(Plato::Elliptic::UpdatedLagrangian::ElastostaticResidual, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 1)
#endif
#ifdef PLATOANALYZE_2D
PLATO_ELLIPTIC_UPLAG_EXPL_DEF(Plato::Elliptic::UpdatedLagrangian::ElastostaticResidual, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 2)
#endif
#ifdef PLATOANALYZE_3D
PLATO_ELLIPTIC_UPLAG_EXPL_DEF(Plato::Elliptic::UpdatedLagrangian::ElastostaticResidual, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 3)
#endif

#include "helmholtz/HelmholtzResidual.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Helmholtz::HelmholtzResidual<Plato::ResidualTypes<Plato::SimplexHelmholtz<1>>>;
template class Plato::Helmholtz::HelmholtzResidual<Plato::JacobianTypes<Plato::SimplexHelmholtz<1>>>;
#endif
#ifdef PLATOANALYZE_2D
template class Plato::Helmholtz::HelmholtzResidual<Plato::ResidualTypes<Plato::SimplexHelmholtz<2>>>;
template class Plato::Helmholtz::HelmholtzResidual<Plato::JacobianTypes<Plato::SimplexHelmholtz<2>>>;
#endif
#ifdef PLATOANALYZE_3D
template class Plato::Helmholtz::HelmholtzResidual<Plato::ResidualTypes<Plato::SimplexHelmholtz<3>>>;
template class Plato::Helmholtz::HelmholtzResidual<Plato::JacobianTypes<Plato::SimplexHelmholtz<3>>>;
#endif

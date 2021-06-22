#include "helmholtz/Problem.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<1>>;
#endif
#ifdef PLATOANALYZE_2D
template class Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<2>>;
#endif
#ifdef PLATOANALYZE_3D
template class Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<3>>;
#endif

#include "elliptic/updated_lagrangian/Problem.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Elliptic::UpdatedLagrangian::Problem<::Plato::Elliptic::UpdatedLagrangian::Mechanics<1>>;
#endif
#ifdef PLATOANALYZE_2D
template class Plato::Elliptic::UpdatedLagrangian::Problem<::Plato::Elliptic::UpdatedLagrangian::Mechanics<2>>;
#endif
#ifdef PLATOANALYZE_3D
template class Plato::Elliptic::UpdatedLagrangian::Problem<::Plato::Elliptic::UpdatedLagrangian::Mechanics<3>>;
#endif

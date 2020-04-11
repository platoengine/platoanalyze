#include "ParabolicProblem.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::ParabolicProblem<::Plato::Thermal<1>>;
#endif
#ifdef PLATOANALYZE_2D
template class Plato::ParabolicProblem<::Plato::Thermal<2>>;
#endif
#ifdef PLATOANALYZE_3D
template class Plato::ParabolicProblem<::Plato::Thermal<3>>;
#endif

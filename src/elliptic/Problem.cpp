#include "elliptic/Problem.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Elliptic::Problem<::Plato::Thermal<1>>;
template class Plato::Elliptic::Problem<::Plato::Mechanics<1>>;
//template class Plato::Elliptic::Problem<::Plato::Electromechanics<1>>;
//template class Plato::Elliptic::Problem<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATOANALYZE_2D
template class Plato::Elliptic::Problem<::Plato::Thermal<2>>;
template class Plato::Elliptic::Problem<::Plato::Mechanics<2>>;
//template class Plato::Elliptic::Problem<::Plato::Electromechanics<2>>;
//template class Plato::Elliptic::Problem<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATOANALYZE_3D
template class Plato::Elliptic::Problem<::Plato::Thermal<3>>;
template class Plato::Elliptic::Problem<::Plato::Mechanics<3>>;
//template class Plato::Elliptic::Problem<::Plato::Electromechanics<3>>;
//template class Plato::Elliptic::Problem<::Plato::Thermomechanics<3>>;
#endif

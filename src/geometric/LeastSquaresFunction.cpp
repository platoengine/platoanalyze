#include "geometric/LeastSquaresFunction.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Geometric::LeastSquaresFunction<::Plato::Geometrical<1>>;
#endif
#ifdef PLATOANALYZE_2D
template class Plato::Geometric::LeastSquaresFunction<::Plato::Geometrical<2>>;
#endif
#ifdef PLATOANALYZE_3D
template class Plato::Geometric::LeastSquaresFunction<::Plato::Geometrical<3>>;
#endif

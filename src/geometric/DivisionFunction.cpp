#include "geometric/DivisionFunction.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Geometric::DivisionFunction<::Plato::Geometrical<1>>;
#endif
#ifdef PLATOANALYZE_2D
template class Plato::Geometric::DivisionFunction<::Plato::Geometrical<2>>;
#endif
#ifdef PLATOANALYZE_3D
template class Plato::Geometric::DivisionFunction<::Plato::Geometrical<3>>;
#endif

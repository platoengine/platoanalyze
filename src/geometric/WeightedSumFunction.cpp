#include "geometric/WeightedSumFunction.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Geometric::WeightedSumFunction<::Plato::Geometrical<1>>;
#endif
#ifdef PLATOANALYZE_2D
template class Plato::Geometric::WeightedSumFunction<::Plato::Geometrical<2>>;
#endif
#ifdef PLATOANALYZE_3D
template class Plato::Geometric::WeightedSumFunction<::Plato::Geometrical<3>>;
#endif

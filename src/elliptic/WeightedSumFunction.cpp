#include "elliptic/WeightedSumFunction.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Elliptic::WeightedSumFunction<::Plato::Thermal<1>>;
template class Plato::Elliptic::WeightedSumFunction<::Plato::Mechanics<1>>;
template class Plato::Elliptic::WeightedSumFunction<::Plato::Electromechanics<1>>;
template class Plato::Elliptic::WeightedSumFunction<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATOANALYZE_2D
template class Plato::Elliptic::WeightedSumFunction<::Plato::Thermal<2>>;
template class Plato::Elliptic::WeightedSumFunction<::Plato::Mechanics<2>>;
template class Plato::Elliptic::WeightedSumFunction<::Plato::Electromechanics<2>>;
template class Plato::Elliptic::WeightedSumFunction<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATOANALYZE_3D
template class Plato::Elliptic::WeightedSumFunction<::Plato::Thermal<3>>;
template class Plato::Elliptic::WeightedSumFunction<::Plato::Mechanics<3>>;
template class Plato::Elliptic::WeightedSumFunction<::Plato::Electromechanics<3>>;
template class Plato::Elliptic::WeightedSumFunction<::Plato::Thermomechanics<3>>;
#endif

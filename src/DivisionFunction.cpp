#include "DivisionFunction.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::DivisionFunction<::Plato::Thermal<1>>;
template class Plato::DivisionFunction<::Plato::Mechanics<1>>;
template class Plato::DivisionFunction<::Plato::Electromechanics<1>>;
template class Plato::DivisionFunction<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATOANALYZE_2D
template class Plato::DivisionFunction<::Plato::Thermal<2>>;
template class Plato::DivisionFunction<::Plato::Mechanics<2>>;
template class Plato::DivisionFunction<::Plato::Electromechanics<2>>;
template class Plato::DivisionFunction<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATOANALYZE_3D
template class Plato::DivisionFunction<::Plato::Thermal<3>>;
template class Plato::DivisionFunction<::Plato::Mechanics<3>>;
template class Plato::DivisionFunction<::Plato::Electromechanics<3>>;
template class Plato::DivisionFunction<::Plato::Thermomechanics<3>>;
#endif

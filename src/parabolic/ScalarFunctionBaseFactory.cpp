#include "parabolic/ScalarFunctionBaseFactory.hpp"
#include "parabolic/ScalarFunctionBaseFactory_def.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::Thermal<1>>;
template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::Thermal<2>>;
template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::Thermal<3>>;
template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<3>>;
#endif

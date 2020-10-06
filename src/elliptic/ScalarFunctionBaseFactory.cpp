#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include "elliptic/ScalarFunctionBaseFactory_def.hpp"


#ifdef PLATOANALYZE_1D
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermal<1>>;
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Mechanics<1>>;
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Electromechanics<1>>;
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<1>>;
#ifdef PLATO_STABILIZED
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<1>>;
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<1>>;
#endif
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermal<2>>;
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Mechanics<2>>;
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Electromechanics<2>>;
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<2>>;
#ifdef PLATO_STABILIZED
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<2>>;
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<2>>;
#endif
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermal<3>>;
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Mechanics<3>>;
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Electromechanics<3>>;
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<3>>;
#ifdef PLATO_STABILIZED
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<3>>;
template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<3>>;
#endif
#endif

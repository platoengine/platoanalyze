#include "elliptic/updated_lagrangian/ScalarFunctionBaseFactory.hpp"
#include "elliptic/updated_lagrangian/ScalarFunctionBaseFactory_def.hpp"


#ifdef PLATOANALYZE_1D
template class Plato::Elliptic::UpdatedLagrangian::ScalarFunctionBaseFactory<::Plato::Elliptic::UpdatedLagrangian::Mechanics<1>>;
// TODO template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Electromechanics<1>>;
// TODO template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<1>>;
#ifdef PLATO_STABILIZED
// TODO template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<1>>;
// TODO template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<1>>;
#endif
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Elliptic::UpdatedLagrangian::ScalarFunctionBaseFactory<::Plato::Elliptic::UpdatedLagrangian::Mechanics<2>>;
// TODO template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Electromechanics<2>>;
// TODO template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<2>>;
#ifdef PLATO_STABILIZED
// TODO template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<2>>;
// TODO template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<2>>;
#endif
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Elliptic::UpdatedLagrangian::ScalarFunctionBaseFactory<::Plato::Elliptic::UpdatedLagrangian::Mechanics<3>>;
// TODO template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Electromechanics<3>>;
// TODO template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<3>>;
#ifdef PLATO_STABILIZED
// TODO template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<3>>;
// TODO template class Plato::Elliptic::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<3>>;
#endif
#endif

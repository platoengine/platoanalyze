#include "hyperbolic/HyperbolicScalarFunctionFactory.hpp"
#include "hyperbolic/HyperbolicScalarFunctionFactory_def.hpp"


#ifdef PLATOANALYZE_1D
template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::Mechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::Mechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::Mechanics<3>>;
#endif

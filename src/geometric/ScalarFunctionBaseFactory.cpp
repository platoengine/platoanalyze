#include "geometric/ScalarFunctionBaseFactory.hpp"
#include "geometric/ScalarFunctionBaseFactory_def.hpp"


#ifdef PLATOANALYZE_1D
template class Plato::Geometric::ScalarFunctionBaseFactory<::Plato::Geometrical<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Geometric::ScalarFunctionBaseFactory<::Plato::Geometrical<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Geometric::ScalarFunctionBaseFactory<::Plato::Geometrical<3>>;
#endif

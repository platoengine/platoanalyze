#include "PhysicsScalarFunctionInc.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::PhysicsScalarFunctionInc<::Plato::Thermal<1>>;
template class Plato::PhysicsScalarFunctionInc<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATOANALYZE_2D
template class Plato::PhysicsScalarFunctionInc<::Plato::Thermal<2>>;
template class Plato::PhysicsScalarFunctionInc<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATOANALYZE_3D
template class Plato::PhysicsScalarFunctionInc<::Plato::Thermal<3>>;
template class Plato::PhysicsScalarFunctionInc<::Plato::Thermomechanics<3>>;
#endif

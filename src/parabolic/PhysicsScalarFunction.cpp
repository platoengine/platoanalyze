#include "parabolic/PhysicsScalarFunction.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermal<1>>;
template class Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATOANALYZE_2D
template class Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermal<2>>;
template class Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATOANALYZE_3D
template class Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermal<3>>;
template class Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermomechanics<3>>;
#endif

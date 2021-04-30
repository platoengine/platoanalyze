#include "elliptic/updated_lagrangian/PhysicsScalarFunction.hpp"

#ifdef PLATOANALYZE_1D
// TODO template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermal<1>>;
template class Plato::Elliptic::UpdatedLagrangian::PhysicsScalarFunction<::Plato::Elliptic::UpdatedLagrangian::Mechanics<1>>;
// TODO template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Electromechanics<1>>;
// TODO template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATOANALYZE_2D
// TODO template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermal<2>>;
template class Plato::Elliptic::UpdatedLagrangian::PhysicsScalarFunction<::Plato::Elliptic::UpdatedLagrangian::Mechanics<2>>;
// TODO template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Electromechanics<2>>;
// TODO template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATOANALYZE_3D
// TODO template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermal<3>>;
template class Plato::Elliptic::UpdatedLagrangian::PhysicsScalarFunction<::Plato::Elliptic::UpdatedLagrangian::Mechanics<3>>;
// TODO template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Electromechanics<3>>;
// TODO template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermomechanics<3>>;
#endif

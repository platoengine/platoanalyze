
#include "elliptic/MassMoment.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Elliptic::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
template class Plato::Elliptic::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
template class Plato::Elliptic::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
template class Plato::Elliptic::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Elliptic::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
template class Plato::Elliptic::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
template class Plato::Elliptic::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
template class Plato::Elliptic::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Elliptic::MassMoment<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
template class Plato::Elliptic::MassMoment<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
template class Plato::Elliptic::MassMoment<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
template class Plato::Elliptic::MassMoment<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif

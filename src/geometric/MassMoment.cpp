#include "geometric/MassMoment.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::Geometric::MassMoment<Plato::Geometric::ResidualTypes<Plato::Simplex<1>>>;
template class Plato::Geometric::MassMoment<Plato::Geometric::GradientXTypes<Plato::Simplex<1>>>;
template class Plato::Geometric::MassMoment<Plato::Geometric::GradientZTypes<Plato::Simplex<1>>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::Geometric::MassMoment<Plato::Geometric::ResidualTypes<Plato::Simplex<2>>>;
template class Plato::Geometric::MassMoment<Plato::Geometric::GradientXTypes<Plato::Simplex<2>>>;
template class Plato::Geometric::MassMoment<Plato::Geometric::GradientZTypes<Plato::Simplex<2>>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Geometric::MassMoment<Plato::Geometric::ResidualTypes<Plato::Simplex<3>>>;
template class Plato::Geometric::MassMoment<Plato::Geometric::GradientXTypes<Plato::Simplex<3>>>;
template class Plato::Geometric::MassMoment<Plato::Geometric::GradientZTypes<Plato::Simplex<3>>>;
#endif

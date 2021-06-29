#pragma once

#define PLATO_ELLIPTIC_UPLAG_EXPL_DEF(C, T, D) \
template class C<Plato::Elliptic::UpdatedLagrangian::ResidualTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Elliptic::UpdatedLagrangian::ResidualTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Elliptic::UpdatedLagrangian::ResidualTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::Elliptic::UpdatedLagrangian::JacobianTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Elliptic::UpdatedLagrangian::JacobianTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Elliptic::UpdatedLagrangian::JacobianTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::Elliptic::UpdatedLagrangian::GradientCTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Elliptic::UpdatedLagrangian::GradientCTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Elliptic::UpdatedLagrangian::GradientCTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::Elliptic::UpdatedLagrangian::GradientXTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Elliptic::UpdatedLagrangian::GradientXTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Elliptic::UpdatedLagrangian::GradientXTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::Elliptic::UpdatedLagrangian::GradientZTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Elliptic::UpdatedLagrangian::GradientZTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Elliptic::UpdatedLagrangian::GradientZTypes<T<D>>, Plato::Heaviside >;

#define PLATO_ELLIPTIC_UPLAG_EXPL_DEC(C, T, D) \
extern template class C<Plato::Elliptic::UpdatedLagrangian::ResidualTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::ResidualTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::ResidualTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::JacobianTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::JacobianTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::JacobianTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::GradientCTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::GradientCTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::GradientCTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::GradientXTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::GradientXTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::GradientXTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::GradientZTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::GradientZTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::GradientZTypes<T<D>>, Plato::Heaviside >;


#define PLATO_ELLIPTIC_UPLAG_EXPL_DEC2(C, T, D) \
extern template class C<Plato::Elliptic::UpdatedLagrangian::ResidualTypes<T<D>>, T<D> >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::JacobianTypes<T<D>>, T<D> >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::GradientXTypes<T<D>>, T<D> >; \
extern template class C<Plato::Elliptic::UpdatedLagrangian::GradientZTypes<T<D>>, T<D> >;

#define PLATO_ELLIPTIC_UPLAG_EXPL_DEF2(C, T, D) \
template class C<Plato::Elliptic::UpdatedLagrangian::ResidualTypes<T<D>>, T<D> >; \
template class C<Plato::Elliptic::UpdatedLagrangian::JacobianTypes<T<D>>, T<D> >; \
template class C<Plato::Elliptic::UpdatedLagrangian::GradientXTypes<T<D>>, T<D> >; \
template class C<Plato::Elliptic::UpdatedLagrangian::GradientZTypes<T<D>>, T<D> >;


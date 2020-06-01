#ifndef PARABOLIC_EXP_INST_MACROS_HPP
#define PARABOLIC_EXP_INST_MACROS_HPP

#define PLATO_PARABOLIC_EXPL_DEF(C, T, D) \
template class C<Plato::Parabolic::ResidualTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Parabolic::ResidualTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Parabolic::ResidualTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::Parabolic::GradientUTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Parabolic::GradientUTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Parabolic::GradientUTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::Parabolic::GradientVTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Parabolic::GradientVTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Parabolic::GradientVTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::Parabolic::GradientXTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Parabolic::GradientXTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Parabolic::GradientXTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::Parabolic::GradientZTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::Parabolic::GradientZTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::Parabolic::GradientZTypes<T<D>>, Plato::Heaviside >;

#define PLATO_PARABOLIC_EXPL_DEC(C, T, D) \
extern template class C<Plato::Parabolic::ResidualTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Parabolic::ResidualTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Parabolic::ResidualTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::Parabolic::GradientUTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Parabolic::GradientUTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Parabolic::GradientUTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::Parabolic::GradientVTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Parabolic::GradientVTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Parabolic::GradientVTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::Parabolic::GradientXTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Parabolic::GradientXTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Parabolic::GradientXTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::Parabolic::GradientZTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::Parabolic::GradientZTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::Parabolic::GradientZTypes<T<D>>, Plato::Heaviside >;

#endif

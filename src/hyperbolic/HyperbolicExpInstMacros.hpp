#ifndef HYPERBOLIC_EXP_INST_MACROS_HPP
#define HYPERBOLIC_EXP_INST_MACROS_HPP

#define PLATO_HYPERBOLIC_EXPL_DEC2(C, T, D) \
extern template class C<Plato::Hyperbolic::ResidualTypes <T<D>>, T<D> >; \
extern template class C<Plato::Hyperbolic::GradientUTypes<T<D>>, T<D> >; \
extern template class C<Plato::Hyperbolic::GradientVTypes<T<D>>, T<D> >; \
extern template class C<Plato::Hyperbolic::GradientATypes<T<D>>, T<D> >; \
extern template class C<Plato::Hyperbolic::GradientXTypes<T<D>>, T<D> >; \
extern template class C<Plato::Hyperbolic::GradientZTypes<T<D>>, T<D> >;

#define PLATO_HYPERBOLIC_EXPL_DEF2(C, T, D) \
template class C<Plato::Hyperbolic::ResidualTypes <T<D>>, T<D> >; \
template class C<Plato::Hyperbolic::GradientUTypes<T<D>>, T<D> >; \
template class C<Plato::Hyperbolic::GradientVTypes<T<D>>, T<D> >; \
template class C<Plato::Hyperbolic::GradientATypes<T<D>>, T<D> >; \
template class C<Plato::Hyperbolic::GradientXTypes<T<D>>, T<D> >; \
template class C<Plato::Hyperbolic::GradientZTypes<T<D>>, T<D> >;

#endif

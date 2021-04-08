#ifndef EXP_INST_MACROS_HPP
#define EXP_INST_MACROS_HPP

#define PLATO_EXPL_DEF_INC(C, T, D) \
template class C<Plato::ResidualTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::ResidualTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::ResidualTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::JacobianPTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::JacobianPTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::JacobianPTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::Heaviside >;

#define PLATO_EXPL_DEC_INC(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::JacobianPTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::JacobianPTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::JacobianPTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::Heaviside >;

#define PLATO_EXPL_DEF(C, T, D) \
template class C<Plato::ResidualTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::ResidualTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::ResidualTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::JacobianTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::GradientXTypes<T<D>>, Plato::Heaviside >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::MSIMP >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::RAMP >; \
template class C<Plato::GradientZTypes<T<D>>, Plato::Heaviside >;

#define PLATO_EXPL_DEC(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::ResidualTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::JacobianTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::GradientXTypes<T<D>>, Plato::Heaviside >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::MSIMP >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::RAMP >; \
extern template class C<Plato::GradientZTypes<T<D>>, Plato::Heaviside >;


#define PLATO_EXPL_DEC_INC_LOCAL(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>, T<D>>; \
extern template class C<Plato::JacobianTypes<T<D>>, T<D>>; \
extern template class C<Plato::JacobianPTypes<T<D>>, T<D>>; \
extern template class C<Plato::LocalJacobianTypes<T<D>>, T<D>>; \
extern template class C<Plato::LocalJacobianPTypes<T<D>>, T<D>>; \
extern template class C<Plato::GradientXTypes<T<D>>, T<D>>; \
extern template class C<Plato::GradientZTypes<T<D>>, T<D>>; 

#define PLATO_EXPL_DEF_INC_LOCAL(C, T, D) \
template class C<Plato::ResidualTypes<T<D>>, T<D>>; \
template class C<Plato::JacobianTypes<T<D>>, T<D>>; \
template class C<Plato::JacobianPTypes<T<D>>, T<D>>; \
template class C<Plato::LocalJacobianTypes<T<D>>, T<D>>; \
template class C<Plato::LocalJacobianPTypes<T<D>>, T<D>>; \
template class C<Plato::GradientXTypes<T<D>>, T<D>>; \
template class C<Plato::GradientZTypes<T<D>>, T<D>>; 

#define PLATO_EXPL_DEC2(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>, T<D> >; \
extern template class C<Plato::JacobianTypes<T<D>>, T<D> >; \
extern template class C<Plato::GradientXTypes<T<D>>, T<D> >; \
extern template class C<Plato::GradientZTypes<T<D>>, T<D> >;

#define PLATO_EXPL_DEF2(C, T, D) \
template class C<Plato::ResidualTypes<T<D>>, T<D> >; \
template class C<Plato::JacobianTypes<T<D>>, T<D> >; \
template class C<Plato::GradientXTypes<T<D>>, T<D> >; \
template class C<Plato::GradientZTypes<T<D>>, T<D> >;

#define PLATO_EXPL_DEC_INC_VMS(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>, T<D>>; \
extern template class C<Plato::JacobianTypes<T<D>>, T<D>>; \
extern template class C<Plato::JacobianPTypes<T<D>>, T<D>>; \
extern template class C<Plato::JacobianNTypes<T<D>>, T<D>>; \
extern template class C<Plato::LocalJacobianTypes<T<D>>, T<D>>; \
extern template class C<Plato::LocalJacobianPTypes<T<D>>, T<D>>; \
extern template class C<Plato::GradientXTypes<T<D>>, T<D>>; \
extern template class C<Plato::GradientZTypes<T<D>>, T<D>>; 

#define PLATO_EXPL_DEF_INC_VMS(C, T, D) \
template class C<Plato::ResidualTypes<T<D>>, T<D>>; \
template class C<Plato::JacobianTypes<T<D>>, T<D>>; \
template class C<Plato::JacobianPTypes<T<D>>, T<D>>; \
template class C<Plato::JacobianNTypes<T<D>>, T<D>>; \
template class C<Plato::LocalJacobianTypes<T<D>>, T<D>>; \
template class C<Plato::LocalJacobianPTypes<T<D>>, T<D>>; \
template class C<Plato::GradientXTypes<T<D>>, T<D>>; \
template class C<Plato::GradientZTypes<T<D>>, T<D>>; 

// INCOMPRESSIBLE FLUID FLOW DECLARATION
#define PLATO_EXPL_DEC_FLUIDS(C, P, S, D, M) \
extern template class C<P<D,M>,Plato::ResultTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::GradConfigTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::GradControlTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::GradCurrentMassTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::GradCurrentEnergyTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::GradCurrentMomentumTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::GradPreviousMassTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::GradPreviousEnergyTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::GradPreviousMomentumTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::GradMomentumPredictorTypes<S<D,M>>>;

// INCOMPRESSIBLE FLUID FLOW DEFINITION
#define PLATO_EXPL_DEF_FLUIDS(C, P, S, D, M) \
template class C<P<D,M>,Plato::ResultTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::GradConfigTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::GradControlTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::GradCurrentMassTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::GradCurrentEnergyTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::GradCurrentMomentumTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::GradPreviousMassTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::GradPreviousEnergyTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::GradPreviousMomentumTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::GradMomentumPredictorTypes<S<D,M>>>;

#endif

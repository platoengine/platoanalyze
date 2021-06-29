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

#define PLATO_EXPL_DEC_INC_LOCAL_1(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>>; \
extern template class C<Plato::JacobianTypes<T<D>>>; \
extern template class C<Plato::JacobianPTypes<T<D>>>; \
extern template class C<Plato::LocalJacobianTypes<T<D>>>; \
extern template class C<Plato::LocalJacobianPTypes<T<D>>>; \
extern template class C<Plato::GradientXTypes<T<D>>>; \
extern template class C<Plato::GradientZTypes<T<D>>>;

#define PLATO_EXPL_DEF_INC_LOCAL_1(C, T, D) \
template class C<Plato::ResidualTypes<T<D>>>; \
template class C<Plato::JacobianTypes<T<D>>>; \
template class C<Plato::JacobianPTypes<T<D>>>; \
template class C<Plato::LocalJacobianTypes<T<D>>>; \
template class C<Plato::LocalJacobianPTypes<T<D>>>; \
template class C<Plato::GradientXTypes<T<D>>>; \
template class C<Plato::GradientZTypes<T<D>>>;

#define PLATO_EXPL_DEC_INC_LOCAL_2(C, T, D) \
extern template class C<Plato::ResidualTypes<T<D>>, T<D>>; \
extern template class C<Plato::JacobianTypes<T<D>>, T<D>>; \
extern template class C<Plato::JacobianPTypes<T<D>>, T<D>>; \
extern template class C<Plato::LocalJacobianTypes<T<D>>, T<D>>; \
extern template class C<Plato::LocalJacobianPTypes<T<D>>, T<D>>; \
extern template class C<Plato::GradientXTypes<T<D>>, T<D>>; \
extern template class C<Plato::GradientZTypes<T<D>>, T<D>>; 

#define PLATO_EXPL_DEF_INC_LOCAL_2(C, T, D) \
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
extern template class C<P<D,M>,Plato::Fluids::ResultTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::Fluids::GradConfigTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::Fluids::GradControlTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::Fluids::GradCurrentMassTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::Fluids::GradCurrentEnergyTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::Fluids::GradCurrentMomentumTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::Fluids::GradPreviousMassTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::Fluids::GradPreviousEnergyTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::Fluids::GradPreviousMomentumTypes<S<D,M>>>; \
extern template class C<P<D,M>,Plato::Fluids::GradMomentumPredictorTypes<S<D,M>>>;

// INCOMPRESSIBLE FLUID FLOW DEFINITION
#define PLATO_EXPL_DEF_FLUIDS(C, P, S, D, M) \
template class C<P<D,M>,Plato::Fluids::ResultTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::Fluids::GradConfigTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::Fluids::GradControlTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::Fluids::GradCurrentMassTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::Fluids::GradCurrentEnergyTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::Fluids::GradCurrentMomentumTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::Fluids::GradPreviousMassTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::Fluids::GradPreviousEnergyTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::Fluids::GradPreviousMomentumTypes<S<D,M>>>; \
template class C<P<D,M>,Plato::Fluids::GradMomentumPredictorTypes<S<D,M>>>;

#endif

/*
 * ElasticWorkCriterion.cpp
 *
 *  Created on: Mar 7, 2020
 */

#include "ElasticWorkCriterion.hpp"

#ifdef PLATOANALYZE_2D
template class Plato::ElasticWorkCriterion<Plato::ResidualTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::ElasticWorkCriterion<Plato::JacobianTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::ElasticWorkCriterion<Plato::JacobianPTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::ElasticWorkCriterion<Plato::JacobianNTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::ElasticWorkCriterion<Plato::LocalJacobianTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::ElasticWorkCriterion<Plato::LocalJacobianPTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::ElasticWorkCriterion<Plato::GradientXTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::ElasticWorkCriterion<Plato::GradientZTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::ElasticWorkCriterion<Plato::ResidualTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::ElasticWorkCriterion<Plato::JacobianTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::ElasticWorkCriterion<Plato::JacobianPTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::ElasticWorkCriterion<Plato::JacobianNTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::ElasticWorkCriterion<Plato::LocalJacobianTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::ElasticWorkCriterion<Plato::LocalJacobianPTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::ElasticWorkCriterion<Plato::GradientXTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::ElasticWorkCriterion<Plato::GradientZTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>;
#endif

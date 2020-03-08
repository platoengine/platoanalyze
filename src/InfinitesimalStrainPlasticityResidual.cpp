/*
 * InfinitesimalStrainPlasticity.cpp
 * 
 * Created on: Mar 3, 2020
 */

#include "InfinitesimalStrainPlasticity.hpp"

#ifdef PLATOANALYZE_2D
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::ResidualTypes<Plato::SimplexPlasticity<1>>, Plato::SimplexPlasticity<1>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::JacobianTypes<Plato::SimplexPlasticity<1>>, Plato::SimplexPlasticity<1>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::JacobianPTypes<Plato::SimplexPlasticity<1>>, Plato::SimplexPlasticity<1>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::JacobianNTypes<Plato::SimplexPlasticity<1>>, Plato::SimplexPlasticity<1>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::LocalJacobianTypes<Plato::SimplexPlasticity<1>>, Plato::SimplexPlasticity<1>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::LocalJacobianPTypes<Plato::SimplexPlasticity<1>>, Plato::SimplexPlasticity<1>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::GradientXTypes<Plato::SimplexPlasticity<1>>, Plato::SimplexPlasticity<1>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::GradientZTypes<Plato::SimplexPlasticity<1>>, Plato::SimplexPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::ResidualTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::JacobianTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::JacobianPTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::JacobianNTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::LocalJacobianTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::LocalJacobianPTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::GradientXTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::GradientZTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::ResidualTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::JacobianTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::JacobianPTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::JacobianNTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::LocalJacobianTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::LocalJacobianPTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::GradientXTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
template class Plato::InfinitesimalStrainPlasticityResidual<Plato::GradientZTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>;
#endif

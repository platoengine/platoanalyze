
#include "IntermediateDensityPenalty.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::IntermediateDensityPenalty<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
template class Plato::IntermediateDensityPenalty<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
template class Plato::IntermediateDensityPenalty<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
template class Plato::IntermediateDensityPenalty<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::IntermediateDensityPenalty<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
template class Plato::IntermediateDensityPenalty<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
template class Plato::IntermediateDensityPenalty<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
template class Plato::IntermediateDensityPenalty<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::IntermediateDensityPenalty<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
template class Plato::IntermediateDensityPenalty<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
template class Plato::IntermediateDensityPenalty<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
template class Plato::IntermediateDensityPenalty<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
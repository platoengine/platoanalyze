#include <elliptic/VolumeAverageCriterion.hpp>

#ifdef PLATOANALYZE_2D
template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Thermal<2>>;
template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Mechanics<2>>;
template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Electromechanics<2>>;
template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Thermomechanics<2>>;
#ifdef PLATO_STABILIZED
template class Plato::Elliptic::VolumeAverageCriterion<::Plato::StabilizedMechanics<2>>;
template class Plato::Elliptic::VolumeAverageCriterion<::Plato::StabilizedThermomechanics<2>>;
#endif
#endif

#ifdef PLATOANALYZE_3D
template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Thermal<3>>;
template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Mechanics<3>>;
template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Electromechanics<3>>;
template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Thermomechanics<3>>;
#ifdef PLATO_STABILIZED
template class Plato::Elliptic::VolumeAverageCriterion<::Plato::StabilizedMechanics<3>>;
template class Plato::Elliptic::VolumeAverageCriterion<::Plato::StabilizedThermomechanics<3>>;
#endif
#endif
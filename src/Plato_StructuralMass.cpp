/*
 * Plato_StructuralMass.cpp
 *
 *  Created on: Apr 17, 2019
 */

#include "Plato_StructuralMass.hpp"

#ifdef PLATOANALYZE_1D
template class Plato::StructuralMass<1>;
#endif

#ifdef PLATOANALYZE_2D
template class Plato::StructuralMass<2>;
#endif

#ifdef PLATOANALYZE_3D
template class Plato::StructuralMass<3>;
#endif

/*
 * CriterionFlowRate.cpp
 *
 *  Created on: Apr 28, 2021
 */

#include "hyperbolic/CriterionFlowRate.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::CriterionFlowRate, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::CriterionFlowRate, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEF_FLUIDS(Plato::Fluids::CriterionFlowRate, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif

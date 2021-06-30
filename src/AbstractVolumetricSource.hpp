/*
 * AbstractVolumetricSource.hpp
 *
 *  Created on: June 16, 2021
 */

#pragma once

#include "WorkSets.hpp"
#include "ExpInstMacros.hpp"

#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"

namespace Plato
{

template<typename PhysicsT, typename EvaluationT>
class AbstractVolumetricSource
{
private:
    // set local ad type
    using ResultT = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */

public:
    virtual std::string type() const = 0;
    
    virtual std::string name() const = 0;

    virtual void evaluate
    (const Plato::WorkSets &aWorkSets, 
     Plato::ScalarMultiVectorT<ResultT> & aResultWS,
     Plato::Scalar aMultiplier = 1.0) 
     const = 0;
};
// class AbstractVolumetricSource

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC_FLUIDS(Plato::AbstractVolumetricSource, Plato::EnergyConservation, Plato::SimplexFluids, 1, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::AbstractVolumetricSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 1, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_FLUIDS(Plato::AbstractVolumetricSource, Plato::EnergyConservation, Plato::SimplexFluids, 2, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::AbstractVolumetricSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 2, 1)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_FLUIDS(Plato::AbstractVolumetricSource, Plato::EnergyConservation, Plato::SimplexFluids, 3, 1)
PLATO_EXPL_DEC_FLUIDS(Plato::AbstractVolumetricSource, Plato::IncompressibleFluids, Plato::SimplexFluids, 3, 1)
#endif

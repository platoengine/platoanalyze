/*
 * FluidsThermalSourceFactory.hpp
 *
 *  Created on: June 17, 2021
 */

#pragma once

#include "hyperbolic/IncompressibleFluids.hpp"

#include "hyperbolic/FluidsUniformThermalSource.hpp"
#include "hyperbolic/FluidsStabilizedUniformThermalSource.hpp"

namespace Plato
{

namespace Fluids
{

struct ThermalSourceFactory
{
public:
    template
    <typename PhysicsT,
     typename EvaluationT>
    std::shared_ptr<Plato::AbstractVolumetricSource<PhysicsT, EvaluationT>>
    createThermalSource
    (const std::string &aFuncType,
     const std::string &aFuncName,
     const Plato::SpatialDomain &aDomain,
     Plato::DataMap &aDataMap,
     Teuchos::ParameterList &aInputs)
    {
        auto tLowerScenario = Plato::Fluids::scenario(aInputs);
        auto tLowerFuncType = Plato::tolower(aFuncType);
        if (tLowerFuncType == "uniform" && tLowerScenario != "density-based topology optimization")
        {
            return std::make_shared<Plato::Fluids::UniformThermalSource<PhysicsT, EvaluationT>>(aFuncName, aDomain, aDataMap, aInputs);
        }
        else
        if(tLowerFuncType == "uniform" && tLowerScenario == "density-based topology optimization")
        {
            return std::make_shared<Plato::Fluids::SIMP::UniformThermalSource<PhysicsT, EvaluationT>>(aFuncName, aDomain, aDataMap, aInputs);
        }
        else
        {
            THROWERR(std::string("Volumetric source of type '") + tLowerFuncType + "' is not supported. Supported options are: 'uniform', 'stabilized uniform'.")
        }
    }

    template
    <typename PhysicsT,
     typename EvaluationT>
    std::shared_ptr<Plato::AbstractVolumetricSource<PhysicsT, EvaluationT>>
    createStabilizedThermalSource
    (const std::string &aFuncType,
     const std::string &aFuncName,
     const Plato::SpatialDomain &aDomain,
     Plato::DataMap &aDataMap,
     Teuchos::ParameterList &aInputs)
    {
        auto tLowerScenario = Plato::Fluids::scenario(aInputs);
        auto tLowerFuncType = Plato::tolower(aFuncType);
        if (tLowerFuncType == "uniform" && tLowerScenario != "density-based topology optimization")
        {
            return std::make_shared<Plato::Fluids::StabilizedUniformThermalSource<PhysicsT, EvaluationT>>(aFuncName, aDomain, aDataMap, aInputs);
        }
        else
        if(tLowerFuncType == "uniform" && tLowerScenario == "density-based topology optimization")
        {
            return std::make_shared<Plato::Fluids::SIMP::StabilizedUniformThermalSource<PhysicsT, EvaluationT>>(aFuncName, aDomain, aDataMap, aInputs);
        }
        else
        {
            THROWERR(std::string("Volumetric source of type '") + tLowerFuncType + "' is not supported. Supported options are: 'uniform', 'stabilized uniform'.")
        }
    }
};
// struct ThermalSourceFactory

}
// namespace Fluids

}
// namespace Plato
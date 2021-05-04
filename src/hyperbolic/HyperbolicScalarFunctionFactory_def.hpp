#pragma once

#include "hyperbolic/HyperbolicScalarFunctionBase.hpp"
#include "hyperbolic/HyperbolicPhysicsScalarFunction.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Hyperbolic
{
    /******************************************************************************//**
     * \brief Create method
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    template <typename PhysicsT>
    std::shared_ptr<Plato::Hyperbolic::ScalarFunctionBase>
    ScalarFunctionFactory<PhysicsT>::create(
        Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aInputParams,
        std::string            & aFunctionName
    )
    {
        auto tProblemFunction = aInputParams.sublist("Criteria").sublist(aFunctionName);
        auto tFunctionType = tProblemFunction.get<std::string>("Type", "Not Defined");

        if(tFunctionType == "Scalar Function")
        {
            return std::make_shared<Hyperbolic::PhysicsScalarFunction<PhysicsT>>(aSpatialModel, aDataMap, aInputParams, aFunctionName);
        }
        else
        {
            const std::string tErrorString = std::string("Unknown function Type '") + tFunctionType +
                            "' specified in function name " + aFunctionName + " ParameterList";
            THROWERR(tErrorString)
        }
    }
} //namespace Hyperbolic

} // namespace Plato

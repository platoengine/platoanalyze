#pragma once

#include "ScalarFunctionBase.hpp"
#include "parabolic/PhysicsScalarFunction.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Parabolic
{
    /******************************************************************************//**
     * @brief Create method
     * @param [in] aSpatialModel Plato Analyze spatial model
     * @param [in] aDataMap Plato Analyze data map
     * @param [in] aProblemParams parameter input
     * @param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    template <typename PhysicsT>
    std::shared_ptr<Plato::Parabolic::ScalarFunctionBase> 
    ScalarFunctionBaseFactory<PhysicsT>::create(
        Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProblemParams,
        std::string            & aFunctionName
    )
    {
        auto tProblemFunction = aProblemParams.sublist("Criteria").sublist(aFunctionName);
        auto tFunctionType = tProblemFunction.get<std::string>("Type", "Not Defined");

        if(tFunctionType == "Scalar Function")
        {
            return std::make_shared<Plato::Parabolic::PhysicsScalarFunction<PhysicsT>>
                (aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        {
            const std::string tErrorString = std::string("Unknown function Type '") + tFunctionType +
                            "' specified in function name " + aFunctionName + " ParameterList";
            THROWERR(tErrorString)
        }
    }
} // namespace Parabolic

} // namespace Plato

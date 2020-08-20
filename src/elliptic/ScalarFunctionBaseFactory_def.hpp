#pragma once

#include "elliptic/ScalarFunctionBase.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
//#include "elliptic/WeightedSumFunction.hpp"
//#include "elliptic/DivisionFunction.hpp"
//#include "elliptic/LeastSquaresFunction.hpp"
//#include "elliptic/MassPropertiesFunction.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Elliptic
{

    /******************************************************************************//**
     * \brief Create method
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato and Analyze data map
     * \param [in] aInputParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    template <typename PhysicsT>
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> 
    ScalarFunctionBaseFactory<PhysicsT>::create(
        Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aInputParams,
        std::string            & aFunctionName
    )
    {
        auto tProblemFunction = aInputParams.sublist(aFunctionName);
        auto tFunctionType = tProblemFunction.get<std::string>("Type", "Not Defined");

/* TODO
        if(tFunctionType == "Weighted Sum")
        {
            return std::make_shared<WeightedSumFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName);
        }
        else
        if(tFunctionType == "Division")
        {
            return std::make_shared<DivisionFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName);
        }
        else
        if(tFunctionType == "Least Squares")
        {
            return std::make_shared<LeastSquaresFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName);
        }
        else
        if(tFunctionType == "Mass Properties")
        {
            return std::make_shared<MassPropertiesFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aInputParams, aFunctionName);
        }
        else
*/
        if(tFunctionType == "Scalar Function")
        {
            return std::make_shared<PhysicsScalarFunction<PhysicsT>>(aSpatialModel, aDataMap, aInputParams, aFunctionName);
        }
        else
        {
            const std::string tErrorString = std::string("Unknown function Type '") + tFunctionType +
                            "' specified in function name " + aFunctionName + " ParameterList";
            THROWERR(tErrorString);
        }
    }

} // namespace Elliptic

} // namespace Plato

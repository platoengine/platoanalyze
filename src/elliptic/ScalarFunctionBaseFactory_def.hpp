#pragma once

#include "elliptic/ScalarFunctionBase.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/WeightedSumFunction.hpp"
#include "elliptic/DivisionFunction.hpp"
#include "elliptic/SolutionFunction.hpp"
#include "elliptic/LeastSquaresFunction.hpp"
#include "elliptic/MassPropertiesFunction.hpp"
#include "elliptic/VolumeAverageCriterion.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Elliptic
{

    /******************************************************************************//**
     * \brief Create method
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato and Analyze data map
     * \param [in] aProblemParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    template <typename PhysicsT>
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> 
    ScalarFunctionBaseFactory<PhysicsT>::create(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aFunctionName
    ) 
    {
        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(aFunctionName);
        auto tFunctionType = tFunctionParams.get<std::string>("Type", "Not Defined");

        if(tFunctionType == "Weighted Sum")
        {
            return std::make_shared<WeightedSumFunction<PhysicsT>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Division")
        {
            return std::make_shared<DivisionFunction<PhysicsT>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Solution")
        {
            return std::make_shared<SolutionFunction<PhysicsT>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Least Squares")
        {
            return std::make_shared<LeastSquaresFunction<PhysicsT>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Mass Properties")
        {
            return std::make_shared<MassPropertiesFunction<PhysicsT>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Volume Average Criterion")
        {
            return std::make_shared<VolumeAverageCriterion<PhysicsT>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        if(tFunctionType == "Scalar Function")
        {
            return std::make_shared<PhysicsScalarFunction<PhysicsT>>(aSpatialModel, aDataMap, aProblemParams, aFunctionName);
        }
        else
        {
            return nullptr;
        }
    }

} // namespace Elliptic

} // namespace Plato

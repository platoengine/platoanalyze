#pragma once

#include "elliptic/ScalarFunctionBase.hpp"
#include "elliptic/WeightedSumFunction.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/DivisionFunction.hpp"
#include "elliptic/LeastSquaresFunction.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Elliptic
{

    /******************************************************************************//**
     * \brief Create method
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aProblemParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    template <typename PhysicsT>
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> 
    ScalarFunctionBaseFactory<PhysicsT>::create(
        Omega_h::Mesh          & aMesh,
        Omega_h::MeshSets      & aMeshSets,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aProblemParams,
        std::string            & aFunctionName)
    {
        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(aFunctionName);
        auto tFunctionType = tFunctionParams.get<std::string>("Type", "Not Defined");

        if(tFunctionType == "Weighted Sum")
        {
            return std::make_shared<WeightedSumFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aProblemParams, aFunctionName);
        }
        else if(tFunctionType == "Division")
        {
            return std::make_shared<DivisionFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aProblemParams, aFunctionName);
        }
        else if(tFunctionType == "Least Squares")
        {
            return std::make_shared<LeastSquaresFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aProblemParams, aFunctionName);
        }
        else if(tFunctionType == "Scalar Function")
        {
            return std::make_shared<PhysicsScalarFunction<PhysicsT>>(aMesh, aMeshSets, aDataMap, aProblemParams, aFunctionName);
        }
        else
        {
            return nullptr;
        }
    }

} // namespace Elliptic

} // namespace Plato

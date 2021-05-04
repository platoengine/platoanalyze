#pragma once

#include "elliptic/updated_lagrangian/ScalarFunctionBase.hpp"
#include "elliptic/updated_lagrangian/PhysicsScalarFunction.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

    /******************************************************************************//**
     * \brief Create method
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato and Analyze data map
     * \param [in] aProblemParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    template <typename PhysicsT>
    std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::ScalarFunctionBase> 
    ScalarFunctionBaseFactory<PhysicsT>::create(
              Plato::SpatialModel                 & aSpatialModel,
        const Plato::Sequence<PhysicsT::SpaceDim> & aSequence,
              Plato::DataMap                      & aDataMap,
              Teuchos::ParameterList              & aProblemParams,
              std::string                         & aFunctionName
    ) 
    {
        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(aFunctionName);
        auto tFunctionType = tFunctionParams.get<std::string>("Type", "Not Defined");

        if(tFunctionType == "Scalar Function")
        {
            return std::make_shared<PhysicsScalarFunction<PhysicsT>>(aSpatialModel, aSequence, aDataMap, aProblemParams, aFunctionName);
        }
        else
        {
            return nullptr;
        }
    }
} // namespace UpdatedLagrangian

} // namespace Elliptic

} // namespace Plato

/*
 * PathDependentScalarFunctionFactory_def.hpp
 *
 *  Created on: Mar 8, 2020
 */

#pragma once

#include "BasicLocalScalarFunction.hpp"
#include "WeightedLocalScalarFunction.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Create interface for the evaluation of path-dependent scalar function
 *  operators, e.g. value and sensitivities.
 * \param [in] aMesh         mesh database
 * \param [in] aMeshSets     side sets database
 * \param [in] aDataMap      output database
 * \param [in] aInputParams  problem inputs in XML file
 * \param [in] aFunctionName scalar function name, i.e. type
 * \return shared pointer to the interface of path-dependent scalar functions
 **********************************************************************************/
template <typename PhysicsT>
std::shared_ptr<Plato::LocalScalarFunctionInc>
PathDependentScalarFunctionFactory<PhysicsT>::create(
    const Plato::SpatialModel    & aSpatialModel,
          Plato::DataMap         & aDataMap,
          Teuchos::ParameterList & aInputParams,
    const std::string            & aFunctionName)
{
    auto tProblemFunction = aInputParams.sublist("Criteria").sublist(aFunctionName);
    auto tFunctionType = tProblemFunction.get < std::string > ("Type", "UNDEFINED");
    if(tFunctionType == "Scalar Function")
    {
        return ( std::make_shared <Plato::BasicLocalScalarFunction<PhysicsT>>
                (aSpatialModel, aDataMap, aInputParams, aFunctionName) );
    } else
    if(tFunctionType == "Weighted Sum")
    {
        return ( std::make_shared <Plato::WeightedLocalScalarFunction<PhysicsT>>
                (aSpatialModel, aDataMap, aInputParams, aFunctionName) );
    }
    else
    {
        const auto tError = std::string("UNKNOWN SCALAR FUNCTION '") + tFunctionType
                + "'. OBJECTIVE OR CONSTRAINT KEYWORD WITH NAME '" + aFunctionName
                + "' IS NOT DEFINED.  MOST LIKELY, SUBLIST '" + aFunctionName
                + "' IS NOT DEFINED IN THE INPUT FILE.";
        THROWERR(tError);
    }
}

}
// namespace Plato

/*
 * FluidsCriterionFactory.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "PlatoUtilities.hpp"

#include "hyperbolic/FluidsScalarFunction.hpp"

namespace Plato
{

namespace Fluids
{

/**************************************************************************//**
* \struct CriterionFactory
*
* \brief Responsible for the construction of Plato criteria.
******************************************************************************/
template<typename PhysicsT>
class CriterionFactory
{
public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    CriterionFactory () {}

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~CriterionFactory() {}

    /******************************************************************************//**
     * \brief Create criterion interface.
     * \param [in] aModel   computational model metadata
     * \param [in] aDataMap output database
     * \param [in] aInputs  input file metadata
     * \param [in] aTag    scalar function tag
     **********************************************************************************/
    std::shared_ptr<Plato::Fluids::CriterionBase>
    createCriterion
    (Plato::SpatialModel    & aModel,
     Plato::DataMap         & aDataMap,
     Teuchos::ParameterList & aInputs,
     std::string            & aTag)
     {
        auto tFunctionTag = aInputs.sublist("Criteria").sublist(aTag);
        auto tType = tFunctionTag.get<std::string>("Type", "Not Defined");
        auto tLowerType = Plato::tolower(tType);

        if(tLowerType == "scalar function")
        {
            auto tCriterion =
                std::make_shared<Plato::Fluids::ScalarFunction<PhysicsT>>
                    (aModel, aDataMap, aInputs, aTag);
            return tCriterion;
        }
        /*else if(tLowerType == "weighted sum")
        {
            auto tCriterion =
                std::make_shared<Plato::Fluids::WeightedScalarFunction<PhysicsT>>
                    (aSpatialModel, aDataMap, aInputs, aName);
            return tCriterion;
        }
        else if(tLowerType == "least squares")
        {
            auto tCriterion =
                std::make_shared<Plato::Fluids::LeastSquaresScalarFunction<PhysicsT>>
                    (aSpatialModel, aDataMap, aInputs, aName);
            return tCriterion;
        }*/
        else
        {
            THROWERR(std::string("Scalar function in block '") + aTag + "' with Type '" + tType + "' is not supported.")
        }
     }
};
// class CriterionFactory

}
// namespace Fluids

}
// namespace Plato

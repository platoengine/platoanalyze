/*
 * ElasticModelFactory.hpp
 *
 *  Created on: Mar 24, 2020
 */

#pragma once

#include "LinearElasticMaterial.hpp"

#include "ElasticModelFactory.hpp"
#include "CubicLinearElasticMaterial.hpp"
#include "CustomLinearElasticMaterial.hpp"
#include "IsotropicLinearElasticMaterial.hpp"
#include "OrthotropicLinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Factory for creating linear elastic material models.
 *
 * \tparam SpatialDim spatial dimensions: options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class ElasticModelFactory
{
public:
    /******************************************************************************//**
    * \brief Linear elastic material model factory constructor.
    * \param [in] aParamList input parameter list
    **********************************************************************************/
    ElasticModelFactory(const Teuchos::ParameterList& aParamList) :
            mParamList(aParamList){}

    /******************************************************************************//**
    * \brief Create a linear elastic material model.
    * \param [in] aModelName name of the model to be created.
    * \return Teuchos reference counter pointer to linear elastic material model
    **********************************************************************************/
    Teuchos::RCP<Plato::LinearElasticMaterial<SpatialDim>>
    create(std::string aModelName)
    {
        if (!mParamList.isSublist("Material Models"))
        {
            REPORT("'Material Models' list not found! Returning 'nullptr'");
            return Teuchos::RCP<Plato::LinearElasticMaterial<SpatialDim>>(nullptr);
        }
        else
        {
            auto tModelsParamList = mParamList.get<Teuchos::ParameterList>("Material Models");
           
            if (!tModelsParamList.isSublist(aModelName))
            {
                std::stringstream ss;
                ss << "Requested a material model ('" << aModelName << "') that isn't defined";
                THROWERR(ss.str());
            }

            auto tModelParamList = tModelsParamList.sublist(aModelName);
            if(tModelParamList.isSublist("Isotropic Linear Elastic"))
            {
                return Teuchos::rcp(new Plato::IsotropicLinearElasticMaterial<SpatialDim>(tModelParamList.sublist("Isotropic Linear Elastic")));
            }
            else if(tModelParamList.isSublist("Cubic Linear Elastic"))
            {
                return Teuchos::rcp(new Plato::CubicLinearElasticMaterial<SpatialDim>(tModelParamList.sublist("Cubic Linear Elastic")));
            }
            else if(tModelParamList.isSublist("Custom Linear Elastic"))
            {
                return Teuchos::rcp(new Plato::CustomLinearElasticMaterial<SpatialDim>(tModelParamList.sublist("Custom Linear Elastic")));
            }
            else if(tModelParamList.isSublist("Orthotropic Linear Elastic"))
            {
                return Teuchos::rcp(new Plato::OrthotropicLinearElasticMaterial<SpatialDim>(tModelParamList.sublist("Orthotropic Linear Elastic")));
            }
            return Teuchos::RCP<Plato::LinearElasticMaterial<SpatialDim>>(nullptr);
        }
    }

private:
    const Teuchos::ParameterList& mParamList; /*!< Input parameter list */
};
// class ElasticModelFactory

}
// namespace Plato

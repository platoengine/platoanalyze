/*
 * ElasticModelFactory.cpp
 *
 *  Created on: Mar 24, 2020
 */

#include "ElasticModelFactory.hpp"
#include "CubicLinearElasticMaterial.hpp"
#include "CustomLinearElasticMaterial.hpp"
#include "IsotropicLinearElasticMaterial.hpp"
#include "OrthotropicLinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
* \brief Create a linear elastic material model. - 1D
**********************************************************************************/
template<>
Teuchos::RCP<LinearElasticMaterial<1>> ElasticModelFactory<1>::create()
{
    auto tModelParamList = mParamList.get<Teuchos::ParameterList>("Material Model");

    if(tModelParamList.isSublist("Isotropic Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::IsotropicLinearElasticMaterial<1>(tModelParamList.sublist("Isotropic Linear Elastic")));
    }
    else if(tModelParamList.isSublist("Cubic Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::CubicLinearElasticMaterial<1>(tModelParamList.sublist("Cubic Linear Elastic")));
    }
    else if(tModelParamList.isSublist("Custom Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::CustomLinearElasticMaterial<1>(tModelParamList.sublist("Custom Linear Elastic")));
    }
    else if(tModelParamList.isSublist("Orthotropic Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::OrthotropicLinearElasticMaterial<1>(tModelParamList.sublist("Orthotropic Linear Elastic")));
    }
    return Teuchos::RCP<Plato::LinearElasticMaterial<1>>(nullptr);
}

/******************************************************************************//**
* \brief Create a linear elastic material model. - 2D
**********************************************************************************/
template<>
Teuchos::RCP<LinearElasticMaterial<2>> ElasticModelFactory<2>::create()
{
    auto tModelParamList = mParamList.get<Teuchos::ParameterList>("Material Model");

    if(tModelParamList.isSublist("Isotropic Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::IsotropicLinearElasticMaterial<2>(tModelParamList.sublist("Isotropic Linear Elastic")));
    }
    else if(tModelParamList.isSublist("Cubic Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::CubicLinearElasticMaterial<2>(tModelParamList.sublist("Cubic Linear Elastic")));
    }
    else if(tModelParamList.isSublist("Custom Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::CustomLinearElasticMaterial<2>(tModelParamList.sublist("Custom Linear Elastic")));
    }
    else if(tModelParamList.isSublist("Orthotropic Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::OrthotropicLinearElasticMaterial<2>(tModelParamList.sublist("Orthotropic Linear Elastic")));
    }
    return Teuchos::RCP<Plato::LinearElasticMaterial<2>>(nullptr);
}

/******************************************************************************//**
* \brief Create a linear elastic material model. - 3D
**********************************************************************************/
template<>
Teuchos::RCP<LinearElasticMaterial<3>> ElasticModelFactory<3>::create()
{
    auto tModelParamList = mParamList.get<Teuchos::ParameterList>("Material Model");

    if(tModelParamList.isSublist("Isotropic Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::IsotropicLinearElasticMaterial<3>(tModelParamList.sublist("Isotropic Linear Elastic")));
    }
    else if(tModelParamList.isSublist("Cubic Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::CubicLinearElasticMaterial<3>(tModelParamList.sublist("Cubic Linear Elastic")));
    }
    else if(tModelParamList.isSublist("Custom Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::CustomLinearElasticMaterial<3>(tModelParamList.sublist("Custom Linear Elastic")));
    }
    else if(tModelParamList.isSublist("Orthotropic Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::OrthotropicLinearElasticMaterial<3>(tModelParamList.sublist("Orthotropic Linear Elastic")));
    }
    return Teuchos::RCP<Plato::LinearElasticMaterial<3>>(nullptr);
}

}
// namespace Plato

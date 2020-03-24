/*
 * ElasticModelFactory.hpp
 *
 *  Created on: Mar 24, 2020
 */

#pragma once

#include "LinearElasticMaterial.hpp"

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
    * \return Teuchos reference counter pointer to linear elastic material model
    **********************************************************************************/
    Teuchos::RCP<Plato::LinearElasticMaterial<SpatialDim>> create();

private:
    const Teuchos::ParameterList& mParamList; /*!< Input parameter list */
};
// class ElasticModelFactory

}
// namespace Plato

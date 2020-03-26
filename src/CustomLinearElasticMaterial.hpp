/*
 * CustomLinearElasticMaterial.hpp
 *
 *  Created on: Mar 24, 2020
 */

#pragma once

#include "CustomMaterial.hpp"
#include "LinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Derived class for custom linear elastic material model
 *
 * \tparam SpatialDim spatial dimensions, options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class CustomLinearElasticMaterial :
  public LinearElasticMaterial<SpatialDim>, public CustomMaterial
{
public:
    /******************************************************************************//**
     * \brief Linear elastic custom material model constructor.
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    CustomLinearElasticMaterial(const Teuchos::ParameterList& aParamList);

    /******************************************************************************//**
     * \brief Linear elastic custom material model destructor.
    **********************************************************************************/
    virtual ~CustomLinearElasticMaterial(){}
};
// class CubicLinearElasticMaterial

}
// namespace Plato

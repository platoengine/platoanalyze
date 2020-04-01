/*
 * CubicLinearElasticMaterial.hpp
 *
 *  Created on: Mar 24, 2020
 */

#pragma once

#include "LinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Derived class for cubic linear elastic material models
 *
 * \tparam SpatialDim spatial dimensions, options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class CubicLinearElasticMaterial : public LinearElasticMaterial<SpatialDim>
{
public:
    /******************************************************************************//**
     * \brief Linear elastic cubic material model constructor.
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    CubicLinearElasticMaterial(const Teuchos::ParameterList& aParamList);

    /******************************************************************************//**
     * \brief Linear elastic cubic material model destructor.
    **********************************************************************************/
    virtual ~CubicLinearElasticMaterial(){}
};
// class CubicLinearElasticMaterial

}
// namespace Plato

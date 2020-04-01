/*
 * IsotropicLinearElasticMaterial.hpp
 *
 *  Created on: Mar 24, 2020
 */

#pragma once

#include "LinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Derived class for isotropic linear elastic material model
 *
 * \tparam SpatialDim spatial dimensions, options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class IsotropicLinearElasticMaterial : public LinearElasticMaterial<SpatialDim>
{
public:
    /******************************************************************************//**
     * \brief Linear elastic isotropic material model constructor.
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    IsotropicLinearElasticMaterial(const Teuchos::ParameterList& aParamList);

    /******************************************************************************//**
     * \brief Linear elastic isotropic material model constructor.
     * \param [in] aYoungsModulus Young's Modulus
     * \param [in] aPoissonsRatio Poisson's Modulus
    **********************************************************************************/
    IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio);

    /******************************************************************************//**
     * \brief Linear elastic isotropic material model destructor.
    **********************************************************************************/
    virtual ~IsotropicLinearElasticMaterial(){}

private:
    Plato::Scalar mPoissonsRatio; /*!< Poisson's Ratio */
    Plato::Scalar mYoungsModulus; /*!< Young's Modulus */
};
// class IsotropicLinearElasticMaterial

}
// namespace Plato

/*
 * OrthotropicLinearElasticMaterial.hpp
 *
 *  Created on: Mar 24, 2020
 */

#pragma once

#include "LinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Linear elastic orthotropic material model.  In contrast to an isotropic
 * material, an orthotropic material has preferred directions of strength which are
 * mutually perpendicular.  The properties along these directions (also known as
 * principal directions) are the extreme values of elastic coefficients.
 *
 * \tparam SpatialDim spatial dimensions, options 1D, 2D, and 3D
 *
 **********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class OrthotropicLinearElasticMaterial : public LinearElasticMaterial<SpatialDim>
{
private:
    /******************************************************************************//**
     * \brief Set linear elastic orthotropic material constants
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    void setOrthoMaterialModel(const Teuchos::ParameterList& aParamList);

    /******************************************************************************//**
     * \brief Check linear elastic orthotropic material input parameters
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    void checkOrthoMaterialInputs(const Teuchos::ParameterList& aParamList);

    /******************************************************************************//**
     * \brief Check linear elastic orthotropic material stability constants
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    void checkOrthoMaterialStability(const Teuchos::ParameterList& aParamList);

public:
    /******************************************************************************//**
     * \brief Linear elastic orthotropic material model constructor.
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    OrthotropicLinearElasticMaterial(const Teuchos::ParameterList& aParamList);

    /******************************************************************************//**
     * \brief Linear elastic orthotropic material model constructor used for unit testing.
    **********************************************************************************/
    OrthotropicLinearElasticMaterial(){}

    /******************************************************************************//**
     * \brief Destructor.
    **********************************************************************************/
    virtual ~OrthotropicLinearElasticMaterial(){}

    /******************************************************************************//**
     * \brief Initialize linear elastic orthotropic material model.
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    void setMaterialModel(const Teuchos::ParameterList& aParamList);
};
// class OrthotropicLinearElasticMaterial

}
// namespace Plato

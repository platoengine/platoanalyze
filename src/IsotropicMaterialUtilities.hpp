/*
 * IsotropicMaterialUtilities.hpp
 *
 *  Created on: Feb 29, 2020
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Compute shear modulus
 * \param [in] aElasticModulus elastic modulus
 * \param [in] aPoissonRatio   poisson's ratio
 * \return shear modulus
*******************************************************************************/
inline Plato::Scalar compute_shear_modulus(const Plato::Scalar & aElasticModulus, const Plato::Scalar & aPoissonRatio)
{
    auto tShearModulus = aElasticModulus /
        ( static_cast<Plato::Scalar>(2) * ( static_cast<Plato::Scalar>(1) + aPoissonRatio) ) ;
    return (tShearModulus);
}
// function compute_shear_modulus

/***************************************************************************//**
 * \brief Compute bulk modulus
 * \param [in] aElasticModulus elastic modulus
 * \param [in] aPoissonRatio   poisson's ratio
 * \return bulk modulus
*******************************************************************************/
inline Plato::Scalar compute_bulk_modulus(const Plato::Scalar & aElasticModulus, const Plato::Scalar & aPoissonRatio)
{
    auto tShearModulus = aElasticModulus /
        ( static_cast<Plato::Scalar>(3) * ( static_cast<Plato::Scalar>(1) - ( static_cast<Plato::Scalar>(2) * aPoissonRatio) ) );
    return (tShearModulus);
}
// function compute_bulk_modulus

/***************************************************************************//**
 * \brief Parse elastic modulus
 * \param [in] aParamList input parameter list
 * \return elastic modulus
*******************************************************************************/
Plato::Scalar parse_elastic_modulus(Teuchos::ParameterList & aParamList)
{
    if(aParamList.isParameter("Youngs Modulus"))
    {
        Plato::Scalar tElasticModulus = aParamList.get<Plato::Scalar>("Youngs Modulus");
        return (tElasticModulus);
    }
    else
    {
        THROWERR("Youngs Modulus parameter is not defined in 'Isotropic Linear Elastic' sublist.")
    }
}
// function parse_elastic_modulus

/***************************************************************************//**
 * \brief Parse Poisson's ratio
 * \param [in] aParamList input parameter list
 * \return Poisson's ratio
*******************************************************************************/
Plato::Scalar parse_poissons_ratio(Teuchos::ParameterList & aParamList)
{
    if(aParamList.isParameter("Poissons Ratio"))
    {
        Plato::Scalar tPoissonsRatio = aParamList.get<Plato::Scalar>("Poissons Ratio");
        return (tPoissonsRatio);
    }
    else
    {
        THROWERR("Poisson's ratio parameter is not defined in 'Isotropic Linear Elastic' sublist.")
    }
}
// function parse_poissons_ratio

}
// namespace Plato

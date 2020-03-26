/*
 * IsotropicLinearElasticMaterial.cpp
 *
 *  Created on: Mar 24, 2020
 */

#include "IsotropicLinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Linear elastic isotropic material model constructor. - 1D
**********************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<1>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<1>(paramList)
{
    mPoissonsRatio = paramList.get<Plato::Scalar>("Poissons Ratio");
    mYoungsModulus = paramList.get<Plato::Scalar>("Youngs Modulus");
    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));
    mCellStiffness(0, 0) = tCoeff * (1.0 - tPoissonRatio);

    if(paramList.isType<Plato::Scalar>("Pressure Scaling"))
    {
        mPressureScaling = paramList.get<Plato::Scalar>("Pressure Scaling");
    }
    else
    {
        mPressureScaling = tYoungsModulus / (3.0 * (1.0 - 2.0 * tPoissonRatio));
    }
    if(paramList.isType<Plato::Scalar>("Mass Density"))
    {
        mCellDensity = paramList.get<Plato::Scalar>("Mass Density");
    }
    else
    {
        mCellDensity = 1.0;
    }
}

/******************************************************************************//**
 * \brief Linear elastic isotropic material model constructor. - 2D
**********************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<2>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<2>(paramList)
{
    mPoissonsRatio = paramList.get<Plato::Scalar>("Poissons Ratio");
    mYoungsModulus = paramList.get<Plato::Scalar>("Youngs Modulus");
    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));

    mCellStiffness(0, 0) = tCoeff * (1.0 - tPoissonRatio);
    mCellStiffness(0, 1) = tCoeff * tPoissonRatio;
    mCellStiffness(1, 0) = tCoeff * tPoissonRatio;
    mCellStiffness(1, 1) = tCoeff * (1.0 - tPoissonRatio);
    mCellStiffness(2, 2) = 1.0 / 2.0 * tCoeff * (1.0 - 2.0 * tPoissonRatio);

    if(paramList.isType<Plato::Scalar>("Pressure Scaling"))
    {
        mPressureScaling = paramList.get<Plato::Scalar>("Pressure Scaling");
    }
    else
    {
        mPressureScaling = tYoungsModulus / (3.0 * (1.0 - 2.0 * tPoissonRatio));
    }
    if(paramList.isType<Plato::Scalar>("Mass Density"))
    {
        mCellDensity = paramList.get<Plato::Scalar>("Mass Density");
    }
    else
    {
        mCellDensity = 1.0;
    }
}

/******************************************************************************//**
 * \brief Linear elastic isotropic material model constructor. - 3D
**********************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<3>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<3>(paramList)
{
    mPoissonsRatio = paramList.get<Plato::Scalar>("Poissons Ratio");
    mYoungsModulus = paramList.get<Plato::Scalar>("Youngs Modulus");
    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));

    mCellStiffness(0, 0) = tCoeff * (1.0 - tPoissonRatio);
    mCellStiffness(0, 1) = tCoeff * tPoissonRatio;
    mCellStiffness(0, 2) = tCoeff * tPoissonRatio;
    mCellStiffness(1, 0) = tCoeff * tPoissonRatio;
    mCellStiffness(1, 1) = tCoeff * (1.0 - tPoissonRatio);
    mCellStiffness(1, 2) = tCoeff * tPoissonRatio;
    mCellStiffness(2, 0) = tCoeff * tPoissonRatio;
    mCellStiffness(2, 1) = tCoeff * tPoissonRatio;
    mCellStiffness(2, 2) = tCoeff * (1.0 - tPoissonRatio);
    mCellStiffness(3, 3) = 1.0 / 2.0 * tCoeff * (1.0 - 2.0 * tPoissonRatio);
    mCellStiffness(4, 4) = 1.0 / 2.0 * tCoeff * (1.0 - 2.0 * tPoissonRatio);
    mCellStiffness(5, 5) = 1.0 / 2.0 * tCoeff * (1.0 - 2.0 * tPoissonRatio);

    if(paramList.isType<Plato::Scalar>("Pressure Scaling"))
    {
        mPressureScaling = paramList.get<Plato::Scalar>("Pressure Scaling");
    }
    else
    {
        mPressureScaling = tYoungsModulus / (3.0 * (1.0 - 2.0 * tPoissonRatio));
    }
    if(paramList.isType<Plato::Scalar>("Mass Density"))
    {
        mCellDensity = paramList.get<Plato::Scalar>("Mass Density");
    }
    else
    {
        mCellDensity = 1.0;
    }
}

/******************************************************************************//**
 * \brief Linear elastic isotropic material model constructor. - 1D
**********************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<1>::
IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio) :
    LinearElasticMaterial<1>(),
    mPoissonsRatio(aPoissonsRatio),
    mYoungsModulus(aYoungsModulus)
{
    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));
    mCellStiffness(0, 0) = tCoeff * (1.0 - tPoissonRatio);
}

/******************************************************************************//**
 * \brief Linear elastic isotropic material model constructor. - 2D
**********************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<2>::
IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio) :
    LinearElasticMaterial<2>(),
    mPoissonsRatio(aPoissonsRatio),
    mYoungsModulus(aYoungsModulus)
{
    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));
    mCellStiffness(0, 0) = tCoeff * (1.0 - tPoissonRatio);
    mCellStiffness(0, 1) = tCoeff * tPoissonRatio;
    mCellStiffness(1, 0) = tCoeff * tPoissonRatio;
    mCellStiffness(1, 1) = tCoeff * (1.0 - tPoissonRatio);
    mCellStiffness(2, 2) = 1.0 / 2.0 * tCoeff * (1.0 - 2.0 * tPoissonRatio);
}

/******************************************************************************//**
 * \brief Linear elastic isotropic material model constructor. - 3D
**********************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<3>::
IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio) :
    LinearElasticMaterial<3>(),
    mPoissonsRatio(aPoissonsRatio),
    mYoungsModulus(aYoungsModulus)
{
    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));
    mCellStiffness(0, 0) = tCoeff * (1.0 - tPoissonRatio);
    mCellStiffness(0, 1) = tCoeff * tPoissonRatio;
    mCellStiffness(0, 2) = tCoeff * tPoissonRatio;
    mCellStiffness(1, 0) = tCoeff * tPoissonRatio;
    mCellStiffness(1, 1) = tCoeff * (1.0 - tPoissonRatio);
    mCellStiffness(1, 2) = tCoeff * tPoissonRatio;
    mCellStiffness(2, 0) = tCoeff * tPoissonRatio;
    mCellStiffness(2, 1) = tCoeff * tPoissonRatio;
    mCellStiffness(2, 2) = tCoeff * (1.0 - tPoissonRatio);
    mCellStiffness(3, 3) = 1.0 / 2.0 * tCoeff * (1.0 - 2.0 * tPoissonRatio);
    mCellStiffness(4, 4) = 1.0 / 2.0 * tCoeff * (1.0 - 2.0 * tPoissonRatio);
    mCellStiffness(5, 5) = 1.0 / 2.0 * tCoeff * (1.0 - 2.0 * tPoissonRatio);
}

}
// namespace Plato

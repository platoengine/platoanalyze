/*
 * CustomLinearElasticMaterial.cpp
 *
 *  Created on: Mar 24, 2020
 */

#include "CustomLinearElasticMaterial.hpp"

namespace Plato
{

template<>
::Plato::CustomLinearElasticMaterial<1>::
CustomLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
  Plato::LinearElasticMaterial<1>(paramList), Plato::CustomMaterial(paramList)
{
    // These are the equivilent values in the custom equation.
    Plato::Scalar mPoissonsRatio = paramList.get<Plato::Scalar>("v");
    Plato::Scalar mYoungsModulus = paramList.get<Plato::Scalar>("E");

    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    // auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));

    // Get the value for tCoeff from the custom equation in the XML file.
    auto tCoeff = GetCustomExpressionValue( paramList );

    // Do everything as usual for a Linear Elastic Material
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

template<>
::Plato::CustomLinearElasticMaterial<2>::
 CustomLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
Plato::LinearElasticMaterial<2>(paramList), Plato::CustomMaterial(paramList)
{
    // These are the equivilent values in the custom equation.
    Plato::Scalar mPoissonsRatio = paramList.get<Plato::Scalar>("v");
    Plato::Scalar mYoungsModulus = paramList.get<Plato::Scalar>("E");

    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    // auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));

    // Get the value for tCoeff from the custom equation in the XML file.
    auto tCoeff = GetCustomExpressionValue( paramList );

    // Do everything as usual for a Linear Elastic Material
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

template<>
::Plato::CustomLinearElasticMaterial<3>::
 CustomLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
Plato::LinearElasticMaterial<3>(paramList), Plato::CustomMaterial(paramList)
{
    // These are the equivilent values in the custom equation.
    Plato::Scalar mPoissonsRatio = paramList.get<Plato::Scalar>("v");
    Plato::Scalar mYoungsModulus = paramList.get<Plato::Scalar>("E");

    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    // auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));

    // Get the value for tCoeff from the custom equation in the XML file.
    auto tCoeff = GetCustomExpressionValue( paramList );

    // Do everything as usual for a Linear Elastic Material
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

}
// namespace Plato

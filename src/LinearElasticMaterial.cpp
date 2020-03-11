#include "LinearElasticMaterial.hpp"

namespace Plato
{

/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<1>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<1>(paramList)
/******************************************************************************/
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
/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<2>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<2>(paramList)
/******************************************************************************/
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
/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<3>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<3>(paramList)
/******************************************************************************/
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

/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<1>::
IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio) :
    LinearElasticMaterial<1>(),
    mPoissonsRatio(aPoissonsRatio),
    mYoungsModulus(aYoungsModulus)
/******************************************************************************/
{
    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));
    mCellStiffness(0, 0) = tCoeff * (1.0 - tPoissonRatio);
}

/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<2>::
IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio) :
    LinearElasticMaterial<2>(),
    mPoissonsRatio(aPoissonsRatio),
    mYoungsModulus(aYoungsModulus)
/******************************************************************************/
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
/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<3>::
IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio) :
    LinearElasticMaterial<3>(),
    mPoissonsRatio(aPoissonsRatio),
    mYoungsModulus(aYoungsModulus)
/******************************************************************************/
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

/******************************************************************************/
template<>
::Plato::CubicLinearElasticMaterial<1>::
CubicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<1>(paramList)
/******************************************************************************/
{
    Plato::Scalar tC11 = paramList.get<Plato::Scalar>("C11");
    mCellStiffness(0, 0) = tC11;

/*
    if( paramList.isType<Plato::Scalar>("Pressure Scaling") ){
      mPressureScaling = paramList.get<Plato::Scalar>("Pressure Scaling");
    } else {
      mPressureScaling = k / (3.0*(1.0-2.0*v));
    }
*/
}

/******************************************************************************/
template<>
::Plato::CubicLinearElasticMaterial<2>::
CubicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<2>(paramList)
/******************************************************************************/
{
    Plato::Scalar tC11 = paramList.get<Plato::Scalar>("C11");
    Plato::Scalar tC12 = paramList.get<Plato::Scalar>("C12");
    Plato::Scalar tC44 = paramList.get<Plato::Scalar>("C44");

    mCellStiffness(0, 0) = tC11;
    mCellStiffness(0, 1) = tC12;
    mCellStiffness(1, 0) = tC12;
    mCellStiffness(1, 1) = tC11;
    mCellStiffness(2, 2) = tC44;

/*
    if( paramList.isType<Plato::Scalar>("Pressure Scaling") ){
      mPressureScaling = paramList.get<Plato::Scalar>("Pressure Scaling");
    } else {
      mPressureScaling = k / (3.0*(1.0-2.0*v));
    }
*/
}

/******************************************************************************/
template<>
::Plato::CubicLinearElasticMaterial<3>::
CubicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<3>(paramList)
/******************************************************************************/
{
    Plato::Scalar tC11 = paramList.get<Plato::Scalar>("C11");
    Plato::Scalar tC12 = paramList.get<Plato::Scalar>("C12");
    Plato::Scalar tC44 = paramList.get<Plato::Scalar>("C44");

    mCellStiffness(0, 0) = tC11;
    mCellStiffness(0, 1) = tC12;
    mCellStiffness(0, 2) = tC12;
    mCellStiffness(1, 0) = tC12;
    mCellStiffness(1, 1) = tC11;
    mCellStiffness(1, 2) = tC12;
    mCellStiffness(2, 0) = tC12;
    mCellStiffness(2, 1) = tC12;
    mCellStiffness(2, 2) = tC11;
    mCellStiffness(3, 3) = tC44;
    mCellStiffness(4, 4) = tC44;
    mCellStiffness(5, 5) = tC44;

/*
    if( paramList.isType<Plato::Scalar>("Pressure Scaling") ){
      mPressureScaling = paramList.get<Plato::Scalar>("Pressure Scaling");
    } else {
      mPressureScaling = k / (3.0*(1.0-2.0*v));
    }

*/
}

/******************************************************************************/
template<>
::Plato::CustomLinearElasticMaterial<1>::
CustomLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
  LinearElasticMaterial<1>(paramList), CustomMaterial(paramList)
/******************************************************************************/
{
    // These are the equivilent values in the custom equation.
    Plato::Scalar mPoissonsRatio = paramList.get<Plato::Scalar>("v");
    Plato::Scalar mYoungsModulus = paramList.get<Plato::Scalar>("E");

    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    // auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));

    // Get the value for tCoeff from the custom equation in the XML file.
    auto tCoeff = GetCustomExpressionValue( paramList );

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

/******************************************************************************/
template<>
::Plato::CustomLinearElasticMaterial<2>::
 CustomLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<2>(paramList), CustomMaterial(paramList)
/******************************************************************************/
{
    // These are the equivilent values in the custom equation.
    Plato::Scalar mPoissonsRatio = paramList.get<Plato::Scalar>("v");
    Plato::Scalar mYoungsModulus = paramList.get<Plato::Scalar>("E");

    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    // auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));

    // Get the value for tCoeff from the custom equation in the XML file.
    auto tCoeff = GetCustomExpressionValue( paramList );

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

/******************************************************************************/
template<>
::Plato::CustomLinearElasticMaterial<3>::
 CustomLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<3>(paramList), CustomMaterial(paramList)
/******************************************************************************/
{
    // These are the equivilent values in the custom equation.
    Plato::Scalar mPoissonsRatio = paramList.get<Plato::Scalar>("v");
    Plato::Scalar mYoungsModulus = paramList.get<Plato::Scalar>("E");

    auto tPoissonRatio = mPoissonsRatio;
    auto tYoungsModulus = mYoungsModulus;
    // auto tCoeff = tYoungsModulus / ((1.0 + tPoissonRatio) * (1.0 - 2.0 * tPoissonRatio));

    // Get the value for tCoeff from the custom equation in the XML file.
    auto tCoeff = GetCustomExpressionValue( paramList );

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

} // namespace Plato

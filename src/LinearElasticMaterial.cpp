#include "LinearElasticMaterial.hpp"

namespace Plato {

/******************************************************************************/
template<>
::Plato::IsotropicLinearElasticMaterial<1>::
IsotropicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<1>(paramList)
/******************************************************************************/
{
    mPoissonsRatio = paramList.get<Plato::Scalar>("Poissons Ratio");
    mYoungsModulus = paramList.get<Plato::Scalar>("Youngs Modulus");
    auto v = mPoissonsRatio;
    auto k = mYoungsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v);

    if( paramList.isType<Plato::Scalar>("Pressure Scaling") ){
      mPressureScaling = paramList.get<Plato::Scalar>("Pressure Scaling");
    } else {
      mPressureScaling = k / (3.0*(1.0-2.0*v));
    }
    if( paramList.isType<Plato::Scalar>("Mass Density") ){
      mCellDensity = paramList.get<Plato::Scalar>("Mass Density");
    } else {
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
    auto v = mPoissonsRatio;
    auto k = mYoungsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));

    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v);
    mCellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);

    if( paramList.isType<Plato::Scalar>("Pressure Scaling") ){
      mPressureScaling = paramList.get<Plato::Scalar>("Pressure Scaling");
    } else {
      mPressureScaling = k / (3.0*(1.0-2.0*v));
    }
    if( paramList.isType<Plato::Scalar>("Mass Density") ){
      mCellDensity = paramList.get<Plato::Scalar>("Mass Density");
    } else {
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
    auto v = mPoissonsRatio;
    auto k = mYoungsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));

    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;       mCellStiffness(0,2)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v); mCellStiffness(1,2)=c*v;
    mCellStiffness(2,0)=c*v;       mCellStiffness(2,1)=c*v;       mCellStiffness(2,2)=c*(1.0-v);
    mCellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);

    if( paramList.isType<Plato::Scalar>("Pressure Scaling") ){
      mPressureScaling = paramList.get<Plato::Scalar>("Pressure Scaling");
    } else {
      mPressureScaling = k / (3.0*(1.0-2.0*v));
    }
    if( paramList.isType<Plato::Scalar>("Mass Density") ){
      mCellDensity = paramList.get<Plato::Scalar>("Mass Density");
    } else {
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
    auto v = mPoissonsRatio;
    auto k = mYoungsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v);
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
    auto v = mPoissonsRatio;
    auto k = mYoungsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v);
    mCellStiffness(2,2)=1.0/2.0*c*(1.0-2.0*v);
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
    auto v = mPoissonsRatio;
    auto k = mYoungsModulus;
    auto c = k/((1.0+v)*(1.0-2.0*v));
    mCellStiffness(0,0)=c*(1.0-v); mCellStiffness(0,1)=c*v;       mCellStiffness(0,2)=c*v;
    mCellStiffness(1,0)=c*v;       mCellStiffness(1,1)=c*(1.0-v); mCellStiffness(1,2)=c*v;
    mCellStiffness(2,0)=c*v;       mCellStiffness(2,1)=c*v;       mCellStiffness(2,2)=c*(1.0-v);
    mCellStiffness(3,3)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(4,4)=1.0/2.0*c*(1.0-2.0*v);
    mCellStiffness(5,5)=1.0/2.0*c*(1.0-2.0*v);
}

/******************************************************************************/
template<>
::Plato::CubicLinearElasticMaterial<1>::
CubicLinearElasticMaterial(const Teuchos::ParameterList& paramList) :
LinearElasticMaterial<1>(paramList)
/******************************************************************************/
{
    Plato::Scalar C11   = paramList.get<Plato::Scalar>("C11");
    mCellStiffness(0,0)=C11;

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
    Plato::Scalar C11   = paramList.get<Plato::Scalar>("C11");
    Plato::Scalar C12   = paramList.get<Plato::Scalar>("C12");
    Plato::Scalar C44   = paramList.get<Plato::Scalar>("C44");

    mCellStiffness(0,0)=C11; mCellStiffness(0,1)=C12;
    mCellStiffness(1,0)=C12; mCellStiffness(1,1)=C11;
    mCellStiffness(2,2)=C44;

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
    Plato::Scalar C11   = paramList.get<Plato::Scalar>("C11");
    Plato::Scalar C12   = paramList.get<Plato::Scalar>("C12");
    Plato::Scalar C44   = paramList.get<Plato::Scalar>("C44");

    mCellStiffness(0,0)=C11; mCellStiffness(0,1)=C12; mCellStiffness(0,2)=C12;
    mCellStiffness(1,0)=C12; mCellStiffness(1,1)=C11; mCellStiffness(1,2)=C12;
    mCellStiffness(2,0)=C12; mCellStiffness(2,1)=C12; mCellStiffness(2,2)=C11;
    mCellStiffness(3,3)=C44;
    mCellStiffness(4,4)=C44;
    mCellStiffness(5,5)=C44;

/*
    if( paramList.isType<Plato::Scalar>("Pressure Scaling") ){
      mPressureScaling = paramList.get<Plato::Scalar>("Pressure Scaling");
    } else {
      mPressureScaling = k / (3.0*(1.0-2.0*v));
    }

*/
}








} // namespace Plato 

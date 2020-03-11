#ifndef LINEARELASTICMATERIAL_HPP
#define LINEARELASTICMATERIAL_HPP

#include <Omega_h_matrix.hpp>
#include <Teuchos_ParameterList.hpp>

#include "PlatoStaticsTypes.hpp"

#include "CustomMaterial.hpp"

namespace Plato
{

/******************************************************************************/
/*!
  \brief Base class for Linear Elastic material models
*/
  template<Plato::OrdinalType SpatialDim>
  class LinearElasticMaterial
/******************************************************************************/
{
protected:
    static constexpr auto mNumVoigtTerms = (SpatialDim == 3) ? 6 :
                                           ((SpatialDim == 2) ? 3 :
                                          (((SpatialDim == 1) ? 1 : 0)));
    static_assert(mNumVoigtTerms, "SpatialDim must be 1, 2, or 3.");

    Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    Omega_h::Vector<mNumVoigtTerms> mReferenceStrain;
    Plato::Scalar mCellDensity;
    Plato::Scalar mPressureScaling;

public:
    LinearElasticMaterial();
    LinearElasticMaterial(const Teuchos::ParameterList& paramList);
    decltype(mCellDensity)     getMassDensity()     const {return mCellDensity;}
    decltype(mCellStiffness)   getStiffnessMatrix() const {return mCellStiffness;}
    decltype(mPressureScaling) getPressureScaling() const {return mPressureScaling;}
    decltype(mReferenceStrain) getReferenceStrain() const {return mReferenceStrain;}

  private:
    void initialize ();
};

/******************************************************************************/
template<Plato::OrdinalType SpatialDim>
void LinearElasticMaterial<SpatialDim>::
initialize()
/******************************************************************************/
{
    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellStiffness(tIndexI, tIndexJ) = 0.0;
        }
    }

    mPressureScaling = 1.0;

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        mReferenceStrain(tIndexI) = 0.0;
    }
}


/******************************************************************************/
template<Plato::OrdinalType SpatialDim>
LinearElasticMaterial<SpatialDim>::
LinearElasticMaterial()
/******************************************************************************/
{
  initialize();
}

/******************************************************************************/
template<Plato::OrdinalType SpatialDim>
LinearElasticMaterial<SpatialDim>::
LinearElasticMaterial(const Teuchos::ParameterList& paramList)
/******************************************************************************/

{
  initialize();

  if( paramList.isType<Plato::Scalar>("e11") )  mReferenceStrain(0) = paramList.get<Plato::Scalar>("e11");
  if( paramList.isType<Plato::Scalar>("e22") )  mReferenceStrain(1) = paramList.get<Plato::Scalar>("e22");
  if( paramList.isType<Plato::Scalar>("e33") )  mReferenceStrain(2) = paramList.get<Plato::Scalar>("e33");
  if( paramList.isType<Plato::Scalar>("e23") )  mReferenceStrain(3) = paramList.get<Plato::Scalar>("e23");
  if( paramList.isType<Plato::Scalar>("e13") )  mReferenceStrain(4) = paramList.get<Plato::Scalar>("e13");
  if( paramList.isType<Plato::Scalar>("e12") )  mReferenceStrain(5) = paramList.get<Plato::Scalar>("e12");
}

/******************************************************************************/
/*!
  \brief Derived class for isotropic linear elastic material model
*/
  template<Plato::OrdinalType SpatialDim>
  class IsotropicLinearElasticMaterial : public LinearElasticMaterial<SpatialDim>
/******************************************************************************/
{
public:
    IsotropicLinearElasticMaterial(const Teuchos::ParameterList& aParamList);
    IsotropicLinearElasticMaterial(const Plato::Scalar & aYoungsModulus, const Plato::Scalar & aPoissonsRatio);
    virtual ~IsotropicLinearElasticMaterial(){}

private:
    Plato::Scalar mPoissonsRatio;
    Plato::Scalar mYoungsModulus;
};
// class IsotropicLinearElasticMaterial

/******************************************************************************/
/*!
  \brief Derived class for cubic linear elastic material model
*/
  template<Plato::OrdinalType SpatialDim>
  class CubicLinearElasticMaterial : public LinearElasticMaterial<SpatialDim>
/******************************************************************************/
{
  public:
    CubicLinearElasticMaterial(const Teuchos::ParameterList& aParamList);
    virtual ~CubicLinearElasticMaterial(){}
};
// class CubicLinearElasticMaterial

/******************************************************************************/
/*!
  \brief Derived class for custom linear elastic material models
*/
  template<Plato::OrdinalType SpatialDim>
  class CustomLinearElasticMaterial :
    public LinearElasticMaterial<SpatialDim>, public CustomMaterial
/******************************************************************************/
{
public:
    CustomLinearElasticMaterial(const Teuchos::ParameterList& aParamList);
    virtual ~CustomLinearElasticMaterial(){}
};
// class CubicLinearElasticMaterial

/******************************************************************************/
/*!
  \brief Factory for creating material models
*/
  template<Plato::OrdinalType SpatialDim>
  class ElasticModelFactory
/******************************************************************************/
{
public:
    ElasticModelFactory(const Teuchos::ParameterList& aParamList) :
            mParamList(aParamList){}
    Teuchos::RCP<Plato::LinearElasticMaterial<SpatialDim>> create();

private:
    const Teuchos::ParameterList& mParamList;
};

/******************************************************************************/
template<Plato::OrdinalType SpatialDim>
Teuchos::RCP<LinearElasticMaterial<SpatialDim>>
ElasticModelFactory<SpatialDim>::create()
/******************************************************************************/
{
    auto tModelParamList = mParamList.get<Teuchos::ParameterList>("Material Model");

    if(tModelParamList.isSublist("Isotropic Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::IsotropicLinearElasticMaterial<SpatialDim>(tModelParamList.sublist("Isotropic Linear Elastic")));
    }
    else if(tModelParamList.isSublist("Cubic Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::CubicLinearElasticMaterial<SpatialDim>(tModelParamList.sublist("Cubic Linear Elastic")));
    }
    else if(tModelParamList.isSublist("Custom Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::CustomLinearElasticMaterial<SpatialDim>(tModelParamList.sublist("Custom Linear Elastic")));
    }
    return Teuchos::RCP<Plato::LinearElasticMaterial<SpatialDim>>(nullptr);
}

} // namespace Plato

#endif

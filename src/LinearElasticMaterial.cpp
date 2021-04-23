#include "LinearElasticMaterial.hpp"

namespace Plato
{

//*********************************************************************************
//**************************** NEXT: 1D Implementation ****************************
//*********************************************************************************

template<>
void LinearElasticMaterial<1>::initialize()
{
    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellStiffness(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        mReferenceStrain(tIndexI) = 0.0;
    }
}

template<>
void LinearElasticMaterial<1>::setReferenceStrainTensor(const Teuchos::ParameterList& aParamList)
{
    if(aParamList.isType<Plato::Scalar>("e11"))
        mReferenceStrain(0) = aParamList.get<Plato::Scalar>("e11");
}

template<>
LinearElasticMaterial<1>::LinearElasticMaterial() :
        mCellDensity(1.0),
        mPressureScaling(1.0),
        mRayleighA(0.0),
        mRayleighB(0.0)
{
    this->initialize();
}

template<>
LinearElasticMaterial<1>::LinearElasticMaterial(const Teuchos::ParameterList& aParamList) :
        mCellDensity(1.0),
        mPressureScaling(1.0),
        mRayleighA(0.0),
        mRayleighB(0.0)
{
    this->initialize();
    this->setReferenceStrainTensor(aParamList);

    if(aParamList.isType<Plato::Scalar>("RayleighA"))
    {
        mRayleighA = aParamList.get<Plato::Scalar>("RayleighA");
    }
    if(aParamList.isType<Plato::Scalar>("RayleighB"))
    {
        mRayleighB = aParamList.get<Plato::Scalar>("RayleighB");
    }
}

//*********************************************************************************
//**************************** NEXT: 2D Implementation ****************************
//*********************************************************************************

template<>
void LinearElasticMaterial<2>::initialize()
{
    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellStiffness(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        mReferenceStrain(tIndexI) = 0.0;
    }
}

template<>
void LinearElasticMaterial<2>::setReferenceStrainTensor(const Teuchos::ParameterList& aParamList)
{
    if(aParamList.isType<Plato::Scalar>("e11"))
        mReferenceStrain(0) = aParamList.get<Plato::Scalar>("e11");
    if(aParamList.isType<Plato::Scalar>("e22"))
        mReferenceStrain(1) = aParamList.get<Plato::Scalar>("e22");
    if(aParamList.isType<Plato::Scalar>("e12"))
        mReferenceStrain(2) = aParamList.get<Plato::Scalar>("e12");
}

template<>
LinearElasticMaterial<2>::LinearElasticMaterial() :
        mCellDensity(1.0),
        mPressureScaling(1.0),
        mRayleighA(0.0),
        mRayleighB(0.0)
{
    this->initialize();
}

template<>
LinearElasticMaterial<2>::LinearElasticMaterial(const Teuchos::ParameterList& aParamList) :
        mCellDensity(1.0),
        mPressureScaling(1.0),
        mRayleighA(0.0),
        mRayleighB(0.0)
{
    this->initialize();
    this->setReferenceStrainTensor(aParamList);

    if(aParamList.isType<Plato::Scalar>("RayleighA"))
    {
        mRayleighA = aParamList.get<Plato::Scalar>("RayleighA");
    }
    if(aParamList.isType<Plato::Scalar>("RayleighB"))
    {
        mRayleighB = aParamList.get<Plato::Scalar>("RayleighB");
    }
}


//*********************************************************************************
//**************************** NEXT: 3D Implementation ****************************
//*********************************************************************************

template<>
void LinearElasticMaterial<3>::initialize()
{
    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < mNumVoigtTerms; tIndexJ++)
        {
            mCellStiffness(tIndexI, tIndexJ) = 0.0;
        }
    }

    for(Plato::OrdinalType tIndexI = 0; tIndexI < mNumVoigtTerms; tIndexI++)
    {
        mReferenceStrain(tIndexI) = 0.0;
    }
}

template<>
void LinearElasticMaterial<3>::setReferenceStrainTensor(const Teuchos::ParameterList& aParamList)
{
    if(aParamList.isType<Plato::Scalar>("e11"))
        mReferenceStrain(0) = aParamList.get<Plato::Scalar>("e11");
    if(aParamList.isType<Plato::Scalar>("e22"))
        mReferenceStrain(1) = aParamList.get<Plato::Scalar>("e22");
    if(aParamList.isType<Plato::Scalar>("e33"))
        mReferenceStrain(2) = aParamList.get<Plato::Scalar>("e33");
    if(aParamList.isType<Plato::Scalar>("e23"))
        mReferenceStrain(3) = aParamList.get<Plato::Scalar>("e23");
    if(aParamList.isType<Plato::Scalar>("e13"))
        mReferenceStrain(4) = aParamList.get<Plato::Scalar>("e13");
    if(aParamList.isType<Plato::Scalar>("e12"))
        mReferenceStrain(5) = aParamList.get<Plato::Scalar>("e12");
}

template<>
LinearElasticMaterial<3>::LinearElasticMaterial() :
        mCellDensity(1.0),
        mPressureScaling(1.0),
        mRayleighA(0.0),
        mRayleighB(0.0)
{
    this->initialize();
}

template<>
LinearElasticMaterial<3>::LinearElasticMaterial(const Teuchos::ParameterList& aParamList) :
        mCellDensity(1.0),
        mPressureScaling(1.0),
        mRayleighA(0.0),
        mRayleighB(0.0)
{
    this->initialize();
    this->setReferenceStrainTensor(aParamList);

    if(aParamList.isType<Plato::Scalar>("RayleighA"))
    {
        mRayleighA = aParamList.get<Plato::Scalar>("RayleighA");
    }
    if(aParamList.isType<Plato::Scalar>("RayleighB"))
    {
        mRayleighB = aParamList.get<Plato::Scalar>("RayleighB");
    }
}

} // namespace Plato

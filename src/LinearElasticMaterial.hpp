#ifndef LINEARELASTICMATERIAL_HPP
#define LINEARELASTICMATERIAL_HPP

#include <Omega_h_matrix.hpp>
#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
<<<<<<< HEAD
{

/******************************************************************************/
/*!
 \brief Base class for Linear Elastic material models
 */
template<int SpatialDim>
class LinearElasticMaterial
/******************************************************************************/
{
protected:
    static constexpr auto mNumVoigtTerms = (SpatialDim == 3) ? 6 : ((SpatialDim == 2) ? 3 : (((SpatialDim == 1) ? 1 : 0)));
    static_assert(mNumVoigtTerms, "SpatialDim must be 1, 2, or 3.");

    Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffness; /*!< fourth-order material stiffness tensor */
    Omega_h::Vector<mNumVoigtTerms> mReferenceStrain;               /*!< reference second-order strain tensor, e.g. microstructure strains */
    Plato::Scalar mCellDensity;                                     /*!< volumetric mass density */
    Plato::Scalar mPressureScaling;                                 /*!< pressure scaling for stabilized mechanics */
    
public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    LinearElasticMaterial();

    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aParamList input parameters list
    *******************************************************************************/
    LinearElasticMaterial(const Teuchos::ParameterList &aParamList);
 
    /***************************************************************************//**
     * \brief Return volumetric mass density
     * \return volumetric mass density
    *******************************************************************************/
    decltype(mCellDensity) getMassDensity() const 
    {
        return mCellDensity;
    }

    /***************************************************************************//**
     * \brief Return fourth-order material stiffness tensor
     * \return fourth-order material stiffness tensor
    *******************************************************************************/
    decltype(mCellStiffness) getStiffnessMatrix() const
    {
        return mCellStiffness;
    }

    /***************************************************************************//**
     * \brief Return pressure scaling for stabilized mechanics
     * \return pressure scaling for stabilized mechanics
    *******************************************************************************/
    decltype(mPressureScaling) getPressureScaling() const
    {
        return mPressureScaling;
    }

    /***************************************************************************//**
     * \brief Return reference second-order strain tensor
     * \return reference second-order strain tensor
    *******************************************************************************/
    decltype(mReferenceStrain) getReferenceStrain() const
    {
        return mReferenceStrain;
    }

private:
    /***************************************************************************//**
     * \brief Initialize member data
    *******************************************************************************/
    void initialize();
};

/******************************************************************************/
template<int SpatialDim>
void LinearElasticMaterial<SpatialDim>::initialize()
/******************************************************************************/
{
    for (int i = 0; i < mNumVoigtTerms; i++)
        for (int j = 0; j < mNumVoigtTerms; j++)
            mCellStiffness(i, j) = 0.0;

    mPressureScaling = 1.0;

    for (int i = 0; i < mNumVoigtTerms; i++)
        mReferenceStrain(i) = 0.0;
}

/******************************************************************************/
template<int SpatialDim>
LinearElasticMaterial<SpatialDim>::LinearElasticMaterial()
/******************************************************************************/
{
    initialize();
}

/******************************************************************************/
template<int SpatialDim>
LinearElasticMaterial<SpatialDim>::LinearElasticMaterial(const Teuchos::ParameterList &paramList)
/******************************************************************************/

{
    initialize();

    if (paramList.isType < Plato::Scalar > ("e11"))
        mReferenceStrain(0) = paramList.get < Plato::Scalar > ("e11");
    if (paramList.isType < Plato::Scalar > ("e22"))
        mReferenceStrain(1) = paramList.get < Plato::Scalar > ("e22");
    if (paramList.isType < Plato::Scalar > ("e33"))
        mReferenceStrain(2) = paramList.get < Plato::Scalar > ("e33");
    if (paramList.isType < Plato::Scalar > ("e23"))
        mReferenceStrain(3) = paramList.get < Plato::Scalar > ("e23");
    if (paramList.isType < Plato::Scalar > ("e13"))
        mReferenceStrain(4) = paramList.get < Plato::Scalar > ("e13");
    if (paramList.isType < Plato::Scalar > ("e12"))
        mReferenceStrain(5) = paramList.get < Plato::Scalar > ("e12");
}

/******************************************************************************/
/*!
 \brief Derived class for isotropic linear elastic material model
 */
template<int SpatialDim>
class IsotropicLinearElasticMaterial: public LinearElasticMaterial<SpatialDim>
/******************************************************************************/
{
public:
    IsotropicLinearElasticMaterial(const Teuchos::ParameterList &paramList);
    IsotropicLinearElasticMaterial(const Plato::Scalar &aYoungsModulus,
                                   const Plato::Scalar &aPoissonsRatio);
    virtual ~IsotropicLinearElasticMaterial()
    {
    }

private:
    Plato::Scalar mPoissonsRatio;
    Plato::Scalar mYoungsModulus;
};
// class IsotropicLinearElasticMaterial

/******************************************************************************/
/*!
  \brief Derived class for cubic linear elastic material model
*/
template<int SpatialDim>
class CubicLinearElasticMaterial : public LinearElasticMaterial<SpatialDim>
/******************************************************************************/
{
  public:
    CubicLinearElasticMaterial(const Teuchos::ParameterList& paramList);
    virtual ~CubicLinearElasticMaterial(){}
};
// class CubicLinearElasticMaterial

/******************************************************************************/
/*!
  \brief Factory for creating material models
*/
template<int SpatialDim>
class ElasticModelFactory
/******************************************************************************/
{
public:
    ElasticModelFactory(const Teuchos::ParameterList &paramList) :
        mParamList(paramList)
    {
    }
    Teuchos::RCP<Plato::LinearElasticMaterial<SpatialDim>> create();
private:
    const Teuchos::ParameterList &mParamList;
};
// class ElasticModelFactory

/******************************************************************************/
template<int SpatialDim>
Teuchos::RCP<LinearElasticMaterial<SpatialDim>> ElasticModelFactory<SpatialDim>::create()
/******************************************************************************/
{
    auto modelParamList = mParamList.get < Teuchos::ParameterList > ("Material Model");

    if (modelParamList.isSublist("Isotropic Linear Elastic"))
    {
        return Teuchos::rcp(new Plato::IsotropicLinearElasticMaterial<SpatialDim>(modelParamList.sublist("Isotropic Linear Elastic")));
    }
    else
    if( modelParamList.isSublist("Cubic Linear Elastic") )
    {
        return Teuchos::rcp(new Plato::CubicLinearElasticMaterial<SpatialDim>(modelParamList.sublist("Cubic Linear Elastic")));
    }
    else
    {
        THROWERR("Input Material Model is NOT Defined.");
    }
}
=======
{

/******************************************************************************//**
 * \brief Base class for Linear Elastic material models
 *
 * \tparam SpatialDim spatial dimensions, options 1D, 2D, and 3D
 *
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class LinearElasticMaterial
{
protected:
    static constexpr auto mNumVoigtTerms = (SpatialDim == 3) ? 6 :
                                           ((SpatialDim == 2) ? 3 :
                                          (((SpatialDim == 1) ? 1 : 0)));
    static_assert(mNumVoigtTerms, "SpatialDim must be 1, 2, or 3."); /*!< number of stress-strain terms */

    Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;   /*!< cell stiffness matrix, i.e. fourth-order material tensor */
    Omega_h::Vector<mNumVoigtTerms> mReferenceStrain;                /*!< reference strain tensor */

    Plato::Scalar mCellDensity;     /*!< material density */
    Plato::Scalar mPressureScaling; /*!< pressure term scaling */

public:
    /******************************************************************************//**
     * \brief Linear elastic material model constructor.
    **********************************************************************************/
    LinearElasticMaterial();

    /******************************************************************************//**
     * \brief Linear elastic material model constructor.
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    LinearElasticMaterial(const Teuchos::ParameterList& aParamList);

    /******************************************************************************//**
     * \brief Return material density (mass unit/volume unit).
     * \return material density
    **********************************************************************************/
    decltype(mCellDensity)     getMassDensity()     const {return mCellDensity;}

    /******************************************************************************//**
     * \brief Return cell stiffness matrix.
     * \return cell stiffness matrix
    **********************************************************************************/
    decltype(mCellStiffness)   getStiffnessMatrix() const {return mCellStiffness;}

    /******************************************************************************//**
     * \brief Return pressure term scaling. Used in the stabilized finite element formulation
     * \return pressure term scaling
    **********************************************************************************/
    decltype(mPressureScaling) getPressureScaling() const {return mPressureScaling;}

    /******************************************************************************//**
     * \brief Return reference strain tensor, i.e. homogenized strain tensor.
     * \return reference strain tensor
    **********************************************************************************/
    decltype(mReferenceStrain) getReferenceStrain() const {return mReferenceStrain;}

private:
    /******************************************************************************//**
     * \brief Initialize member data to default values.
    **********************************************************************************/
    void initialize();

    /******************************************************************************//**
     * \brief Set reference strain tensor.
    **********************************************************************************/
    void setReferenceStrainTensor(const Teuchos::ParameterList& aParamList);
};
// class LinearElasticMaterial
>>>>>>> master

}
// namespace Plato

#endif

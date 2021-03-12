#ifndef LINEARELASTICMATERIAL_HPP
#define LINEARELASTICMATERIAL_HPP

#include <Omega_h_matrix.hpp>
#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
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

    Plato::Scalar mRayleighA; // mass coefficient
    Plato::Scalar mRayleighB; // stiffness coefficient

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

    decltype(mRayleighA)       getRayleighA()       const {return mRayleighA;}
    decltype(mRayleighB)       getRayleighB()       const {return mRayleighB;}

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

}
// namespace Plato

#endif

/*
 * LinearMaterialModelTests.cpp
 *
 *  Created on: Mar 23, 2020
 */

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

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
    void setOrthoMaterialModel(const Teuchos::ParameterList& aParamList);
    void checkOrthoMaterialInputs(const Teuchos::ParameterList& aParamList);
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


template<>
void Plato::OrthotropicLinearElasticMaterial<3>::checkOrthoMaterialStability
(const Teuchos::ParameterList& aParamList)
{
    // Stability Check 1: Positive material properties
    auto tYoungsModulusX = aParamList.get<Plato::Scalar>("Youngs Modulus X");
    if(tYoungsModulusX < static_cast<Plato::Scalar>(0)) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Youngs Modulus X' must be positive.")
    }

    auto tYoungsModulusY = aParamList.get<Plato::Scalar>("Youngs Modulus Y");
    if(tYoungsModulusY < static_cast<Plato::Scalar>(0)) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Youngs Modulus Y' must be positive.")
    }

    auto tYoungsModulusZ = aParamList.get<Plato::Scalar>("Youngs Modulus Z");
    if(tYoungsModulusZ < static_cast<Plato::Scalar>(0)) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Youngs Modulus Z' must be positive.")
    }

    auto tShearModulusXY = aParamList.get<Plato::Scalar>("Shear Modulus XY");
    if(tShearModulusXY < static_cast<Plato::Scalar>(0)) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Shear Modulus XY' must be positive.")
    }

    auto tShearModulusXZ = aParamList.get<Plato::Scalar>("Shear Modulus XZ");
    if(tShearModulusXZ < static_cast<Plato::Scalar>(0)) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Shear Modulus XZ' must be positive.")
    }

    auto tShearModulusYZ = aParamList.get<Plato::Scalar>("Shear Modulus YZ");
    if(tShearModulusYZ < static_cast<Plato::Scalar>(0)) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Shear Modulus YZ' must be positive.")
    }

    // Stability Check 2: Symmetry relationships
    auto tPoissonRatioXY = aParamList.get<Plato::Scalar>("Poissons Ratio XY");
    auto tSqrtYoungsModulusXOverYoungsModulusY = sqrt(tYoungsModulusX / tYoungsModulusY);
    if(abs(tPoissonRatioXY) > tSqrtYoungsModulusXOverYoungsModulusY)
    {
        std::stringstream tMsg;
        tMsg << "OrthotropicLinearElasticMaterial 3D: Condition abs(Poisson's Ratio XY) < sqrt(Young's Modulus X / Young's Modulus Y) "
                << "was not met.  The value of abs(Poisson's Ratio XY) = " << abs(tPoissonRatioXY) << " and the value of "
                << "sqrt(Young's Modulus X / Young's Modulus Y) = " << tSqrtYoungsModulusXOverYoungsModulusY << ".";
        THROWERR(tMsg.str().c_str())
    }

    auto tPoissonRatioYZ = aParamList.get<Plato::Scalar>("Poissons Ratio YZ");
    auto tSqrtYoungsModulusYOverYoungsModulusZ = sqrt(tYoungsModulusY / tYoungsModulusZ);
    if(abs(tPoissonRatioYZ) > tSqrtYoungsModulusYOverYoungsModulusZ)
    {
        std::stringstream tMsg;
        tMsg << "OrthotropicLinearElasticMaterial 3D: Condition abs(Poisson's Ratio YZ) < sqrt(Young's Modulus Y / Young's Modulus Z) "
                << "was not met.  The value of abs(Poisson's Ratio YZ) = " << abs(tPoissonRatioYZ) << " and the value of "
                << "sqrt(Young's Modulus Y / Young's Modulus Z) = " << tSqrtYoungsModulusYOverYoungsModulusZ << ".";
        THROWERR(tMsg.str().c_str())
    }

    auto tPoissonRatioXZ = aParamList.get<Plato::Scalar>("Poissons Ratio XZ");
    auto tSqrtYoungsModulusXOverYoungsModulusZ = sqrt(tYoungsModulusX / tYoungsModulusZ);
    if(abs(tPoissonRatioXZ) > tSqrtYoungsModulusXOverYoungsModulusZ)
    {
        std::stringstream tMsg;
        tMsg << "OrthotropicLinearElasticMaterial 3D: Condition abs(Poisson's Ratio XZ) < sqrt(Young's Modulus X / Young's Modulus Z) "
                << "was not met.  The value of abs(Poisson's Ratio XZ) = " << abs(tPoissonRatioXZ) << " and the value of "
                << "sqrt(Young's Modulus X / Young's Modulus Z) = " << tSqrtYoungsModulusXOverYoungsModulusZ << ".";
        THROWERR(tMsg.str().c_str())
    }

    auto tPoissonRatioYX = tPoissonRatioXY * (tYoungsModulusY / tYoungsModulusX);
    auto tSqrtYoungsModulusYOverYoungsModulusX = sqrt(tYoungsModulusY / tYoungsModulusX);
    if(abs(tPoissonRatioYX) > tSqrtYoungsModulusYOverYoungsModulusX)
    {
        std::stringstream tMsg;
        tMsg << "OrthotropicLinearElasticMaterial 3D: Condition abs(Poisson's Ratio YX) < sqrt(Young's Modulus Y / Young's Modulus X) "
                << "was not met.  The value of abs(Poisson's Ratio YX) = " << abs(tPoissonRatioYX) << " and the value of "
                << "sqrt(Young's Modulus Y / Young's Modulus X) = " << tSqrtYoungsModulusYOverYoungsModulusX << ".";
        THROWERR(tMsg.str().c_str())
    }

    auto tPoissonRatioZY = tPoissonRatioYZ * (tYoungsModulusZ / tYoungsModulusY);
    auto tSqrtYoungsModulusZOverYoungsModulusY = sqrt(tYoungsModulusZ / tYoungsModulusY);
    if(abs(tPoissonRatioZY) > tSqrtYoungsModulusZOverYoungsModulusY)
    {
        std::stringstream tMsg;
        tMsg << "OrthotropicLinearElasticMaterial 3D: Condition abs(Poisson's Ratio ZY) < sqrt(Young's Modulus Z / Young's Modulus Y) "
                << "was not met.  The value of abs(Poisson's Ratio ZY) = " << abs(tPoissonRatioZY) << " and the value of "
                << "sqrt(Young's Modulus Z / Young's Modulus Y) = " << tSqrtYoungsModulusZOverYoungsModulusY << ".";
        THROWERR(tMsg.str().c_str())
    }

    auto tPoissonRatioZX = tPoissonRatioXZ * (tYoungsModulusZ / tYoungsModulusX);
    auto tSqrtYoungsModulusZOverYoungsModulusX = sqrt(tYoungsModulusZ / tYoungsModulusX);
    if(abs(tPoissonRatioZX) > tSqrtYoungsModulusZOverYoungsModulusX)
    {
        std::stringstream tMsg;
        tMsg << "OrthotropicLinearElasticMaterial 3D: Condition abs(Poisson's Ratio ZX) < sqrt(Young's Modulus Z / Young's Modulus X) "
                << "was not met.  The value of abs(Poisson's Ratio ZX) = " << abs(tPoissonRatioZX) << " and the value of "
                << "sqrt(Young's Modulus Z / Young's Modulus X) = " << tSqrtYoungsModulusZOverYoungsModulusX << ".";
        THROWERR(tMsg.str().c_str())
    }

    auto tDetComplianceMat = static_cast<Plato::Scalar>(1.0) - (tPoissonRatioXY * tPoissonRatioYX)
            - (tPoissonRatioYZ * tPoissonRatioZY) - (tPoissonRatioXZ * tPoissonRatioZX)
            - static_cast<Plato::Scalar>(2.0) * (tPoissonRatioYX * tPoissonRatioZY * tPoissonRatioXZ);
    if(tDetComplianceMat < static_cast<Plato::Scalar>(0.0))
    {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Determinant of the compliance matrix is negative.")
    }
}

template<>
void Plato::OrthotropicLinearElasticMaterial<3>::checkOrthoMaterialInputs
(const Teuchos::ParameterList& aParamList)
{
    if(aParamList.isParameter("Youngs Modulus X") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Youngs Modulus X' is not defined.")
    }

    if(aParamList.isParameter("Youngs Modulus Y") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Youngs Modulus Y' is not defined.")
    }

    if(aParamList.isParameter("Youngs Modulus Z") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Youngs Modulus Z' is not defined.")
    }

    if(aParamList.isParameter("Shear Modulus XY") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Shear Modulus XY' is not defined.")
    }

    if(aParamList.isParameter("Shear Modulus XZ") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Shear Modulus XZ' is not defined.")
    }

    if(aParamList.isParameter("Shear Modulus YZ") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Shear Modulus YZ' is not defined.")
    }

    if(aParamList.isParameter("Poissons Ratio XY") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Poissons Ratio XY' is not defined.")
    }

    if(aParamList.isParameter("Poissons Ratio XZ") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Poissons Ratio XZ' is not defined.")
    }

    if(aParamList.isParameter("Poissons Ratio YZ") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 3D: Parameter Keyword 'Poissons Ratio YZ' is not defined.")
    }
}

template<>
void Plato::OrthotropicLinearElasticMaterial<3>::setOrthoMaterialModel
(const Teuchos::ParameterList& aParamList)
{
    mCellDensity = aParamList.isParameter("Mass Density") ? aParamList.get<Plato::Scalar>("Mass Density") : 1.0;

    auto tYoungsModulusX = aParamList.get<Plato::Scalar>("Youngs Modulus X");
    auto tYoungsModulusY = aParamList.get<Plato::Scalar>("Youngs Modulus Y");
    auto tYoungsModulusZ = aParamList.get<Plato::Scalar>("Youngs Modulus Z");
    auto tShearModulusXY = aParamList.get<Plato::Scalar>("Shear Modulus XY");
    auto tShearModulusXZ = aParamList.get<Plato::Scalar>("Shear Modulus XZ");
    auto tShearModulusYZ = aParamList.get<Plato::Scalar>("Shear Modulus YZ");
    auto tPoissonRatioXY = aParamList.get<Plato::Scalar>("Poissons Ratio XY");
    auto tPoissonRatioXZ = aParamList.get<Plato::Scalar>("Poissons Ratio XZ");
    auto tPoissonRatioYZ = aParamList.get<Plato::Scalar>("Poissons Ratio YZ");

    auto tPoissonRatioYX = tPoissonRatioXY * (tYoungsModulusY / tYoungsModulusX);
    auto tPoissonRatioZX = tPoissonRatioXZ * (tYoungsModulusZ / tYoungsModulusX);
    auto tPoissonRatioZY = tPoissonRatioYZ * (tYoungsModulusZ / tYoungsModulusY);
    auto tDeterminantComplianceMat = static_cast<Plato::Scalar>(1.0) - (tPoissonRatioXY * tPoissonRatioYX)
            - (tPoissonRatioYZ * tPoissonRatioZY) - (tPoissonRatioXZ * tPoissonRatioZX)
            - static_cast<Plato::Scalar>(2.0) * (tPoissonRatioYX * tPoissonRatioZY * tPoissonRatioXZ);
    auto tDelta = tDeterminantComplianceMat / (tYoungsModulusX * tYoungsModulusY * tYoungsModulusZ);

    // Row One
    auto tDenominator1 = tYoungsModulusY * tYoungsModulusZ * tDelta;
    mCellStiffness(0,0) = (static_cast<Plato::Scalar>(1.0) - (tPoissonRatioYZ * tPoissonRatioZY)) / tDenominator1;
    mCellStiffness(0,1) = (tPoissonRatioYX + (tPoissonRatioZX * tPoissonRatioYZ)) / tDenominator1;
    mCellStiffness(0,2) = (tPoissonRatioZX + (tPoissonRatioYX * tPoissonRatioZY)) / tDenominator1;

    // Row Two
    auto tDenominator2 = tYoungsModulusZ * tYoungsModulusX * tDelta;
    mCellStiffness(1,0) = mCellStiffness(0,1);
    mCellStiffness(1,1) = (static_cast<Plato::Scalar>(1.0) - (tPoissonRatioZX * tPoissonRatioXZ)) / tDenominator2;
    mCellStiffness(1,2) = (tPoissonRatioZY + (tPoissonRatioZX * tPoissonRatioXY)) / tDenominator2;

    // Row Three
    auto tDenominator3 = tYoungsModulusX * tYoungsModulusY * tDelta;
    mCellStiffness(2,0) = mCellStiffness(0,2);
    mCellStiffness(2,1) = mCellStiffness(1,2);
    mCellStiffness(2,2) = (static_cast<Plato::Scalar>(1.0) - (tPoissonRatioXY * tPoissonRatioYX)) / tDenominator3;

    // Shear Terms
    mCellStiffness(3,3) = tShearModulusYZ;
    mCellStiffness(4,4) = tShearModulusXZ;
    mCellStiffness(5,5) = tShearModulusXY;
}

template<>
Plato::OrthotropicLinearElasticMaterial<3>::
OrthotropicLinearElasticMaterial(const Teuchos::ParameterList& aParamList) :
     Plato::LinearElasticMaterial<3>(aParamList)
{
    this->checkOrthoMaterialInputs(aParamList);
    this->checkOrthoMaterialStability(aParamList);
    this->setOrthoMaterialModel(aParamList);
}

template<>
void Plato::OrthotropicLinearElasticMaterial<3>::setMaterialModel
(const Teuchos::ParameterList& aParamList)
{
    this->checkOrthoMaterialInputs(aParamList);
    this->checkOrthoMaterialStability(aParamList);
    this->setOrthoMaterialModel(aParamList);
}

template<>
void Plato::OrthotropicLinearElasticMaterial<2>::checkOrthoMaterialStability
(const Teuchos::ParameterList& aParamList)
{
    // Stability Check 1: Positive material properties
    auto tYoungsModulusX = aParamList.get<Plato::Scalar>("Youngs Modulus X");
    if(tYoungsModulusX < static_cast<Plato::Scalar>(0)) {
        THROWERR("OrthotropicLinearElasticMaterial 2D: Parameter Keyword 'Youngs Modulus X' must be positive.")
    }

    auto tYoungsModulusY = aParamList.get<Plato::Scalar>("Youngs Modulus Y");
    if(tYoungsModulusY < static_cast<Plato::Scalar>(0)) {
        THROWERR("OrthotropicLinearElasticMaterial 2D: Parameter Keyword 'Youngs Modulus Y' must be positive.")
    }

    auto tShearModulusXY = aParamList.get<Plato::Scalar>("Shear Modulus XY");
    if(tShearModulusXY < static_cast<Plato::Scalar>(0)) {
        THROWERR("OrthotropicLinearElasticMaterial 2D: Parameter Keyword 'Shear Modulus XY' must be positive.")
    }

    // Stability Check 2: Symmetry relationships
    auto tPoissonRatioXY = aParamList.get<Plato::Scalar>("Poissons Ratio XY");
    auto tSqrtYoungsModulusXOverYoungsModulusY = sqrt(tYoungsModulusX / tYoungsModulusY);
    if(abs(tPoissonRatioXY) > tSqrtYoungsModulusXOverYoungsModulusY)
    {
        std::stringstream tMsg;
        tMsg << "OrthotropicLinearElasticMaterial 2D: Condition abs(Poisson's Ratio XY) < sqrt(Young's Modulus X / Young's Modulus Y) "
                << "was not met.  The value of abs(Poisson's Ratio XY) = " << abs(tPoissonRatioXY) << " and the value of "
                << "sqrt(Young's Modulus X / Young's Modulus Y) = " << tSqrtYoungsModulusXOverYoungsModulusY << ".";
        THROWERR(tMsg.str().c_str())
    }

    auto tPoissonRatioYX = tPoissonRatioXY * (tYoungsModulusY / tYoungsModulusX);
    auto tSqrtYoungsModulusYOverYoungsModulusX = sqrt(tYoungsModulusY / tYoungsModulusX);
    if(abs(tPoissonRatioYX) > tSqrtYoungsModulusYOverYoungsModulusX)
    {
        std::stringstream tMsg;
        tMsg << "OrthotropicLinearElasticMaterial 2D: Condition abs(Poisson's Ratio YX) < sqrt(Young's Modulus Y / Young's Modulus X) "
                << "was not met.  The value of abs(Poisson's Ratio YX) = " << abs(tPoissonRatioYX) << " and the value of "
                << "sqrt(Young's Modulus Y / Young's Modulus X) = " << tSqrtYoungsModulusYOverYoungsModulusX << ".";
        THROWERR(tMsg.str().c_str())
    }
}

template<>
void Plato::OrthotropicLinearElasticMaterial<2>::checkOrthoMaterialInputs
(const Teuchos::ParameterList& aParamList)
{
    if(aParamList.isParameter("Youngs Modulus X") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 2D: Parameter Keyword 'Youngs Modulus X' is not defined.")
    }

    if(aParamList.isParameter("Youngs Modulus Y") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 2D: Parameter Keyword 'Youngs Modulus Y' is not defined.")
    }

    if(aParamList.isParameter("Shear Modulus XY") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 2D: Parameter Keyword 'Shear Modulus XY' is not defined.")
    }

    if(aParamList.isParameter("Poissons Ratio XY") == false) {
        THROWERR("OrthotropicLinearElasticMaterial 2D: Parameter Keyword 'Poissons Ratio XY' is not defined.")
    }
}

template<>
void Plato::OrthotropicLinearElasticMaterial<2>::setOrthoMaterialModel
(const Teuchos::ParameterList& aParamList)
{
    mCellDensity = aParamList.isParameter("Mass Density") ? aParamList.get<Plato::Scalar>("Mass Density") : 1.0;

    auto tYoungsModulusX = aParamList.get<Plato::Scalar>("Youngs Modulus X");
    auto tYoungsModulusY = aParamList.get<Plato::Scalar>("Youngs Modulus Y");
    auto tShearModulusXY = aParamList.get<Plato::Scalar>("Shear Modulus XY");
    auto tPoissonRatioXY = aParamList.get<Plato::Scalar>("Poissons Ratio XY");

    auto tPoissonRatioYX = tPoissonRatioXY * (tYoungsModulusY / tYoungsModulusX);
    mCellStiffness(0,0) = tYoungsModulusX / (static_cast<Plato::Scalar>(1.0) - (tPoissonRatioXY * tPoissonRatioYX));
    mCellStiffness(0,1) = (tPoissonRatioXY * tYoungsModulusY) / (static_cast<Plato::Scalar>(1.0) - (tPoissonRatioXY * tPoissonRatioYX));
    mCellStiffness(1,0) = (tPoissonRatioXY * tYoungsModulusY) / (static_cast<Plato::Scalar>(1.0) - (tPoissonRatioXY * tPoissonRatioYX));
    mCellStiffness(1,1) = tYoungsModulusY / (static_cast<Plato::Scalar>(1.0) - (tPoissonRatioXY * tPoissonRatioYX));
    mCellStiffness(2,2) = tShearModulusXY;
}

template<>
Plato::OrthotropicLinearElasticMaterial<2>::
OrthotropicLinearElasticMaterial(const Teuchos::ParameterList& aParamList) :
     Plato::LinearElasticMaterial<2>(aParamList)
{
    this->checkOrthoMaterialInputs(aParamList);
    this->checkOrthoMaterialStability(aParamList);
    this->setOrthoMaterialModel(aParamList);
}

template<>
void Plato::OrthotropicLinearElasticMaterial<2>::setMaterialModel
(const Teuchos::ParameterList& aParamList)
{
    this->checkOrthoMaterialInputs(aParamList);
    this->checkOrthoMaterialStability(aParamList);
    this->setOrthoMaterialModel(aParamList);
}

template<>
void Plato::OrthotropicLinearElasticMaterial<1>::checkOrthoMaterialStability
(const Teuchos::ParameterList& aParamList)
{ return; }

template<>
void Plato::OrthotropicLinearElasticMaterial<1>::checkOrthoMaterialInputs
(const Teuchos::ParameterList& aParamList)
{
    if(aParamList.isParameter("Poissons Ratio") == false){
        THROWERR("OrthotropicLinearElasticMaterial 1D: Parameter Keyword 'Poissons Ratio' is not defined.")
    }

    if(aParamList.isParameter("Youngs Modulus") == false){
        THROWERR("OrthotropicLinearElasticMaterial 1D: Parameter Keyword 'Youngs Modulus' is not defined.")
    }
}

template<>
void Plato::OrthotropicLinearElasticMaterial<1>::setOrthoMaterialModel
(const Teuchos::ParameterList& aParamList)
{
    mCellDensity = aParamList.isParameter("Mass Density") ? aParamList.get<Plato::Scalar>("Mass Density") : 1.0;

    auto tPoissonRatio = aParamList.get<Plato::Scalar>("Poissons Ratio");
    auto tYoungsModulus = aParamList.get<Plato::Scalar>("Youngs Modulus");
    auto tStiffCoeff = tYoungsModulus / ( (static_cast<Plato::Scalar>(1.0) + tPoissonRatio)
            * (static_cast<Plato::Scalar>(1.0) - static_cast<Plato::Scalar>(2.0) * tPoissonRatio));
    mCellStiffness(0, 0) = tStiffCoeff * (static_cast<Plato::Scalar>(1.0) - tPoissonRatio);
}

template<>
Plato::OrthotropicLinearElasticMaterial<1>::
OrthotropicLinearElasticMaterial(const Teuchos::ParameterList& aParamList) :
     Plato::LinearElasticMaterial<1>(aParamList)
{
    this->checkOrthoMaterialInputs(aParamList);
    this->checkOrthoMaterialStability(aParamList);
    this->setOrthoMaterialModel(aParamList);
}

template<>
void Plato::OrthotropicLinearElasticMaterial<1>::setMaterialModel
(const Teuchos::ParameterList& aParamList)
{
    this->checkOrthoMaterialInputs(aParamList);
    this->checkOrthoMaterialStability(aParamList);
    this->setOrthoMaterialModel(aParamList);
}

}
// namespace Plato

namespace OrthotropicLinearElasticMaterialTest
{

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic1D)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                 \n"
      "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus' type='double' value='1.0'/>   \n"
      "    </ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 1;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat(*tParams);
    auto tStiffMatrix = tOrthoMat.getStiffnessMatrix();
    const Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(1.3461538, tStiffMatrix(0,0), tTolerance);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic1D_PoissonRatioKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "<ParameterList name='Orthotropic Linear Elastic'>                 \n"
      "  <Parameter  name='Poissons Rtio' type='double' value='0.3'/>  \n"
      "  <Parameter  name='Youngs Modulus' type='double' value='1.0'/>   \n"
      "</ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 1;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic1D_YoungModulusKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                 \n"
      "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Moulus' type='double' value='1.0'/>  \n"
      "    </ParameterList>                                                  \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 1;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='1.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='0.8'/>   \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat(*tParams);
    auto tStiffMatrix = tOrthoMat.getStiffnessMatrix();
    const Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(1.0775862, tStiffMatrix(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.2586206, tStiffMatrix(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,       tStiffMatrix(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.2586206, tStiffMatrix(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.8620689, tStiffMatrix(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,       tStiffMatrix(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,       tStiffMatrix(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,       tStiffMatrix(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.3,       tStiffMatrix(2,2), tTolerance);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_PoissonRatioKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissos Ratio XY' type='double' value='0.3'/> \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='1.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='1.0'/>   \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_ShearModulusKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Shear Modulu XY' type='double' value='0.3'/>    \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='1.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='1.0'/>   \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_YoungsModulusXKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulu X' type='double' value='1.0'/>    \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='1.0'/>   \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_YoungsModulusYKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='1.0'/>   \n"
      "      <Parameter  name='Youngs Modulu Y' type='double' value='1.0'/>    \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_YoungsModulusXCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='-1.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='1.0'/>   \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_YoungsModulusYCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.3'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='1.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='-1.0'/>  \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic2D_PoissonRatioXYCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                   \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.75'/> \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='0.3'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='1.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='2.0'/>   \n"
      "    </ParameterList>                                                    \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 2;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat(*tParams);
    auto tStiffMatrix = tOrthoMat.getStiffnessMatrix();
    const Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(128.942, tStiffMatrix(0,0), tTolerance);
    TEST_FLOATING_EQUALITY(5.253,   tStiffMatrix(0,1), tTolerance);
    TEST_FLOATING_EQUALITY(5.253,   tStiffMatrix(0,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffMatrix(0,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffMatrix(0,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,     tStiffMatrix(0,5), tTolerance);

    TEST_FLOATING_EQUALITY(5.253,  tStiffMatrix(1,0), tTolerance);
    TEST_FLOATING_EQUALITY(13.309, tStiffMatrix(1,1), tTolerance);
    TEST_FLOATING_EQUALITY(5.452,  tStiffMatrix(1,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffMatrix(1,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffMatrix(1,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffMatrix(1,5), tTolerance);

    TEST_FLOATING_EQUALITY(5.253,  tStiffMatrix(2,0), tTolerance);
    TEST_FLOATING_EQUALITY(5.452,  tStiffMatrix(2,1), tTolerance);
    TEST_FLOATING_EQUALITY(13.309, tStiffMatrix(2,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffMatrix(2,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffMatrix(2,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,    tStiffMatrix(2,5), tTolerance);

    TEST_FLOATING_EQUALITY(0.0,   tStiffMatrix(3,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,   tStiffMatrix(3,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,   tStiffMatrix(3,2), tTolerance);
    TEST_FLOATING_EQUALITY(3.928, tStiffMatrix(3,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,   tStiffMatrix(3,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0,   tStiffMatrix(3,5), tTolerance);

    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(4,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(4,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(4,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(4,3), tTolerance);
    TEST_FLOATING_EQUALITY(6.6, tStiffMatrix(4,4), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(4,5), tTolerance);

    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(5,0), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(5,1), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(5,2), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(5,3), tTolerance);
    TEST_FLOATING_EQUALITY(0.0, tStiffMatrix(5,4), tTolerance);
    TEST_FLOATING_EQUALITY(6.6, tStiffMatrix(5,5), tTolerance);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_YoungsModulusXKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulu X' type='double' value='126.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "</ParameterList>                                                         \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_YoungsModulusYKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulu Y' type='double' value='11.0'/>    \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_YoungsModulusZKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulu Z' type='double' value='11.0'/>    \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_ShearModulusYZKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulu YZ' type='double' value='3.928'/>   \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_ShearModulusXZKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulu XZ' type='double' value='6.60'/>    \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_ShearModulusXYKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulu XY' type='double' value='6.60'/>    \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_PoissonsRatioXYKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Rati XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_PoissonsRatioXZKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio YZ' type='double' value='0.40'/>  \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_PoissonsRatioYZKeyword_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

// TODO: Stability Conditions - Unit Tests

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_YoungsModulusXCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='-126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_YoungsModulusYCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='-11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_YoungsModulusZCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='-11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_ShearModulusXYCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='-6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_ShearModulusXZCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='-6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_ShearModulusYZCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='-3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>   \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_PoissonRationXYCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='3.40'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>  \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_PoissonRationXZCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='3.40'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='0.40'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>  \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

TEUCHOS_UNIT_TEST(LinearElasticMaterialTest, Orthotropic3D_PoissonRationYZCondition_Error)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
      Teuchos::getParametersFromXmlString(
      "    <ParameterList name='Orthotropic Linear Elastic'>                    \n"
      "      <Parameter  name='Poissons Ratio XY' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Ratio XZ' type='double' value='0.28'/>  \n"
      "      <Parameter  name='Poissons Rati YZ' type='double' value='2.0'/>   \n"
      "      <Parameter  name='Shear Modulus XY' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus XZ' type='double' value='6.60'/>   \n"
      "      <Parameter  name='Shear Modulus YZ' type='double' value='3.928'/>  \n"
      "      <Parameter  name='Youngs Modulus X' type='double' value='126.0'/>  \n"
      "      <Parameter  name='Youngs Modulus Y' type='double' value='11.0'/>   \n"
      "      <Parameter  name='Youngs Modulus Z' type='double' value='11.0'/>  \n"
      "    </ParameterList>                                                     \n"
    );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    Plato::OrthotropicLinearElasticMaterial<tSpaceDim> tOrthoMat;
    TEST_THROW(tOrthoMat.setMaterialModel(*tParams), std::runtime_error);
}

}
// namespace OrthotropicLinearElasticMaterialTest



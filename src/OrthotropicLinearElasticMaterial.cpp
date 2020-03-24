/*
 * OrthotropicLinearElasticMaterial.cpp
 *
 *  Created on: Mar 24, 2020
 */

#include "OrthotropicLinearElasticMaterial.hpp"


namespace Plato
{

//*********************************************************************************
//**************************** NEXT: 3D Implementation ****************************
//*********************************************************************************

/******************************************************************************//**
 * \brief Check linear elastic orthotropic material stability constants - 3D
**********************************************************************************/
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

/******************************************************************************//**
 * \brief Check linear elastic orthotropic material input parameters - 3D
**********************************************************************************/
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

/******************************************************************************//**
 * \brief Set linear elastic orthotropic material constants - 3D
**********************************************************************************/
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

/******************************************************************************//**
 * \brief Linear elastic orthotropic material model constructor. - 3D
**********************************************************************************/
template<>
Plato::OrthotropicLinearElasticMaterial<3>::
OrthotropicLinearElasticMaterial(const Teuchos::ParameterList& aParamList) :
     Plato::LinearElasticMaterial<3>(aParamList)
{
    this->checkOrthoMaterialInputs(aParamList);
    this->checkOrthoMaterialStability(aParamList);
    this->setOrthoMaterialModel(aParamList);
}

/******************************************************************************//**
 * \brief Initialize linear elastic orthotropic material model. - 3D
**********************************************************************************/
template<>
void Plato::OrthotropicLinearElasticMaterial<3>::setMaterialModel
(const Teuchos::ParameterList& aParamList)
{
    this->checkOrthoMaterialInputs(aParamList);
    this->checkOrthoMaterialStability(aParamList);
    this->setOrthoMaterialModel(aParamList);
}

//*********************************************************************************
//**************************** NEXT: 2D Implementation ****************************
//*********************************************************************************

/******************************************************************************//**
 * \brief Check linear elastic orthotropic material stability constants - 2D
**********************************************************************************/
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

/******************************************************************************//**
 * \brief Check linear elastic orthotropic material input parameters - 2D
**********************************************************************************/
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

/******************************************************************************//**
 * \brief Set linear elastic orthotropic material constants - 2D
**********************************************************************************/
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

/******************************************************************************//**
 * \brief Linear elastic orthotropic material model constructor. - 2D
**********************************************************************************/
template<>
Plato::OrthotropicLinearElasticMaterial<2>::
OrthotropicLinearElasticMaterial(const Teuchos::ParameterList& aParamList) :
     Plato::LinearElasticMaterial<2>(aParamList)
{
    this->checkOrthoMaterialInputs(aParamList);
    this->checkOrthoMaterialStability(aParamList);
    this->setOrthoMaterialModel(aParamList);
}

/******************************************************************************//**
 * \brief Initialize linear elastic orthotropic material model. - 2D
**********************************************************************************/
template<>
void Plato::OrthotropicLinearElasticMaterial<2>::setMaterialModel
(const Teuchos::ParameterList& aParamList)
{
    this->checkOrthoMaterialInputs(aParamList);
    this->checkOrthoMaterialStability(aParamList);
    this->setOrthoMaterialModel(aParamList);
}

//*********************************************************************************
//**************************** NEXT: 1D Implementation ****************************
//*********************************************************************************

/******************************************************************************//**
 * \brief Check linear elastic orthotropic material stability constants - 1D
**********************************************************************************/
template<>
void Plato::OrthotropicLinearElasticMaterial<1>::checkOrthoMaterialStability
(const Teuchos::ParameterList& aParamList)
{ return; }

/******************************************************************************//**
 * \brief Check linear elastic orthotropic material input parameters - 1D
**********************************************************************************/
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

/******************************************************************************//**
 * \brief Set linear elastic orthotropic material constants - 1D
**********************************************************************************/
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

/******************************************************************************//**
 * \brief Linear elastic orthotropic material model constructor. - 1D
**********************************************************************************/
template<>
Plato::OrthotropicLinearElasticMaterial<1>::
OrthotropicLinearElasticMaterial(const Teuchos::ParameterList& aParamList) :
     Plato::LinearElasticMaterial<1>(aParamList)
{
    this->checkOrthoMaterialInputs(aParamList);
    this->checkOrthoMaterialStability(aParamList);
    this->setOrthoMaterialModel(aParamList);
}

/******************************************************************************//**
 * \brief Initialize linear elastic orthotropic material model. - 1D
**********************************************************************************/
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

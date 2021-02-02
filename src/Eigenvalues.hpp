#pragma once

#include "PlatoStaticsTypes.hpp"
#include "PlatoTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Compute Eigenvalues
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class Eigenvalues
{
private:
    Plato::OrdinalType mNumFixedIterations;

public:
    /******************************************************************************//**
     * \brief Constructor
    **********************************************************************************/
    Eigenvalues(Plato::OrdinalType aNumFixedIterations = 6) :
    mNumFixedIterations(aNumFixedIterations)
    {
    }

    /******************************************************************************//**
     * \brief Set fixed number of jacobi iterations
     * \param [in] aNumFixedIterations fixed number of iterations for jacobi eigenvalue solver
    **********************************************************************************/
    void setNumFixedIterations(const Plato::Scalar aNumFixedIterations)
    {
        mNumFixedIterations = aNumFixedIterations;
    }

    /******************************************************************************//**
     * \brief Compute Eigenvalues
     * \param [in] aCellOrdinal cell/element index
     * \param [in] aVoigtTensor cell/element voigt tensor
     * \param [in] aIsStrainType engineering factor - divide shear terms by 2
     * \param [out] aEigenvalues cell/element tensor eigenvalues
    **********************************************************************************/
    template<typename ResultType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<ResultType> & aVoigtTensor,
               const Plato::ScalarMultiVectorT<ResultType> & aEigenvalues,
               const bool & aIsStrainType) const;
};
// class Eigenvalues

/******************************************************************************//**
 * \brief Eigenvalues for 3D problems
 *
 * \param [in] aCellOrdinal cell/element local ordinal
 * \param [in] aVoigtTensor cell/element voigt tensor
 * \param [in] aIsStrainType engineering factor - divide shear terms by 2
 * \param [out] aEigenvalues cell/element tensor eigenvalues
**********************************************************************************/
template<>
template<typename ResultType>
DEVICE_TYPE inline void
Eigenvalues<3>::operator()(const Plato::OrdinalType & aCellOrdinal,
                           const Plato::ScalarMultiVectorT<ResultType> & aVoigtTensor,
                           const Plato::ScalarMultiVectorT<ResultType> & aEigenvalues,
                           const bool & aIsStrainType) const
{
    constexpr Plato::Scalar tZeroScalar      = static_cast<Plato::Scalar>(0.0);
    constexpr Plato::Scalar tOneScalar       = static_cast<Plato::Scalar>(1.0);
    constexpr Plato::Scalar tMinusOneScalar  = static_cast<Plato::Scalar>(-1.0);
    constexpr Plato::Scalar tTwoScalar       = static_cast<Plato::Scalar>(2.0);

    ResultType tTensor[3][3];

    // Fill diagonal elements
    tTensor[0][0] = aVoigtTensor(aCellOrdinal, 0);
    tTensor[1][1] = aVoigtTensor(aCellOrdinal, 1);
    tTensor[2][2] = aVoigtTensor(aCellOrdinal, 2);

    // Fill off diagonal elements
    tTensor[0][1] = aVoigtTensor(aCellOrdinal, 5);
    if (aIsStrainType)
        tTensor[0][1] /= tTwoScalar;
    
    tTensor[0][2] = aVoigtTensor(aCellOrdinal, 4);  
    if (aIsStrainType)
        tTensor[0][2] /= tTwoScalar;

    tTensor[1][2] = aVoigtTensor(aCellOrdinal, 3);
    if (aIsStrainType)
        tTensor[1][2] /= tTwoScalar;
        

    // Symmetrize
    tTensor[1][0] = tTensor[0][1]; 
    tTensor[2][0] = tTensor[0][2]; 
    tTensor[2][1] = tTensor[1][2];

    ResultType tGivensRotation[3][3];
    ResultType tTensorRotated[3][3];
    ResultType tSine, tCosine, tTangent, tTau, tTemp01, tTemp02, tTemp12;

    constexpr Plato::OrdinalType tZero = static_cast<Plato::OrdinalType>(0);
    constexpr Plato::OrdinalType tOne  = static_cast<Plato::OrdinalType>(1);
    constexpr Plato::OrdinalType tTwo  = static_cast<Plato::OrdinalType>(2);

    Plato::OrdinalType i, j, k, l, p, q;
    for (Plato::OrdinalType tIteration = 0; tIteration < mNumFixedIterations; ++tIteration)
    {
        tTemp01 = fabs(tTensor[0][1]);
        tTemp02 = fabs(tTensor[0][2]);
        tTemp12 = fabs(tTensor[1][2]);
        auto tCondition1 = tTemp02 >= tTemp12;
        auto tCondition2 = tTemp01 >= tTemp02;
        auto tCondition3 = tTemp01 >= tTemp12;
        // ########## Compute location of max off-diagonal entry ##########
        auto tCondition4 = !(tCondition1 || tCondition3);
        p = tCondition4 ? tOne : tZero;
        auto tCondition5 =  (tCondition2 && tCondition3);
        q = tCondition5 ? tOne : tTwo;

        // ########## Compute rotation sine and cosine ##########
        auto tCondition6 = fabs(tTensor[p][q]) > 1.0e-15;
        tTau = tZeroScalar;
        if (tCondition6)
            tTau = (tTensor[q][q] - tTensor[p][p]) / (tTwoScalar * tTensor[p][q]);

        tTangent = tMinusOneScalar / (sqrt(tOneScalar + tTau*tTau) - tTau);
        if (tTau >= tZeroScalar)
            tTangent = tOneScalar / (tTau + sqrt(tOneScalar + tTau*tTau));

        tCosine = tOneScalar;
        if (tCondition6)
            tCosine = tOneScalar / sqrt(tOneScalar + tTangent*tTangent);

        tSine = tZeroScalar;
        if (tCondition6)
            tSine = tTangent * tCosine;

        // ########## Apply similarity transform with Givens rotation ##########
        tGivensRotation[0][0] =  tOneScalar; tGivensRotation[0][1] = tZeroScalar; tGivensRotation[0][2] = tZeroScalar;
        tGivensRotation[1][0] = tZeroScalar; tGivensRotation[1][1] =  tOneScalar; tGivensRotation[1][2] = tZeroScalar;
        tGivensRotation[2][0] = tZeroScalar; tGivensRotation[2][1] = tZeroScalar; tGivensRotation[2][2] =  tOneScalar;

        tGivensRotation[p][p] = tCosine;  tGivensRotation[p][q] =   tSine;
        tGivensRotation[q][p] =  -tSine;  tGivensRotation[q][q] = tCosine;

        for (i = 0; i < 3; ++i)
            for (l = i; l < 3; ++l) // Note that symmetry is being employed for speed
            {
                tTensorRotated[i][l] = tZeroScalar;
                for (j = 0; j < 3; ++j)
                    for (k = 0; k < 3; ++k)
                        tTensorRotated[i][l] += tGivensRotation[j][i] * tTensor[j][k] * tGivensRotation[k][l];

                tTensorRotated[l][i] = tTensorRotated[i][l];
            }

        for (i = 0; i < 3; ++i)
            for (j = 0; j < 3; ++j)
                tTensor[i][j] = tTensorRotated[i][j];
    }

    aEigenvalues(aCellOrdinal, 0) = tTensor[0][0];
    aEigenvalues(aCellOrdinal, 1) = tTensor[1][1];
    aEigenvalues(aCellOrdinal, 2) = tTensor[2][2];
}

/******************************************************************************//**
 * \brief Eigenvalues for 2D problems
 *
 * \param [in] aCellOrdinal cell/element local ordinal
 * \param [in] aVoigtTensor cell/element voigt tensor
 * \param [in] aIsStrainType engineering factor - divide shear terms by 2
 * \param [out] aEigenvalues cell/element tensor eigenvalues
**********************************************************************************/
template<>
template<typename ResultType>
DEVICE_TYPE inline void
Eigenvalues<2>::operator()(const Plato::OrdinalType & aCellOrdinal,
                           const Plato::ScalarMultiVectorT<ResultType> & aVoigtTensor,
                           const Plato::ScalarMultiVectorT<ResultType> & aEigenvalues,
                           const bool & aIsStrainType) const
{
    constexpr Plato::Scalar tZero      = static_cast<Plato::Scalar>(0.0);
    constexpr Plato::Scalar tOne       = static_cast<Plato::Scalar>(1.0);
    constexpr Plato::Scalar tMinusOne  = static_cast<Plato::Scalar>(-1.0);
    constexpr Plato::Scalar tTwo       = static_cast<Plato::Scalar>(2.0);

    ResultType tTensor12 = aVoigtTensor(aCellOrdinal, 2);
    if (aIsStrainType)
        tTensor12 /= tTwo;

    ResultType tTensor11 = aVoigtTensor(aCellOrdinal, 0);
    ResultType tTensor22 = aVoigtTensor(aCellOrdinal, 1);
    ResultType tTau = tZero;
    if (tTensor12 > tZero)
        tTau = (tTensor22 - tTensor11) / (tTwo * tTensor12);

    ResultType tTangent = tMinusOne / (sqrt(tOne + tTau*tTau) - tTau);
    if (tTau >= tZero)
        tTangent = tOne / (tTau + sqrt(tOne + tTau*tTau));

    ResultType tCosine = tOne / sqrt(tOne + tTangent*tTangent);
    ResultType tSine   = tTangent * tCosine;
    aEigenvalues(aCellOrdinal, 0) = tCosine * (tTensor11*tCosine - tTensor12*tSine) -
                                    tSine   * (tTensor12*tCosine - tTensor22*tSine);
    aEigenvalues(aCellOrdinal, 1) = tCosine * (tTensor22*tCosine + tTensor12*tSine) +
                                    tSine   * (tTensor12*tCosine + tTensor11*tSine);
}

/******************************************************************************//**
 * \brief Eigenvalues for 1D problems
 *
 * \param [in] aCellOrdinal cell/element local ordinal
 * \param [in] aVoigtTensor cell/element voigt tensor
 * \param [in] aIsStrainType engineering factor - divide shear terms by 2
 * \param [out] aEigenvalues cell/element tensor eigenvalues
**********************************************************************************/
template<>
template<typename ResultType>
DEVICE_TYPE inline void
Eigenvalues<1>::operator()(const Plato::OrdinalType & aCellOrdinal,
                           const Plato::ScalarMultiVectorT<ResultType> & aVoigtTensor,
                           const Plato::ScalarMultiVectorT<ResultType> & aEigenvalues,
                           const bool & aIsStrainType) const
{
    aEigenvalues(aCellOrdinal, 0) = aVoigtTensor(aCellOrdinal, 0);
}

}
// namespace Plato

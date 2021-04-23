/*
 * ComputeCauchyStress.hpp
 *
 *  Created on: Apr 6, 2020
 */

#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 *
 * \tparam SpaceDim number of spatial dimensions
 *
 * \brief Compute the Cauchy stress tensor:
 *
 * \f$ \sigma_{ij} = \mu(\epsilon_{ij} + \epsilon_{ji}) + \lambda\delta{ij}\epsilon_{kk} \f$,
 * where
 * \f$ \lambda = K-\frac{2\mu}{3} \f$
 *
 * Here, \f$ \epsilon_{ij} \f$ is the strain tensor, \f$ \mu \f$ is the shear modulus,
 * \f$ \epsilon_{kk} \f$ is the trace of the strain tensor, and K is the bulk modulus.
 *
**********************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeCauchyStress
{
public:
    /******************************************************************************//**
     * \brief Functor constructor
    **********************************************************************************/
    ComputeCauchyStress(){}

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor:
     *
     * \param [in]  aCellOrdinal           cell/element index
     * \param [in]  aPenalizedBulkModulus  penalized elastic bulk modulus
     * \param [in]  aPenalizedShearModulus penalized elastic shear modulus
     * \param [in]  aStrain                elastic strain tensor
     * \param [out] aStress                Cauchy stress tensor
    **********************************************************************************/
    template<typename StrainT, typename ControlT, typename StressT>
    DEVICE_TYPE
    inline void operator()
    (const Plato::OrdinalType & aCellOrdinal,
     const ControlT & aPenalizedBulkModulus,
     const ControlT & aPenalizedShearModulus,
     const Plato::ScalarMultiVectorT<StrainT> & aStrain,
     const Plato::ScalarMultiVectorT<StressT> & aStress) const;
};
// class ComputeCauchyStress

/******************************************************************************//**
 * \brief Compute the Cauchy stress for 3D
**********************************************************************************/
template<>
template<typename StrainT, typename ControlT, typename StressT>
DEVICE_TYPE inline void
ComputeCauchyStress<3>::operator()
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPenalizedBulkModulus,
 const ControlT & aPenalizedShearModulus,
 const Plato::ScalarMultiVectorT<StrainT> & aStrain,
 const Plato::ScalarMultiVectorT<StressT> & aStress) const
{
    // compute hydrostatic strain
    StrainT tTrace = aStrain(aCellOrdinal, 0) + aStrain(aCellOrdinal, 1) + aStrain(aCellOrdinal, 3);
    StrainT tTraceOver3 = tTrace / static_cast<Plato::Scalar>(3.0);

    // compute normal stress components
    aStress(aCellOrdinal, 0) = (static_cast<Plato::Scalar>(2.0) * aPenalizedShearModulus)
                                   * (aStrain(aCellOrdinal, 0) - tTraceOver3);
    aStress(aCellOrdinal, 1) = (static_cast<Plato::Scalar>(2.0) * aPenalizedShearModulus)
                                   * (aStrain(aCellOrdinal, 1) - tTraceOver3);
    aStress(aCellOrdinal, 2) = (static_cast<Plato::Scalar>(2.0) * aPenalizedShearModulus)
                                   * (aStrain(aCellOrdinal, 2) - tTraceOver3);

    // add hydrostatic stress to normal components
    aStress(aCellOrdinal, 0) += aPenalizedBulkModulus * tTrace;
    aStress(aCellOrdinal, 1) += aPenalizedBulkModulus * tTrace;
    aStress(aCellOrdinal, 2) += aPenalizedBulkModulus * tTrace;

    // compute shear components - elastic strain already has 2 multiplier, see equation in function declaration
    aStress(aCellOrdinal, 3) = aPenalizedShearModulus * aStrain(aCellOrdinal, 3);
    aStress(aCellOrdinal, 4) = aPenalizedShearModulus * aStrain(aCellOrdinal, 4);
    aStress(aCellOrdinal, 5) = aPenalizedShearModulus * aStrain(aCellOrdinal, 5);
}

/******************************************************************************//**
 * \brief Compute the Cauchy stress for 2D - Plane strain
**********************************************************************************/
template<>
template<typename StrainT, typename ControlT, typename StressT>
DEVICE_TYPE inline void
ComputeCauchyStress<2>::operator()
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPenalizedBulkModulus,
 const ControlT & aPenalizedShearModulus,
 const Plato::ScalarMultiVectorT<StrainT> & aStrain,
 const Plato::ScalarMultiVectorT<StressT> & aStress) const
{
    StrainT tTrace = aStrain(aCellOrdinal, 0) + aStrain(aCellOrdinal, 1) + aStrain(aCellOrdinal, 3);
    StrainT tTraceOver3 = tTrace / static_cast<Plato::Scalar>(3.0);

    // compute normal stress components
    aStress(aCellOrdinal, 0) = (2.0 * aPenalizedShearModulus) * (aStrain(aCellOrdinal, 0) - tTraceOver3);
    aStress(aCellOrdinal, 1) = (2.0 * aPenalizedShearModulus) * (aStrain(aCellOrdinal, 1) - tTraceOver3);
    aStress(aCellOrdinal, 3) = (2.0 * aPenalizedShearModulus) * (aStrain(aCellOrdinal, 3) - tTraceOver3);
    
    // add hydrostatic stress to normal components
    aStress(aCellOrdinal, 0) += aPenalizedBulkModulus * tTrace; // sigma_11
    aStress(aCellOrdinal, 1) += aPenalizedBulkModulus * tTrace; // sigma_22
    aStress(aCellOrdinal, 3) += aPenalizedBulkModulus * tTrace; // sigma_33

    // compute shear components - elastic strain already has 2 multiplier, see equation in function declaration
    aStress(aCellOrdinal, 2) = aPenalizedShearModulus * aStrain(aCellOrdinal, 2); // sigma_12
}

/******************************************************************************//**
 * \brief Compute the Cauchy stress for 1D
**********************************************************************************/
template<>
template<typename StrainT, typename ControlT, typename StressT>
DEVICE_TYPE inline void
ComputeCauchyStress<1>::operator()
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPenalizedBulkModulus,
 const ControlT & aPenalizedShearModulus,
 const Plato::ScalarMultiVectorT<StrainT> & aStrain,
 const Plato::ScalarMultiVectorT<StressT> & aStress) const
{
    // compute hydrostatic and deviatoric strain
    StrainT tTraceOver3 = aStrain(aCellOrdinal, 0) / static_cast<Plato::Scalar>(3.0);
    StrainT tTwoTimesDeviatoricStrain = static_cast<Plato::Scalar>(2.0) * (aStrain(aCellOrdinal, 0) - tTraceOver3);
    // compute normal stress components
    aStress(aCellOrdinal, 0) = aPenalizedShearModulus * tTwoTimesDeviatoricStrain + aPenalizedBulkModulus * tTraceOver3;
}

}
// namespace Plato

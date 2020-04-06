/*
 * ComputeDeviatoricStress.hpp
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
 * \brief Compute deviatoric stress:
 *
 * \f$ \sigma_{ij} = \sigma_{ij} - \frac{\sigma_{kk}}{3}\delta_{ij} \f$,
 * or
 * \f$ 2*\mu*\epsilon_{ij} - \frac{\epsilon_{kk}}{3}\delta_{ij} \f$
 *
 * where, \f$ \epsilon_{ij} \f$ is the strain tensor, \f$ \sigma_{ij} \f$ is the
 * stress tensor, \f$ \sigma_{kk} \f$ is the trace of the stress tensor, \f$ \mu \f$
 * is the shear modulus, \f$ and \epsilon_{kk} \f$ is the trace of the strain tensor.
 *
**********************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeDeviatoricStress
{
public:
    /******************************************************************************//**
     * \brief Functor constructor
    **********************************************************************************/
    ComputeDeviatoricStress(){}

    /******************************************************************************//**
     * \brief Compute the deviatoric stress
     * \param [in]  aCellOrdinal           cell/element index
     * \param [in]  aPenalizedShearModulus penalized elastic shear modulus
     * \param [in]  aStrain                elastic strain tensor
     * \param [out] aStress                deviatoric stress tensor
    **********************************************************************************/
    template<typename StrainT, typename ControlT, typename StressT>
    DEVICE_TYPE inline void operator()
    (const Plato::OrdinalType & aCellOrdinal,
     const ControlT & aPenalizedShearModulus,
     const Plato::ScalarMultiVectorT<StrainT> & aStrain,
     const Plato::ScalarMultiVectorT<StressT> & aStress) const;
};
// class ComputeDeviatoricStress

/******************************************************************************//**
 * \brief Compute the deviatoric stress for 1D
**********************************************************************************/
template<>
template<typename StrainT, typename ControlT, typename StressT>
DEVICE_TYPE inline void
ComputeDeviatoricStress<1>::operator ()
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPenalizedShearModulus,
 const Plato::ScalarMultiVectorT<StrainT> & aStrain,
 const Plato::ScalarMultiVectorT<StressT> & aStress) const
{
  StrainT tTraceOver3 = aStrain(aCellOrdinal, 0) / static_cast<Plato::Scalar>(3.0);

  // sigma_11
  aStress(aCellOrdinal, 0) = (static_cast<Plato::Scalar>(2.0) * aPenalizedShearModulus)
                           * (aStrain(aCellOrdinal, 0) - tTraceOver3);
}

/******************************************************************************//**
 * \brief Compute the deviatoric stress for 2D - plane strain
**********************************************************************************/
template<>
template<typename StrainT, typename ControlT, typename StressT>
DEVICE_TYPE inline void
ComputeDeviatoricStress<2>::operator ()
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPenalizedShearModulus,
 const Plato::ScalarMultiVectorT<StrainT> & aStrain,
 const Plato::ScalarMultiVectorT<StressT> & aStress) const
{
  StrainT tTraceOver3 = (aStrain(aCellOrdinal, 0) + aStrain(aCellOrdinal, 1)
          + aStrain(aCellOrdinal, 3)) / static_cast<Plato::Scalar>(3.0);

  // sigma_11
  aStress(aCellOrdinal, 0) = (static_cast<Plato::Scalar>(2.0) * aPenalizedShearModulus)
                                        * (aStrain(aCellOrdinal, 0) - tTraceOver3);

  // sigma_22
  aStress(aCellOrdinal, 1) = (static_cast<Plato::Scalar>(2.0) * aPenalizedShearModulus)
                                        * (aStrain(aCellOrdinal, 1) - tTraceOver3);

  // sigma_12
  aStress(aCellOrdinal, 2) = aPenalizedShearModulus * aStrain(aCellOrdinal, 2);

  // sigma_33 - out-of-plane stress
  aStress(aCellOrdinal, 3) = (static_cast<Plato::Scalar>(2.0) * aPenalizedShearModulus)
                                        * (aStrain(aCellOrdinal, 3) - tTraceOver3);
}

/******************************************************************************//**
 * \brief Compute the deviatoric stress for 3D
**********************************************************************************/
template<>
template<typename StrainT, typename ControlT, typename StressT>
DEVICE_TYPE inline void
ComputeDeviatoricStress<3>::operator()
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPenalizedShearModulus,
 const Plato::ScalarMultiVectorT<StrainT> & aStrain,
 const Plato::ScalarMultiVectorT<StressT> & aStress) const
{
  StrainT tTraceOver3 = (  aStrain(aCellOrdinal, 0) + aStrain(aCellOrdinal, 1)
                                + aStrain(aCellOrdinal, 2) ) / static_cast<Plato::Scalar>(3.0);
  aStress(aCellOrdinal, 0) = (2.0 * aPenalizedShearModulus) * (aStrain(aCellOrdinal, 0) -
                                                                         tTraceOver3);
  aStress(aCellOrdinal, 1) = (2.0 * aPenalizedShearModulus) * (aStrain(aCellOrdinal, 1) -
                                                                         tTraceOver3);
  aStress(aCellOrdinal, 2) = (2.0 * aPenalizedShearModulus) * (aStrain(aCellOrdinal, 2) -
                                                                         tTraceOver3);
  aStress(aCellOrdinal, 3) = aPenalizedShearModulus * aStrain(aCellOrdinal, 3);
  aStress(aCellOrdinal, 4) = aPenalizedShearModulus * aStrain(aCellOrdinal, 4);
  aStress(aCellOrdinal, 5) = aPenalizedShearModulus * aStrain(aCellOrdinal, 5);
}

}
// namespace Plato

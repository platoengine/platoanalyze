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
 * \brief Compute the Cauchy stress tensor using the pressure field
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
class ComputeStabilizedCauchyStress
{
public:
    /******************************************************************************//**
     * \brief Functor constructor
    **********************************************************************************/
    ComputeStabilizedCauchyStress(){}

    /******************************************************************************//**
     * \brief Compute the Cauchy stress tensor:
     *
     * \param [in]  aCellOrdinal           cell/element index
     * \param [in]  aPressure              pressure
     * \param [in]  aDeviatoricStress      deviatoric stress
     * \param [out] aStress                Cauchy stress tensor
    **********************************************************************************/
    template<typename PressureT, typename DevStressT, typename StressT>
    DEVICE_TYPE
    inline void operator()
    (const Plato::OrdinalType & aCellOrdinal,
     const Plato::ScalarVectorT<PressureT> & aPressure,
     const Plato::ScalarMultiVectorT<DevStressT> & aDeviatoricStress,
     const Plato::ScalarMultiVectorT<StressT> & aStress) const;
};
// class ComputeCauchyStress

/******************************************************************************//**
 * \brief Compute the Cauchy stress for 3D
**********************************************************************************/
template<>
template<typename PressureT, typename DevStressT, typename StressT>
DEVICE_TYPE inline void
ComputeStabilizedCauchyStress<3>::operator()
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVectorT<PressureT> & aPressure,
 const Plato::ScalarMultiVectorT<DevStressT> & aDeviatoricStress,
 const Plato::ScalarMultiVectorT<StressT> & aStress) const
{
    // add pressure stress to normal components
    aStress(aCellOrdinal, 0) = aDeviatoricStress(aCellOrdinal, 0) + aPressure(aCellOrdinal);
    aStress(aCellOrdinal, 1) = aDeviatoricStress(aCellOrdinal, 1) + aPressure(aCellOrdinal);
    aStress(aCellOrdinal, 2) = aDeviatoricStress(aCellOrdinal, 2) + aPressure(aCellOrdinal);

    // shear components
    aStress(aCellOrdinal, 3) = aDeviatoricStress(aCellOrdinal, 3);
    aStress(aCellOrdinal, 4) = aDeviatoricStress(aCellOrdinal, 4);
    aStress(aCellOrdinal, 5) = aDeviatoricStress(aCellOrdinal, 5);
}

/******************************************************************************//**
 * \brief Compute the Cauchy stress for 2D - Plane strain
**********************************************************************************/
template<>
template<typename PressureT, typename DevStressT, typename StressT>
DEVICE_TYPE inline void
ComputeStabilizedCauchyStress<2>::operator()
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVectorT<PressureT> & aPressure,
 const Plato::ScalarMultiVectorT<DevStressT> & aDeviatoricStress,
 const Plato::ScalarMultiVectorT<StressT> & aStress) const
{
    // add pressure stress to normal components
    aStress(aCellOrdinal, 0) = aDeviatoricStress(aCellOrdinal, 0) + aPressure(aCellOrdinal);
    aStress(aCellOrdinal, 1) = aDeviatoricStress(aCellOrdinal, 1) + aPressure(aCellOrdinal);
    aStress(aCellOrdinal, 3) = aDeviatoricStress(aCellOrdinal, 3) + aPressure(aCellOrdinal);

    // shear components
    aStress(aCellOrdinal, 2) = aDeviatoricStress(aCellOrdinal, 2);
}

}
// namespace Plato

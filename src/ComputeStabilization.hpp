/*
 * ComputeStabilization.hpp
 *
 *  Created on: Feb 29, 2020
 */

#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/***************************************************************************//**
 *
 * \brief Compute stabilization term, defined as:
 *
 *   /f$ \tau\nabla{p} - \nabla{\Pi} /f$,
 *
 *   where /f$\tau/f$ is a stabilization multiplier, /f$ \nabla{p} /f$ is the pressure
 *   gradient, and /f$ \nabla{\Pi} /f$ is the projected pressure gradient.  The
 *   stabilization parameter, /f$ \tau /f$ is defined as /f$ \frac{\Omega_{e}^{2/3}}{2G} /f$,
 *   where G is the shear modulus.
 *
 * \tparam SpaceDim spatial dimensions
 *
*******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeStabilization
{
private:
    Plato::Scalar mTwoOverThree;         /*!< 2/3 constant - avoids repeated calculation */
    Plato::Scalar mPressureScaling;      /*!< pressure scaling term */
    Plato::Scalar mElasticShearModulus;  /*!< elastic shear modulus */

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aScaling       multiplier used to improve the system of equations condition number
     * \param [in] aShearModulus  elastic shear modulus
    *******************************************************************************/
    explicit ComputeStabilization(const Plato::Scalar & aScaling, const Plato::Scalar & aShearModulus) :
        mTwoOverThree(2.0/3.0),
        mPressureScaling(aScaling),
        mElasticShearModulus(aShearModulus)
    {
    }

    /***************************************************************************//**
     * \brief Compute stabilization term
     *
     * \tparam ConfigT        POD type for 1-D Kokkos::View
     * \tparam PressGradT     POD type for 2-D Kokkos::View
     * \tparam ProjPressGradT POD type for 2-D Kokkos::View
     * \tparam ResultT        POD type for 2-D Kokkos::View
     *
     * \param [in] aCellOrdinal           cell ordinal, i.e index
     * \param [in] aCellVolume            cell volume
     * \param [in] aPressureGrad          pressure gradient
     * \param [in] aProjectedPressureGrad projected pressure gradient
     * \param [in/out] aStabilization     stabilization term
     *
    *******************************************************************************/
    template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType &aCellOrdinal,
               const Plato::ScalarVectorT<ConfigT> & aCellVolume,
               const Plato::ScalarMultiVectorT<PressGradT> &aPressureGrad,
               const Plato::ScalarMultiVectorT<ProjPressGradT> &aProjectedPressureGrad,
               const Plato::ScalarMultiVectorT<ResultT> &aStabilization) const;
};
// class ComputeStabilization

/***************************************************************************//**
 *
 * \brief Specialization for 3-D applications
 *
 * \tparam ConfigT        POD type for 1-D Kokkos::View
 * \tparam PressGradT     POD type for 2-D Kokkos::View
 * \tparam ProjPressGradT POD type for 2-D Kokkos::View
 * \tparam ResultT        POD type for 2-D Kokkos::View
 *
 * \param [in] aCellOrdinal           cell ordinal, i.e index
 * \param [in] aCellVolume            cell volume
 * \param [in] aPressureGrad          pressure gradient
 * \param [in] aProjectedPressureGrad projected pressure gradient
 * \param [in/out] aStabilization     stabilization term
 *
*******************************************************************************/
template<>
template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
DEVICE_TYPE inline void
ComputeStabilization<3>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                    const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                    const Plato::ScalarMultiVectorT<PressGradT> & aPressureGrad,
                                    const Plato::ScalarMultiVectorT<ProjPressGradT> & aProjectedPressureGrad,
                                    const Plato::ScalarMultiVectorT<ResultT> & aStabilization) const
{
    ConfigT tTau = pow(aCellVolume(aCellOrdinal), mTwoOverThree) / (static_cast<Plato::Scalar>(2.0) * mElasticShearModulus);
    aStabilization(aCellOrdinal, 0) = mPressureScaling * tTau
        * ((mPressureScaling * aPressureGrad(aCellOrdinal, 0)) - aProjectedPressureGrad(aCellOrdinal, 0));

    aStabilization(aCellOrdinal, 1) = mPressureScaling * tTau
        * ((mPressureScaling * aPressureGrad(aCellOrdinal, 1)) - aProjectedPressureGrad(aCellOrdinal, 1));

    aStabilization(aCellOrdinal, 2) = mPressureScaling * tTau
        * ((mPressureScaling * aPressureGrad(aCellOrdinal, 2)) - aProjectedPressureGrad(aCellOrdinal, 2));
}

/***************************************************************************//**
 *
 * \brief Specialization for 2-D applications
 *
 * \tparam ConfigT        POD type for 1-D Kokkos::View
 * \tparam PressGradT     POD type for 2-D Kokkos::View
 * \tparam ProjPressGradT POD type for 2-D Kokkos::View
 * \tparam ResultT        POD type for 2-D Kokkos::View
 *
 * \param [in] aCellOrdinal           cell ordinal, i.e index
 * \param [in] aCellVolume            cell volume
 * \param [in] aPressureGrad          pressure gradient
 * \param [in] aProjectedPressureGrad projected pressure gradient
 * \param [in/out] aStabilization     stabilization term
 *
*******************************************************************************/
template<>
template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
DEVICE_TYPE inline void
ComputeStabilization<2>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                    const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                    const Plato::ScalarMultiVectorT<PressGradT> & aPressureGrad,
                                    const Plato::ScalarMultiVectorT<ProjPressGradT> & aProjectedPressureGrad,
                                    const Plato::ScalarMultiVectorT<ResultT> & aStabilization) const
{
    ConfigT tTau = pow(aCellVolume(aCellOrdinal), mTwoOverThree) / (static_cast<Plato::Scalar>(2.0) * mElasticShearModulus);
    aStabilization(aCellOrdinal, 0) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 0) - aProjectedPressureGrad(aCellOrdinal, 0));

    aStabilization(aCellOrdinal, 1) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 1) - aProjectedPressureGrad(aCellOrdinal, 1));
}

/***************************************************************************//**
 *
 * \brief Specialization for 1-D applications
 *
 * \tparam ConfigT        POD type for 1-D Kokkos::View
 * \tparam PressGradT     POD type for 2-D Kokkos::View
 * \tparam ProjPressGradT POD type for 2-D Kokkos::View
 * \tparam ResultT        POD type for 2-D Kokkos::View
 *
 * \param [in] aCellOrdinal           cell ordinal, i.e index
 * \param [in] aCellVolume            cell volume
 * \param [in] aPressureGrad          pressure gradient
 * \param [in] aProjectedPressureGrad projected pressure gradient
 * \param [in/out] aStabilization     stabilization term
 *
*******************************************************************************/
template<>
template<typename ConfigT, typename PressGradT, typename ProjPressGradT, typename ResultT>
DEVICE_TYPE inline void
ComputeStabilization<1>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                    const Plato::ScalarVectorT<ConfigT> & aCellVolume,
                                    const Plato::ScalarMultiVectorT<PressGradT> & aPressureGrad,
                                    const Plato::ScalarMultiVectorT<ProjPressGradT> & aProjectedPressureGrad,
                                    const Plato::ScalarMultiVectorT<ResultT> & aStabilization) const
{
    ConfigT tTau = pow(aCellVolume(aCellOrdinal), mTwoOverThree) / (static_cast<Plato::Scalar>(2.0) * mElasticShearModulus);
    aStabilization(aCellOrdinal, 0) = mPressureScaling * tTau
        * (mPressureScaling * aPressureGrad(aCellOrdinal, 0) - aProjectedPressureGrad(aCellOrdinal, 0));
}

}
// namespace Plato

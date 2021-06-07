/*
 * StrainDivergence.hpp
 *
 *  Created on: Feb 29, 2020
 */

#pragma once

#include "PlatoStaticsTypes.hpp"
#include "MaterialModel.hpp"

namespace Plato
{

/***************************************************************************//**
 *
 * \brief Apply divergence operator to the strain tensor, i.e.
 *            /f$ \div\cdot\epsilon /f$,
 *   where /f$ \epsilon /f$ denotes the strain tensor.  Used in stabilized
 *   elasto- and thermo-plasticity problems.
 *
 * \tparam SpaceDim spatial dimensions
 *
*******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class StrainDivergence
{
    Plato::TensorConstant<SpaceDim> mReferenceStrain;

public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    StrainDivergence() : mReferenceStrain(0.0) {}

    StrainDivergence(Plato::MaterialModel<SpaceDim> aMaterialParameters) :
        mReferenceStrain(aMaterialParameters.getTensorConstant("Reference Strain")) {}


    /***************************************************************************//**
     *
     * \brief Apply the divergence operator to the strain tensor.
     *
     * \tparam StrainType POD type for 2-D Kokkos::View
     * \tparam ResultType POD type for 1-D Kokkos::View
     *
     * \param [in] aCellOrdinal cell ordinal, i.e. index
     * \param [in] aStrain      strain tensor
     * \param [in] aOutput      strain tensor divergence
     *
    *******************************************************************************/
    template<typename StrainType, typename ResultType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<StrainType> & aStrain,
               const Plato::ScalarVectorT<ResultType> & aOutput) const;

    /***************************************************************************//**
     *
     * \brief Apply the divergence operator to the strain tensor.
     *
     * \tparam StrainType POD type for 2-D Kokkos::View
     * \tparam ResultType POD type for 1-D Kokkos::View
     *
     * \param [in] aCellOrdinal cell ordinal, i.e. index
     * \param [in] aStrainIncr  strain tensor
     * \param [in] aPrevStrain  strain tensor
     * \param [in] aOutput      strain tensor divergence
     *
    *******************************************************************************/
    template<typename StrainType, typename ResultType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<StrainType> & aStrainIncr,
               const Plato::ScalarMultiVector & aPrevStrain,
               const Plato::ScalarVectorT<ResultType> & aOutput) const;
};
// class StrainDivergence

template<>
template<typename StrainType, typename ResultType>
DEVICE_TYPE inline void
StrainDivergence <3>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                 const Plato::ScalarMultiVectorT<StrainType> & aStrain,
                                 const Plato::ScalarVectorT<ResultType> & aOutput) const
{
    aOutput(aCellOrdinal) = aStrain(aCellOrdinal, 0) + aStrain(aCellOrdinal, 1) + aStrain(aCellOrdinal, 2);
}

template<>
template<typename StrainType, typename ResultType>
DEVICE_TYPE inline void
StrainDivergence <3>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                 const Plato::ScalarMultiVectorT<StrainType> & aStrainIncr,
                                 const Plato::ScalarMultiVector & aPrevStrain,
                                 const Plato::ScalarVectorT<ResultType> & aOutput) const
{
    aOutput(aCellOrdinal) = aStrainIncr(aCellOrdinal, 0) + aStrainIncr(aCellOrdinal, 1) + aStrainIncr(aCellOrdinal, 2)
                          + aPrevStrain(aCellOrdinal, 0) + aPrevStrain(aCellOrdinal, 1) + aPrevStrain(aCellOrdinal, 2)
                          - mReferenceStrain(0,0)        - mReferenceStrain(1,1)        - mReferenceStrain(2,2);
}

/***************************************************************************//**
 *
 * \brief Specialization for 2-D applications. Plane Strain formulation, i.e.
 *   out-of-plane strain (e_33) is zero.
 *
 * \tparam StrainType POD type for 2-D Kokkos::View
 * \tparam ResultType POD type for 1-D Kokkos::View
 *
 * \param [in] aCellOrdinal cell ordinal, i.e. index
 * \param [in] aStrain      strain tensor
 * \param [in] aOutput      strain tensor divergence
 *
*******************************************************************************/
template<>
template<typename StrainType, typename ResultType>
DEVICE_TYPE inline void
StrainDivergence <2>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                 const Plato::ScalarMultiVectorT<StrainType> & aStrain,
                                 const Plato::ScalarVectorT<ResultType> & aOutput) const
{
    aOutput(aCellOrdinal) = aStrain(aCellOrdinal, 0)  // e^{elastic}_{11}
                          + aStrain(aCellOrdinal, 1); // e^{elastic}_{22}
}

template<>
template<typename StrainType, typename ResultType>
DEVICE_TYPE inline void
StrainDivergence <2>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                 const Plato::ScalarMultiVectorT<StrainType> & aStrainIncr,
                                 const Plato::ScalarMultiVector & aPrevStrain,
                                 const Plato::ScalarVectorT<ResultType> & aOutput) const
{
    aOutput(aCellOrdinal) = aStrainIncr(aCellOrdinal, 0) + aPrevStrain(aCellOrdinal, 0) - mReferenceStrain(0,0)  /* e^{elastic}_{11} */
                          + aStrainIncr(aCellOrdinal, 1) + aPrevStrain(aCellOrdinal, 1) - mReferenceStrain(1,1); /* e^{elastic}_{22} */
}

/***************************************************************************//**
 *
 * \brief Specialization for 1-D applications.
 *
 * \tparam StrainType POD type for 2-D Kokkos::View
 * \tparam ResultType POD type for 1-D Kokkos::View
 *
 * \param [in] aCellOrdinal cell ordinal, i.e. index
 * \param [in] aStrain      strain tensor
 * \param [in] aOutput      strain tensor divergence
 *
*******************************************************************************/
template<>
template<typename StrainType, typename ResultType>
DEVICE_TYPE inline void
StrainDivergence <1>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                 const Plato::ScalarMultiVectorT<StrainType> & aStrain,
                                 const Plato::ScalarVectorT<ResultType> & aOutput) const
{
    aOutput(aCellOrdinal) = aStrain(aCellOrdinal, 0);
}

}
// namespace Plato

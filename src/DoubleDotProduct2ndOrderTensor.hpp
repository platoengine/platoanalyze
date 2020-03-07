/*
 * DoubleDotProduct2ndOrderTensor.hpp
 *
 *  Created on: Feb 29, 2020
 */

#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Double dot product for 2nd order tensors, e.g. strain tensor, using
 *   Voigt notation.  Operation is defined as: \f$ \alpha = A(i,j)B(i,j) \f$
 *
 * \tparam SpaceDim spatial dimensions
*******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class DoubleDotProduct2ndOrderTensor
{
public:
    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    DoubleDotProduct2ndOrderTensor(){}

    /***************************************************************************//**
     * \brief Double dot product for 2nd order tensors, e.g. strain tensor, using
     *   Voigt notation.
     *
     * \tparam AViewType POD type for Kokkos::View
     * \tparam BViewType POD type for Kokkos::View
     * \tparam CViewType POD type for Kokkos::View
     *
     * \param [in] aCellOrdinal cell, i.e. element, index
     * \param [in] aA           input container A
     * \param [in] aB           input container B
     * \param [in] aOutput      output container
     *******************************************************************************/
    template<typename AViewType, typename BViewType, typename CViewType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType& aCellOrdinal,
               const Plato::ScalarMultiVectorT<AViewType> & aA,
               const Plato::ScalarMultiVectorT<BViewType> & aB,
               const Plato::ScalarVectorT<CViewType> & aOutput) const;
};
// class DoubleDotProduct2ndOrderTensor


/***************************************************************************//**
 * \brief Double dot product for 2nd order tensors, e.g. strain tensor, using
 *   Voigt notation. Specialized for 3-D problems
 *
 *   tensor = {tensor_11,tensor_22,tensor_33,tensor_23,tensor_13,tensor_23}
 *
 * \tparam AViewType POD type for Kokkos::View
 * \tparam BViewType POD type for Kokkos::View
 * \tparam CViewType POD type for Kokkos::View
 *
 * \param [in] aCellOrdinal cell, i.e. element, index
 * \param [in] aA           input container A
 * \param [in] aB           input container B
 * \param [in] aOutput      output container
*******************************************************************************/
template<>
template<typename AViewType, typename BViewType, typename CViewType>
DEVICE_TYPE inline void
DoubleDotProduct2ndOrderTensor<3>::operator()(const Plato::OrdinalType& aCellOrdinal,
                                              const Plato::ScalarMultiVectorT<AViewType> & aA,
                                              const Plato::ScalarMultiVectorT<BViewType> & aB,
                                              const Plato::ScalarVectorT<CViewType> & aOutput) const
{
    aOutput(aCellOrdinal) = aA(aCellOrdinal, 0) * aB(aCellOrdinal, 0)
                          + aA(aCellOrdinal, 1) * aB(aCellOrdinal, 1)
                          + aA(aCellOrdinal, 2) * aB(aCellOrdinal, 2)
                          + static_cast<Plato::Scalar>(2.0) * aA(aCellOrdinal, 3) * aB(aCellOrdinal, 3)
                          + static_cast<Plato::Scalar>(2.0) * aA(aCellOrdinal, 4) * aB(aCellOrdinal, 4)
                          + static_cast<Plato::Scalar>(2.0) * aA(aCellOrdinal, 5) * aB(aCellOrdinal, 5);
}

/***************************************************************************//**
 * \brief Double dot product for 2nd order tensors, e.g. strain and stress
 *   tensors.  Recall that a plane strain assumption is used in 2-D problems.
 *   Hence, a general stress/strain tensor is given by:
 *
 *   epsilon = {epsilon_11,epsilon_22,2*epsilon_12,epsilon_33} (Voigt Notation)
 *
 *   The out-of-plane tensor value, i.e. epsilon, is placed in the last entry
 *   for convenience since the Strain functor assumes that the shear component,
 *   i.e. epsilon_12, is the third entry.
 *
 * \tparam AViewType POD type for Kokkos::View
 * \tparam BViewType POD type for Kokkos::View
 * \tparam CViewType POD type for Kokkos::View
 *
 * \param [in] aCellOrdinal cell, i.e. element, index
 * \param [in] aA           input container A
 * \param [in] aB           input container B
 * \param [in] aOutput      output container
*******************************************************************************/
template<>
template<typename AViewType, typename BViewType, typename CViewType>
DEVICE_TYPE inline void
DoubleDotProduct2ndOrderTensor<2>::operator()(const Plato::OrdinalType& aCellOrdinal,
                                              const Plato::ScalarMultiVectorT<AViewType> & aA,
                                              const Plato::ScalarMultiVectorT<BViewType> & aB,
                                              const Plato::ScalarVectorT<CViewType> & aOutput) const
{
    aOutput(aCellOrdinal) = aA(aCellOrdinal, 0) * aB(aCellOrdinal, 0) // e_11
                          + aA(aCellOrdinal, 1) * aB(aCellOrdinal, 1) // e_22
                          + aA(aCellOrdinal, 3) * aB(aCellOrdinal, 3) // e_33
                          + static_cast<Plato::Scalar>(2.0) * aA(aCellOrdinal, 2) * aB(aCellOrdinal, 2); // e_12
}
/***************************************************************************//**
 * \brief Double dot product for 2nd order tensors, e.g. strain tensor, using
 *   Voigt notation. Specialized for 1-D problems
 *
 * \tparam AViewType POD type for Kokkos::View
 * \tparam BViewType POD type for Kokkos::View
 * \tparam CViewType POD type for Kokkos::View
 *
 * \param [in] aCellOrdinal cell, i.e. element, index
 * \param [in] aA           input container A
 * \param [in] aB           input container B
 * \param [in] aOutput      output container
*******************************************************************************/
template<>
template<typename AViewType, typename BViewType, typename CViewType>
DEVICE_TYPE inline void
DoubleDotProduct2ndOrderTensor<1>::operator()(const Plato::OrdinalType& aCellOrdinal,
                                              const Plato::ScalarMultiVectorT<AViewType> & aA,
                                              const Plato::ScalarMultiVectorT<BViewType> & aB,
                                              const Plato::ScalarVectorT<CViewType> & aOutput) const
{
    aOutput(aCellOrdinal) = aA(aCellOrdinal, 0) * aB(aCellOrdinal, 0);
}

}
// namespace Plato

/*
 * ComputeDeviatoricStrain.hpp
 *
 *  Created on: Mar 7, 2020
 */

#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Compute the deviatoric strain, which is given by:
 *
 *    \f$ \epsilon_{ij}^{d} = \epsilon_{ij}^{e} - \epsilon_{kk}^{e}\delta_{ij}\f$
 *
 * where \f$ \epsilon_{ij}^{e} \f$ is the infinitesimal, i.e. elastic, strain tensor.
 *
 * \tparam SpaceDim spatial dimensions
 *
**********************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeDeviatoricStrain
{
public:
    /******************************************************************************//**
     * \brief Compute the deviatoric strain
     *
     * \tparam ElasticStrainT    elastic strain tensor forward automatic differentiation (FAD) type
     * \tparam DeviatoricStrainT deviatoric strain tensor FAD type
     *
     * \param [in]     aCellOrdinal       element index
     * \param [in]     aElasticStrain     elastic strain tensor
     * \param [in\out] aDeviatoricStrain  deviatoric strain tensor
    **********************************************************************************/
    template<typename ElasticStrainT, typename DeviatoricStrainT>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarMultiVectorT<ElasticStrainT> & aElasticStrain,
                                       const Plato::ScalarMultiVectorT<DeviatoricStrainT> & aDeviatoricStrain) const;
};
// class ComputeDeviatoricStrain

template<>
template<typename ElasticStrainT, typename DeviatoricStrainT>
DEVICE_TYPE inline void
ComputeDeviatoricStrain<3>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarMultiVectorT<ElasticStrainT> & aElasticStrain,
                                       const Plato::ScalarMultiVectorT<DeviatoricStrainT> & aDeviatoricStrain) const
{
    ElasticStrainT tTraceOver3 = (aElasticStrain(aCellOrdinal, 0) + aElasticStrain(aCellOrdinal, 1)
            + aElasticStrain(aCellOrdinal, 2)) / static_cast<Plato::Scalar>(3.0);

    aDeviatoricStrain(aCellOrdinal, 0) = aElasticStrain(aCellOrdinal, 0) - tTraceOver3;
    aDeviatoricStrain(aCellOrdinal, 1) = aElasticStrain(aCellOrdinal, 1) - tTraceOver3;
    aDeviatoricStrain(aCellOrdinal, 2) = aElasticStrain(aCellOrdinal, 2) - tTraceOver3;
    aDeviatoricStrain(aCellOrdinal, 3) = aElasticStrain(aCellOrdinal, 3);
    aDeviatoricStrain(aCellOrdinal, 4) = aElasticStrain(aCellOrdinal, 4);
    aDeviatoricStrain(aCellOrdinal, 5) = aElasticStrain(aCellOrdinal, 5);
}

template<>
template<typename ElasticStrainT, typename DeviatoricStrainT>
DEVICE_TYPE inline void
ComputeDeviatoricStrain<2>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarMultiVectorT<ElasticStrainT> & aElasticStrain,
                                       const Plato::ScalarMultiVectorT<DeviatoricStrainT> & aDeviatoricStrain) const
{
    ElasticStrainT tTraceOver3 = (aElasticStrain(aCellOrdinal, 0) + aElasticStrain(aCellOrdinal, 1)
            + aElasticStrain(aCellOrdinal, 3)) / static_cast<Plato::Scalar>(3.0);

    aDeviatoricStrain(aCellOrdinal, 0) = aElasticStrain(aCellOrdinal, 0) - tTraceOver3;
    aDeviatoricStrain(aCellOrdinal, 1) = aElasticStrain(aCellOrdinal, 1) - tTraceOver3;
    aDeviatoricStrain(aCellOrdinal, 2) = aElasticStrain(aCellOrdinal, 2);
    aDeviatoricStrain(aCellOrdinal, 3) = aElasticStrain(aCellOrdinal, 3) - tTraceOver3;
}

template<>
template<typename ElasticStrainT, typename DeviatoricStrainT>
DEVICE_TYPE inline void
ComputeDeviatoricStrain<1>::operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarMultiVectorT<ElasticStrainT> & aElasticStrain,
                                       const Plato::ScalarMultiVectorT<DeviatoricStrainT> & aDeviatoricStrain) const
{
    ElasticStrainT tTraceOver3 = aElasticStrain(aCellOrdinal, 0) / static_cast<Plato::Scalar>(3.0);
    aDeviatoricStrain(aCellOrdinal, 0) = aElasticStrain(aCellOrdinal, 0) - tTraceOver3;
}

}
// namespace Plato

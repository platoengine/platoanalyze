/*
 * ComputeElasticWork.hpp
 *
 *  Created on: Mar 7, 2020
 */

#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Compute elastic work, which is given by:
 *
 *    \f$ w_e = \mu\epsilon_{ij}^{d}\epsilon_{ij}^{d} + \kappa\epsilon_{kk}^{2}\f$
 *
 * where \f$ w_e \f$ is the elastic work, \f$ \mu \f$ is the shear modulus,
 * \f$ \epsilon_{ij}^{d} \f$ is the infinitesimal deviatoric strain tensor,
 * \f$ \kappa \f$ is the bulk modulus, \f$ \epsilon_{kk} \f$ is the infinitesimal
 * volumetric strain.
 *
 * \tparam SpaceDim spatial dimensions
 *
**********************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ComputeElasticWork
{
public:
    /******************************************************************************//**
     * \brief Compute the elastic work
     *
     * \tparam OutputT           output forward automatic differentiation (FAD) type
     * \tparam ElasticStrainT    elastic strain tensor FAD type
     * \tparam DeviatoricStrainT deviatoric strain tensor FAD type
     * \tparam ShearModulusT     shear modulus FAD type
     * \tparam BulkModulusT      bulk modulus FAD type
     *
     * \param [in]     aCellOrdinal       element index
     * \param [in]     aElasticStrain     elastic strain tensor
     * \param [in\out] aDeviatoricStrain  deviatoric strain tensor
    **********************************************************************************/
    template<typename OutputT, typename ElasticStrainT, typename DeviatoricStrainT, typename ShearModulusT, typename BulkModulusT>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType &aCellOrdinal,
               const ShearModulusT &aShearModulus,
               const BulkModulusT &aBulkModulus,
               const Plato::ScalarMultiVectorT<ElasticStrainT> &aElasticStrain,
               const Plato::ScalarMultiVectorT<DeviatoricStrainT> &aDeviatoricStrain,
               const Plato::ScalarVectorT<OutputT> & aOutput) const;
};

template<>
template<typename OutputT, typename ElasticStrainT, typename DeviatoricStrainT, typename ShearModulusT, typename BulkModulusT>
DEVICE_TYPE inline void
ComputeElasticWork<3>::operator()(const Plato::OrdinalType &aCellOrdinal,
                                  const ShearModulusT &aShearModulus,
                                  const BulkModulusT &aBulkModulus,
                                  const Plato::ScalarMultiVectorT<ElasticStrainT> &aElasticStrain,
                                  const Plato::ScalarMultiVectorT<DeviatoricStrainT> &aDeviatoricStrain,
                                  const Plato::ScalarVectorT<OutputT> & aOutput) const
{
    // Volumetric strain contribution
    ElasticStrainT tTrace = aElasticStrain(aCellOrdinal, 0) + aElasticStrain(aCellOrdinal, 1) + aElasticStrain(aCellOrdinal, 2);
    ElasticStrainT tTraceSquaredOver2 = ( tTrace * tTrace ) / static_cast<Plato::Scalar>(2.0);
    OutputT tBulkModulusTimesTraceSquaredOver2 = aBulkModulus * tTraceSquaredOver2;

    // Deviatoric strain contribution
    DeviatoricStrainT tDeviatoricStrainSquared = ( aDeviatoricStrain(aCellOrdinal, 0) * aDeviatoricStrain(aCellOrdinal, 0) )
        + ( aDeviatoricStrain(aCellOrdinal, 1) * aDeviatoricStrain(aCellOrdinal, 1) )
        + ( aDeviatoricStrain(aCellOrdinal, 2) * aDeviatoricStrain(aCellOrdinal, 2) )
        + static_cast<Plato::Scalar>(2) * ( aDeviatoricStrain(aCellOrdinal, 3) * aDeviatoricStrain(aCellOrdinal, 3) )
        + static_cast<Plato::Scalar>(2) * ( aDeviatoricStrain(aCellOrdinal, 4) * aDeviatoricStrain(aCellOrdinal, 4) )
        + static_cast<Plato::Scalar>(2) * ( aDeviatoricStrain(aCellOrdinal, 5) * aDeviatoricStrain(aCellOrdinal, 5) );
    OutputT tShearModulusTimesDeviatoricStrainSquared = aShearModulus * tDeviatoricStrainSquared;

    aOutput(aCellOrdinal) = tShearModulusTimesDeviatoricStrainSquared + tBulkModulusTimesTraceSquaredOver2;
}

template<>
template<typename OutputT, typename ElasticStrainT, typename DeviatoricStrainT, typename ShearModulusT, typename BulkModulusT>
DEVICE_TYPE inline void
ComputeElasticWork<2>::operator()(const Plato::OrdinalType &aCellOrdinal,
                                  const ShearModulusT &aShearModulus,
                                  const BulkModulusT &aBulkModulus,
                                  const Plato::ScalarMultiVectorT<ElasticStrainT> &aElasticStrain,
                                  const Plato::ScalarMultiVectorT<DeviatoricStrainT> &aDeviatoricStrain,
                                  const Plato::ScalarVectorT<OutputT> & aOutput) const
{
    // Volumetric strain contribution
    ElasticStrainT tTrace = aElasticStrain(aCellOrdinal, 0) + aElasticStrain(aCellOrdinal, 1) + aElasticStrain(aCellOrdinal, 3);
    ElasticStrainT tTraceSquaredOver2 = ( tTrace * tTrace ) / static_cast<Plato::Scalar>(2.0);
    OutputT tBulkModulusTimesTraceSquaredOver2 = aBulkModulus * tTraceSquaredOver2;

    // Deviatoric strain contribution
    DeviatoricStrainT tDeviatoricStrainSquared = ( aDeviatoricStrain(aCellOrdinal, 0) * aDeviatoricStrain(aCellOrdinal, 0) )
        + ( aDeviatoricStrain(aCellOrdinal, 1) * aDeviatoricStrain(aCellOrdinal, 1) )
        + ( aDeviatoricStrain(aCellOrdinal, 3) * aDeviatoricStrain(aCellOrdinal, 3) )
        + static_cast<Plato::Scalar>(2) * ( aDeviatoricStrain(aCellOrdinal, 2) * aDeviatoricStrain(aCellOrdinal, 2) );
    OutputT tShearModulusTimesDeviatoricStrainSquared = aShearModulus * tDeviatoricStrainSquared;

    aOutput(aCellOrdinal) = tShearModulusTimesDeviatoricStrainSquared + tBulkModulusTimesTraceSquaredOver2;
}

template<>
template<typename OutputT, typename ElasticStrainT, typename DeviatoricStrainT, typename ShearModulusT, typename BulkModulusT>
DEVICE_TYPE inline void
ComputeElasticWork<1>::operator()(const Plato::OrdinalType &aCellOrdinal,
                                  const ShearModulusT &aShearModulus,
                                  const BulkModulusT &aBulkModulus,
                                  const Plato::ScalarMultiVectorT<ElasticStrainT> &aElasticStrain,
                                  const Plato::ScalarMultiVectorT<DeviatoricStrainT> &aDeviatoricStrain,
                                  const Plato::ScalarVectorT<OutputT> & aOutput) const
{
    // Volumetric strain contribution
    ElasticStrainT tTrace = aElasticStrain(aCellOrdinal, 0);
    ElasticStrainT tTraceSquaredOver2 = ( tTrace * tTrace ) / static_cast<Plato::Scalar>(2.0);
    OutputT tBulkModulusTimesTraceSquaredOver2 = aBulkModulus * tTraceSquaredOver2;

    // Deviatoric strain contribution
    DeviatoricStrainT tDeviatoricStrainSquared = aDeviatoricStrain(aCellOrdinal, 0) * aDeviatoricStrain(aCellOrdinal, 0);
    OutputT tShearModulusTimesDeviatoricStrainSquared = aShearModulus * tDeviatoricStrainSquared;

    aOutput(aCellOrdinal) = tShearModulusTimesDeviatoricStrainSquared + tBulkModulusTimesTraceSquaredOver2;
}

}
// namespace Plato

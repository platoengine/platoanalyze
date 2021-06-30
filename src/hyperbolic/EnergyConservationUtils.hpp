/*
 * EnergyConservationUtils.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam NumNodes  number of nodes
 * \tparam SpaceDim  spatial dimensions
 * \tparam ConfigT   configuration Forward Automatic Differentiation (FAD) type
 * \tparam PrevVelT  previous velocity FAD type
 * \tparam PrevTempT previous temperatue FAD type
 * \tparam ResultT   result/output FAD type
 *
 * \fn DEVICE_TYPE inline void calculate_convective_forces
 *
 * \brief Calculate convective forces.
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aGradient    spatial gradient workset
 * \param [in] aPrevVelGP   previous velocity at the Gauss points
 * \param [in] aPrevTemp    previous temperature workset
 * \param [in] aMultiplier  scalar multiplier
 * \param [in\out] aResult  output/result workset
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename PrevVelT,
         typename PrevTempT,
         typename ResultT>
DEVICE_TYPE inline void
calculate_convective_forces
(const Plato::OrdinalType & aCellOrdinals,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<PrevTempT> & aPrevTemp,
 const Plato::ScalarVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aResult(aCellOrdinals) += aMultiplier * aPrevVelGP(aCellOrdinals, tDim)
                * ( aGradient(aCellOrdinals, tNode, tDim) * aPrevTemp(aCellOrdinals, tNode) );
        }
    }
}
// function calculate_convective_forces


/***************************************************************************//**
 * \tparam NumNodes  number of nodes
 * \tparam SpaceDim  spatial dimensions
 * \tparam ConfigT   configuration Forward Automatic Differentiation (FAD) type
 * \tparam PrevVelT  previous velocity FAD type
 * \tparam PrevTempT previous temperature FAD type
 * \tparam ResultT   result/output FAD type
 *
 * \fn DEVICE_TYPE inline void integrate_scalar_field
 *
 * \brief Integrate scalar field, defined as
 *
 *   \f[ \int_{\Omega_e} w^h F d\Omega_e \f]
 *
 * where \f$ w^h \f$ are the test functions and \f$ F \f$ is the scalar field.
 *
 * \param [in] aCellOrdinal    cell/element ordinal
 * \param [in] aBasisFunctions basis functions
 * \param [in] aCellVolume     cell volume workset
 * \param [in] aField          scalar field workset
 * \param [in] aMultiplier     scalar multiplier
 * \param [in\out] aResult     output/result workset
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 typename ConfigT,
 typename SourceT,
 typename ResultT,
 typename ScalarT>
DEVICE_TYPE inline void
integrate_scalar_field
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarVectorT<SourceT> & aField,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
       ScalarT aMultiplier)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        aResult(aCellOrdinal, tNode) += aMultiplier * aBasisFunctions(tNode) *
            aField(aCellOrdinal) * aCellVolume(aCellOrdinal);
    }
}
// function integrate_scalar_field

/***************************************************************************//**
 * \tparam NumNodes  number of nodes
 * \tparam SpaceDim  spatial dimensions
 * \tparam ConfigT   configuration Forward Automatic Differentiation (FAD) type
 * \tparam FluxT     flux FAD type
 * \tparam ScalarT   scalar multiplier FAD type
 * \tparam ResultT   result/output FAD type
 *
 * \fn DEVICE_TYPE inline void calculate_flux_divergence
 *
 * \brief Calculate flux divergence, defined as
 *
 *   \f[ \int_{\Omega_e} \frac{\partial}{\partial x_i} F_i d\Omega_e \f]
 *
 * where \f$ w^h \f$ are the test functions and \f$ F_i \f$ is the i-th flux.
 *
 * \param [in] aCellOrdinal ell/element ordinal
 * \param [in] aGradient    spatial gradient
 * \param [in] aCellVolume  cell volume workset
 * \param [in] aFlux        flux
 * \param [in] aMultiplier  scalar multiplier
 * \param [in\out] aResult  output/result workset
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ConfigT,
 typename FluxT,
 typename ResultT,
 typename ScalarT>
DEVICE_TYPE inline void
calculate_flux_divergence
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<FluxT> & aFlux,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 const ScalarT & aMultiplier)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aCellVolume(aCellOrdinal) *
                ( aGradient(aCellOrdinal, tNode, tDim) * aFlux(aCellOrdinal, tDim) );
        }
    }
}
// function calculate_flux_divergence


/***************************************************************************//**
 * \tparam NumNodes number of nodes
 * \tparam SpaceDim spatial dimensions
 * \tparam FluxT    flux FAD type
 * \tparam ConfigT  configuration Forward Automatic Differentiation (FAD) type
 * \tparam StateT   state FAD type
 *
 * \fn DEVICE_TYPE inline void calculate_flux
 *
 * \brief Calculate flux divergence, defined as
 *
 *   \f[ \int_{\Omega_e} \frac{\partial}{\partial x_i} F d\Omega_e \f]
 *
 * where \f$ w^h \f$ are the test functions and \f$ F \f$ is a scalar field.
 *
 * \param [in] aCellOrdinal ell/element ordinal
 * \param [in] aGradient    spatial gradient
 * \param [in] aScalarField scalar field
 * \param [in\out] aFlux output/result workset
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename FluxT,
 typename ConfigT,
 typename StateT>
DEVICE_TYPE inline void
calculate_flux
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<StateT> & aScalarField,
 const Plato::ScalarMultiVectorT<FluxT> & aFlux)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aFlux(aCellOrdinal, tDim) += aGradient(aCellOrdinal, tNode, tDim) * aScalarField(aCellOrdinal, tNode);
        }
    }
}
// function calculate_flux


/***************************************************************************//**
 * \tparam NumNodesPerCell number of nodes per cell/element (integer)
 * \tparam ControlT control Forward Automatic Differentiation (FAD) evaluation type
 *
 * \fn DEVICE_TYPE inline ControlT penalize_thermal_diffusivity
 *
 * \brief Penalize thermal diffusivity ratio.
 *
 * \param [in] aCellOrdinal      cell/element ordinal
 * \param [in] aThermalDiffRatio thermal diffusivity ratio (solid diffusivity/fluid diffusivity)
 * \param [in] aPenaltyExponent  SIMP penalty model exponent
 * \param [in] aControl          control work set
 *
 * \return penalized thermal diffusivity ratio
 ******************************************************************************/
template
<Plato::OrdinalType NumNodesPerCell,
 typename ControlT>
DEVICE_TYPE inline ControlT
penalize_thermal_diffusivity
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aThermalDiffRatio,
 const Plato::Scalar & aPenaltyExponent,
 const Plato::ScalarMultiVectorT<ControlT> & aControl)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControl);
    ControlT tPenalizedDensity = pow(tDensity, aPenaltyExponent);
    ControlT tPenalizedThermalDiff = aThermalDiffRatio + ( (static_cast<Plato::Scalar>(1.0) - aThermalDiffRatio) * tPenalizedDensity);
    return tPenalizedThermalDiff;
}
// function penalize_thermal_diffusivity



/***************************************************************************//**
 * \tparam NumNodesPerCell number of nodes per cell/element (integer)
 * \tparam ControlT control Forward Automatic Differentiation (FAD) evaluation type
 *
 * \fn DEVICE_TYPE inline ControlT penalize_heat_source_constant
 *
 * \brief Penalize heat source constant. This function is only needed for
 *   density-based topology optimization problems.
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aConstant    heat source constant
 * \param [in] aPenalty     penalty exponent used for density-based penalty model
 * \param [in] aControl     control workset
 *
 * \return penalized heat source constant
 ******************************************************************************/
template
<Plato::OrdinalType NumNodesPerCell,
 typename ControlT>
DEVICE_TYPE inline ControlT
penalize_heat_source_constant
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aConstant,
 const Plato::Scalar & aPenalty,
 const Plato::ScalarMultiVectorT<ControlT> & aControl)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControl);
    ControlT tPenalizedDensity = pow(tDensity, aPenalty);
    auto tPenalizedProperty = (static_cast<Plato::Scalar>(1) - tPenalizedDensity) * aConstant;
    return tPenalizedProperty;
}
// function penalize_heat_source_constant

/***************************************************************************//**
 * \tparam NumNodes number of nodes per cell/element (integer)
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ResultT  result Forward Automatic Differentiation (FAD) evaluation type
 * \tparam ConfigT  configuration FAD evaluation type
 * \tparam PrevVelT previous velocity FAD evaluation type
 * \tparam StabT    stabilization FAD evaluation type
 * \tparam ScalarT  scalar multiplier FAD evaluation type
 *
 * \fn DEVICE_TYPE inline void integrate_stabilizing_scalar_forces
 *
 * \brief Integrate stabilizing scalar field.
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aCellVolume  cell/element volume workset
 * \param [in] aGradient    spatial gradient
 * \param [in] aPrevVelGP   previous velocity at Gauss points
 * \param [in] aStabForce   stabilizing force workset
 * \param [in] aMultiplier  scalar multiplier
 * \param [in/out] aResult  result/output workset
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT,
 typename StabT,
 typename ScalarT>
DEVICE_TYPE inline void
integrate_stabilizing_scalar_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarVectorT<StabT> & aStabForce,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 const ScalarT & aMultiplier)
 {
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimK = 0; tDimK < SpaceDim; tDimK++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * ( ( aGradient(aCellOrdinal, tNode, tDimK) *
                aPrevVelGP(aCellOrdinal, tDimK) ) * aStabForce(aCellOrdinal) ) * aCellVolume(aCellOrdinal);
        }
    }
 }
// function integrate_stabilizing_scalar_forces

}
// namespace Fluids

}
// namespace Plato

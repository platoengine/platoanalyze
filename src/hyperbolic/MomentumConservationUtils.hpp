/*
 * MomentumConservationUtils.hpp
 *
 *  Created on: Apr 6, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "Plato_TopOptFunctors.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ResultT   output work set Forward Automatic Differentiation (FAD) type
 * \tparam ControlT  control work set FAD type
 * \tparam PrevVelT  previous velocity work set FAD type
 *
 * \fn device_type inline void calculate_brinkman_forces
 *
 * \brief Calculate Brinkmann forces, defined as
 *
 * \f[ \alpha\beta u^{n}_i \f]
 *
 * where \f$\alpha\f$ is a scalar multiplier, \f$ \beta \f$ is the dimensionless
 * impermeability constant and \f$u_i^{n}\f$ is i-th component of the velocity
 * field at time step n.
 *
 * \param [in] aCellOrdinal    cell/element ordinal
 * \param [in] aImpermeability impermeability constant
 * \param [in] aPrevTempGP     previous velocity at Gauss points
 * \param [in] aMultiplier     scalar multiplier (default = 1.0)
 * \param [in/out] aResult     result/output workset
 *
 ******************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ControlT,
 typename PrevVelT>
DEVICE_TYPE inline void
calculate_brinkman_forces
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aImpermeability,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for (Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
    {
        aResult(aCellOrdinal, tDim) += aMultiplier * aImpermeability * aPrevVelGP(aCellOrdinal, tDim);
    }
}
// function calculate_brinkman_forces

/***************************************************************************//**
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ResultT   output work set Forward Automatic Differentiation (FAD) type
 * \tparam ControlT  control work set FAD type
 * \tparam PrevTempT previous temperature work set FAD type
 *
 * \fn device_type inline void calculate_natural_convective_forces
 *
 * \brief Calculate natural convective forces, defined as
 *
 * \f[ \alpha Gr_i Pr^2\bar{T}^n \f]
 *
 * where \f$\alpha\f$ denotes a scalar multiplier, \f$ Gr_i \f$ is the Grashof
 * number, \f$ Pr \f$ is the Prandtl number and \f$ T^n \f$ is the temperature
 * field at time step n.
 *
 * \param [in] aCellOrdinal   cell/element ordinal
 * \param [in] aPrTimesPr     Prandtl number squared
 * \param [in] aGrashofNum    Grashof number
 * \param [in] aPrevTempGP    previous temperature at Gauss points
 * \param [in] aMultiplier    scalar multiplier (default = 1.0)
 * \param [in/out] aResult    result/output workset
 *
 ******************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ControlT,
 typename PrevTempT>
DEVICE_TYPE inline void
calculate_natural_convective_forces
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPrTimesPr,
 const Plato::ScalarVector & aGrashofNum,
 const Plato::ScalarVectorT<PrevTempT> & aPrevTempGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for (Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
    {
        aResult(aCellOrdinal, tDim) += aMultiplier * aGrashofNum(tDim) * aPrTimesPr * aPrevTempGP(aCellOrdinal);
    }
}
// function calculate_natural_convective_forces

/***************************************************************************//**
 * \tparam NumNodes number of nodes in cell/element (integer)
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ResultT  output work set Forward Automatic Differentiation (FAD) type
 * \tparam ConfigT  configuration work set FAD type
 * \tparam ControlT control work set FAD type
 * \tparam StrainT  strain rate work set FAD type
 *
 * \fn device_type inline void integrate_viscous_forces
 *
 * \brief Integrate element viscous forces, defined as
 *
 * \f[ \alpha\int_{\Omega}\frac{\partial w_i^h}{\partial\bar{x}_j}\bar\tau_{ij}^n\,d\Omega \f]
 *
 * where \f$\bar\tau_{ij}^n\f$ is the deviatoric stress tensor and \f$\alpha\f$
 * is a scalar multiplier.
 *
 * \param [in] aCellOrdinal   cell/element ordinal
 * \param [in] aPrandtlNumber dimensionless Prandtl number
 * \param [in] aCellVolume    cell/element volume workset
 * \param [in] aGradient      spatial gradient workset
 * \param [in] aStrainRate    strain workset
 * \param [in] aMultiplier    scalar multiplier (default = 1.0)
 * \param [in/out] aResult    result/output workset
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename ControlT,
 typename StrainT>
DEVICE_TYPE inline void
integrate_viscous_forces
(const Plato::OrdinalType & aCellOrdinal,
 const ControlT & aPrandtlNumber,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarArray3DT<StrainT> & aStrainRate,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < SpaceDim; tDimI++)
        {
            auto tDofIndex = (SpaceDim * tNode) + tDimI;
            for(Plato::OrdinalType tDimJ = 0; tDimJ < SpaceDim; tDimJ++)
            {
                aResult(aCellOrdinal, tDofIndex) += aMultiplier * aCellVolume(aCellOrdinal) * aGradient(aCellOrdinal, tNode, tDimJ)
                    * ( static_cast<Plato::Scalar>(2.0) * aPrandtlNumber * aStrainRate(aCellOrdinal, tDimI, tDimJ) );
            }
        }
    }
}
// function integrate_viscous_forces

/***************************************************************************//**
 * \tparam NumNodes number of nodes in cell/element (integer)
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ResultT  output work set Forward Automatic Differentiation (FAD) type
 * \tparam ConfigT  configuration work set FAD type
 * \tparam PrevVelT previous velocity work set FAD type
 *
 * \fn device_type inline void calculate_advected_momentum_forces
 *
 * \brief Calculate advection momentum forces, defined as
 *
 * \f[ \alpha\bar{u}_j^n \frac{\partial \bar{u}_i^n}{\partial\bar{x}_j} \f]
 *
 * where \f$\alpha\f$ is a scalar multiplier, \f$ u_i \f$ is the i-th velocity
 * component and \f$ x_i \f$ is the i-th coordinate.
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aGradient    spatial gradient workset
 * \param [in] aPrevVelWS   previous velocity workset
 * \param [in] aPrevVelGP   previous velocity evaluated at Gauss points
 * \param [in] aMultiplier  scalar multiplier (default = 1.0)
 * \param [in/out] aResult  result/output workset
 *
 ******************************************************************************/
template
<Plato::OrdinalType NumNodes,
 Plato::OrdinalType SpaceDim,
 typename ResultT,
 typename ConfigT,
 typename PrevVelT>
DEVICE_TYPE inline void
calculate_advected_momentum_forces
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelWS,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVelGP,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < SpaceDim; tDimI++)
        {
            auto tCellDofI = (SpaceDim * tNode) + tDimI;
            for(Plato::OrdinalType tDimJ = 0; tDimJ < SpaceDim; tDimJ++)
            {
                aResult(aCellOrdinal, tDimI) += aMultiplier * ( aPrevVelGP(aCellOrdinal, tDimJ) *
                    ( aGradient(aCellOrdinal, tNode, tDimJ) * aPrevVelWS(aCellOrdinal, tCellDofI) ) );
            }
        }
    }
}
// function calculate_advected_momentum_forces

/***************************************************************************//**
 * \tparam NumNodesPerCell number of nodes per cell (integer)
 * \tparam ControlT        control work set Forward Automatic Differentiation (FAD) type
 *
 * \fn DEVICE_TYPE inline ControlT brinkman_penalization
 *
 * \brief Evaluate fictitious material penalty model.
 *
 * \f$  \alpha\frac{\left( 1 - \rho \right)}{1 + \epsilon\rho} \f$
 *
 * where \f$ \alpha \f$ denotes a scalar physical parameter, \f$ \rho \f$ denotes
 * the fictitious density field used to depict the geometry, and \f$ \epsilon \f$
 * is a parameter used to improve the convexity of the Brinkman penalization model.
 *
 * \param [in] aCellOrdinal    element/cell ordinal
 * \param [in] aPhysicalParam  physical parameter to be penalized
 * \param [in] aConvexityParam Brinkman model's convexity parameter
 * \param [in] aControlWS      2D control work set
 *
 * \return penalized physical parameter
 ******************************************************************************/
template
<Plato::OrdinalType NumNodesPerCell,
typename ControlT>
DEVICE_TYPE inline ControlT
brinkman_penalization
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar      & aPhysicalParam,
 const Plato::Scalar      & aConvexityParam,
 const Plato::ScalarMultiVectorT<ControlT> & aControlWS)
{
    ControlT tDensity = Plato::cell_density<NumNodesPerCell>(aCellOrdinal, aControlWS);
    ControlT tPenalizedPhysicalParam = aPhysicalParam * (static_cast<Plato::Scalar>(1.0) - tDensity)
        / (static_cast<Plato::Scalar>(1.0) + (aConvexityParam * tDensity));
    return tPenalizedPhysicalParam;
}
// function brinkman_penalization

/***************************************************************************//**
 * \tparam NumNodesPerCell number of nodes per cell (integer)
 * \tparam NumSpaceDim     number of spatial dimensions (integer)
 * \tparam AViewTypeT      input view Forward Automatic Differentiation (FAD) type
 * \tparam BViewTypeT      input view FAD type
 * \tparam CViewTypeT      input view FAD type
 *
 * \fn DEVICE_TYPE inline void strain_rate
 *
 * \brief Evaluate strain rate.
 *
 * \f[ \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) \f]
 *
 * where \f$ \alpha \f$ denotes a scalar physical parameter, \f$ u_i \f$ denotes the
 * i-th component of the velocity field and \f$ x_i \f$ denotes the i-th coordinate.
 *
 * \param [in] aCellOrdinal element/cell ordinal
 * \param [in] aStateWS     2D view with element state work set
 * \param [in] aGradient    3D view with shape function's derivatives
 * \param [in] aStrainRate  3D view with element strain rate
 ******************************************************************************/
template
<Plato::OrdinalType NumNodesPerCell,
 Plato::OrdinalType NumSpaceDim,
 typename AViewTypeT,
 typename BViewTypeT,
 typename CViewTypeT>
DEVICE_TYPE inline void
strain_rate
(const Plato::OrdinalType & aCellOrdinal,
 const AViewTypeT & aStateWS,
 const BViewTypeT & aGradient,
 const CViewTypeT & aStrainRate)
{
    // calculate strain rate for incompressible flows, which is defined as
    // \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDimI = 0; tDimI < NumSpaceDim; tDimI++)
        {
            for(Plato::OrdinalType tDimJ = 0; tDimJ < NumSpaceDim; tDimJ++)
            {
                auto tLocalDimI = tNode * NumSpaceDim + tDimI;
                auto tLocalDimJ = tNode * NumSpaceDim + tDimJ;
                aStrainRate(aCellOrdinal, tDimI, tDimJ) += static_cast<Plato::Scalar>(0.5) *
                    ( ( aGradient(aCellOrdinal, tNode, tDimJ) * aStateWS(aCellOrdinal, tLocalDimI) )
                    + ( aGradient(aCellOrdinal, tNode, tDimI) * aStateWS(aCellOrdinal, tLocalDimJ) ) );
            }
        }
    }
}
// function strain_rate

/***************************************************************************//**
 * \fn inline bool is_impermeability_defined
 *
 * \brief Return true if dimensionless impermeability number is defined; return
 *   false if it is not defined.
 *
 * \param [in] aInputs input file metadata
 *
 * \return boolean (true or false)
 ******************************************************************************/
inline bool
is_impermeability_defined
(Teuchos::ParameterList & aInputs)
{
    auto tHyperbolic = aInputs.sublist("Hyperbolic");
    if( !tHyperbolic.isSublist("Dimensionless Properties") )
    {
        THROWERR("Parameter Sublist 'Dimensionless Properties' is not defined.")
    }
    auto tSublist = tHyperbolic.sublist("Dimensionless Properties");
    return (tSublist.isParameter("Impermeability Number"));
}
// function is_impermeability_defined

}
// namespace Fluids

}
// namespace Plato

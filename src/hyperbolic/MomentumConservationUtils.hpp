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

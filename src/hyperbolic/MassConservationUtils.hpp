/*
 * MassConservationUtils.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam NumNodes   number of nodes on the cell
 * \tparam SpaceDim   spatial dimensions
 * \tparam ConfigT    configuration Forward Automaitc Differentiation (FAD) type
 * \tparam CurPressT  current pressure FAD type
 * \tparam PrevPressT previous pressure FAD type
 * \tparam PressGradT pressure gradient FAD type
 *
 * \fn device_type void calculate_pressure_gradient
 * \brief Calculate pressure gradient, defined as
 *
 * \f[
 *   \frac{\partial p^{n+\theta_2}}{\partial x_i} =
 *     \alpha\left( 1-\theta_2 \right)\frac{partial p^n}{partial x_i}
 *     + \theta_2\frac{\partial\delta{p}}{\partial x_i}
 * \f]
 *
 * where \f$ \delta{p} = p^{n+1} - p^{n} \f$, \f$ x_i \f$ is the i-th coordinate,
 * \f$ \theta_2 \f$ is artificial pressure damping and \f$ \alpha \f$ is a scalar
 * multiplier.
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aTheta       artificial damping
 * \param [in] aGradient    spatial gradient workset
 * \param [in] aCurPress    current pressure workset
 * \param [in] aPrevPress   previous pressure workset
 * \param [in\out] aPressGrad pressure gradient workset
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename CurPressT,
         typename PrevPressT,
         typename PressGradT>
DEVICE_TYPE inline void
calculate_pressure_gradient
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::Scalar & aTheta,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<CurPressT> & aCurPress,
 const Plato::ScalarMultiVectorT<PrevPressT> & aPrevPress,
 const Plato::ScalarMultiVectorT<PressGradT> & aPressGrad)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aPressGrad(aCellOrdinal, tDim) += ( (static_cast<Plato::Scalar>(1.0) - aTheta)
                * aGradient(aCellOrdinal, tNode, tDim) * aPrevPress(aCellOrdinal, tNode) )
                + ( aTheta * aGradient(aCellOrdinal, tNode, tDim) * aCurPress(aCellOrdinal, tNode) );
        }
    }
}
// function calculate_pressure_gradient

/***************************************************************************//**
 * \tparam NumNodes   number of nodes on the cell
 * \tparam SpaceDim   spatial dimensions
 * \tparam ConfigT    configuration Forward Automaitc Differentiation (FAD) type
 * \tparam FieldT     scalar field FAD type
 * \tparam FieldGradT scalar field gradient FAD type
 *
 * \fn device_type void calculate_scalar_field_gradient
 * \brief Calculate scalar field gradient, defined as
 *
 * \f[ \frac{\partial p^n}{\partial x_i} = \frac{\partial}{\partial x_i} p^n \f]
 *
 * where \f$ p^{n} \f$ is the pressure field at time step n, \f$ x_i \f$ is the ]
 * i-th coordinate.
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aGradient    spatial gradient workset
 * \param [in] aScalarField scalar field workset
 * \param [in\out] aResult  output/result workset
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename FieldT,
         typename FieldGradT>
DEVICE_TYPE inline void
calculate_scalar_field_gradient
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarMultiVectorT<FieldT> & aScalarField,
 const Plato::ScalarMultiVectorT<FieldGradT> & aResult)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aResult(aCellOrdinal, tDim) += aGradient(aCellOrdinal, tNode, tDim) * aScalarField(aCellOrdinal, tNode);
        }
    }
}
// function calculate_scalar_field_gradient

}
// namespace Fluids

}
// namespace Plato

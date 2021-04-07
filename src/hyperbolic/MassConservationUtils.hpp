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
 * \fn device_type void integrate_divergence_operator
 *
 * \tparam NumNodes number of nodes on cell/element (integer)
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ConfigT  configuration FAD evaluation type
 * \tparam PrevVelT previous velocity FAD evaluation type
 * \tparam aResult  result/output FAD evaluation type
 *
 * \brief Integrate momentum divergence, which is defined as
 *
 * \f[
 *   \alpha\int_{\Omega} v^h\frac{\partial\bar{u}_i^{n}}{\partial\bar{x}_i} d\Omega
 * \f]
 *
 * where \f$ v^h \f$ is the test function, \f$ \bar{u}_i^{n} \f$ is the previous
 * velocity, and \f$ \alpha \f$ denotes a scalar multiplier.
 *
 * \param [in] aCellOrdinal    cell/element ordinal
 * \param [in] aBasisFunctions basis functions
 * \param [in] aGradient       spatial gradient workset
 * \param [in] aCellVolume     cell/element volume workset
 * \param [in] aPrevVel        previous velocity workset
 * \param [in] aMultiplier     scalar multiplier
 * \param [in/out] aResult     result/output workset
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename PrevVelT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_divergence_operator
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarVector & aBasisFunctions,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<PrevVelT> & aPrevVel,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aCellVolume(aCellOrdinal) *
                aBasisFunctions(tNode) * aGradient(aCellOrdinal, tNode, tDim) * aPrevVel(aCellOrdinal, tDim);
        }
    }
}
// function integrate_divergence_operator

/***************************************************************************//**
 * \fn device_type void integrate_laplacian_operator
 *
 * \tparam NumNodes number of nodes on cell/element (integer)
 * \tparam SpaceDim spatial dimensions (integer)
 * \tparam ConfigT  configuration FAD evaluation type
 * \tparam FieldT   field FAD evaluation type
 * \tparam aResult  result/output FAD evaluation type
 *
 * \brief Integrate Laplacian operator, defined as
 *
 * \f[
 *   \alpha\int_{\Omega} \frac{\partial v^h}{\partial x_i}\frac{\partial p^n}{\partial x_i} d\Omega
 * \f]
 *
 * where \f$ v^h \f$ is the test function, \f$ p^{n} \f$ is a scalar field at
 * time step n, \f$ x_i \f$ is the i-th coordinate and \f$ \alpha \f$ denotes
 * a scalar multiplier.
 *
 * \param [in] aCellOrdinal    cell/element ordinal
 * \param [in] aGradient       spatial gradient workset
 * \param [in] aCellVolume     cell/element volume workset
 * \param [in] aField          vector field workset
 * \param [in] aMultiplier     scalar multiplier
 * \param [in/out] aResult     result/output workset
 *
 ******************************************************************************/
template<Plato::OrdinalType NumNodes,
         Plato::OrdinalType SpaceDim,
         typename ConfigT,
         typename FieldT,
         typename ResultT>
DEVICE_TYPE inline void
integrate_laplacian_operator
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarArray3DT<ConfigT> & aGradient,
 const Plato::ScalarVectorT<ConfigT> & aCellVolume,
 const Plato::ScalarMultiVectorT<FieldT> & aField,
 const Plato::ScalarMultiVectorT<ResultT> & aResult,
 Plato::Scalar aMultiplier = 1.0)
{
    for(Plato::OrdinalType tNode = 0; tNode < NumNodes; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < SpaceDim; tDim++)
        {
            aResult(aCellOrdinal, tNode) += aMultiplier * aCellVolume(aCellOrdinal) *
                aGradient(aCellOrdinal, tNode, tDim) * aField(aCellOrdinal, tDim);
        }
    }
}
// function integrate_laplacian_operator

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

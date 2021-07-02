/*
 * InterpolateGradientFromScalarNodal.hpp
 *
 *  Created on: Jan 20, 2021
 */

#pragma once

#include "PlatoStaticsTypes.hpp"
#include "Simplex.hpp"

namespace Plato
{

/***********************************************************************************
* 
* \brief InterpolateGradientFromScalarNodal Functor.
*
* Evaluate cell's nodal state gradients at cubature points.
*
***********************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumDofsPerNode=SpaceDim, Plato::OrdinalType DofOffset=0>
class InterpolateGradientFromScalarNodal : public Plato::Simplex<SpaceDim>
{
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;

public:
    /*******************************************************************************
    * 
    * \brief Constructor
    *
    *******************************************************************************/
    InterpolateGradientFromScalarNodal()
    {
    }


    /*******************************************************************************
    *
    * \brief Compute state gradient at cubature points
    *
    * The state gradients are computed as follows: \hat{s} = \sum_{i=1}^{I}
    * \sum_\nabla\phi_{i} s_i, where \hat{s} is the state value,
    * \nabla\phi_{i} is the array of basis function gradients.
    *
    * The input arguments are defined as:
    *
    *   \param aCellOrdinal      cell (i.e. element) ordinal
    *   \param aGradient         cell interpolation function gradients
    *   \param aNodalCellStates  cell nodal states
    *   \param aStateGradients   cell interpolated state at the cubature points
    *
    *******************************************************************************/
    template<typename GradientType, typename InStateType, typename OutStateType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarMultiVectorT<GradientType> & aGradient,
                                       const Plato::ScalarMultiVectorT<InStateType>  & aNodalCellStates,
                                       const Plato::ScalarMultiVectorT<OutStateType> & aStateGradients) const
    {
        for(Plato::OrdinalType tDofIndex = 0; tDofIndex < SpaceDim; ++tDofIndex)
        {
            aStateGradients(aCellOrdinal, tDofIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; ++tNodeIndex)
            {
                Plato::OrdinalType tLocalOrdinal = (tNodeIndex * NumDofsPerNode) + DofOffset;
                aStateGradients(aCellOrdinal, tDofIndex) += aNodalCellStates(aCellOrdinal, tLocalOrdinal) * aGradient(aCellOrdinal, tNodeIndex, tDofIndex);
            }
        }
    }
};
// class InterpolateGradientFromScalarNodal

} // namespace Plato

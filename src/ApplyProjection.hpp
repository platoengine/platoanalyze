/*
 * ApplyProjection.hpp
 *
 *  Created on: Apr 23, 2018
 */

#ifndef APPLYPROJECTION_HPP_
#define APPLYPROJECTION_HPP_

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Projection Functor.

 \brief Given a set of cell control values, apply Heaviside projection operator.
  Assumes single point integration.

 \tparam ProjectionFunction projection function class

 */
/******************************************************************************/
template<class ProjectionFunction>
class ApplyProjection
{
public:

    /***************************************************************************//**
     * \brief Constructor
    *******************************************************************************/
    ApplyProjection() :
            mProjectionFunction()
    {
    }
    
    /***************************************************************************//**
     * \brief Constructor
     *
     * \param [in] aProjectionFunction type of projection function
    *******************************************************************************/
    explicit ApplyProjection(const ProjectionFunction & aProjectionFunction) :
            mProjectionFunction(aProjectionFunction)
    {
    }


    /***************************************************************************//**
     * \brief Apply projection operator to element density.
     *
     * \tparam WeightScalarType forward automatic differentiation type
     *
     * \param [in] aCellOrdinal element index
     * \param [in] aControl     side sets database
     * \return element density
    *******************************************************************************/
    template<typename WeightScalarType>
    DEVICE_TYPE inline WeightScalarType
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<WeightScalarType> & aControl) const
    {
        WeightScalarType tCellDensity = 0.0;
        const Plato::OrdinalType tRangePolicy = aControl.extent(1);
        for(Plato::OrdinalType tIndex = 0; tIndex < tRangePolicy; tIndex++)
        {
            tCellDensity += aControl(aCellOrdinal, tIndex);
        }
        tCellDensity = (tCellDensity / tRangePolicy);
        tCellDensity = mProjectionFunction.apply(tCellDensity);

        return (tCellDensity);
    }

private:
    ProjectionFunction mProjectionFunction; /*!< projection function */
};
// class ApplyProjection

}
// namespace Plato

#endif /* APPLYPROJECTION_HPP_ */

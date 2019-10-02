#ifndef PRESSURE_GRADIENT_HPP
#define PRESSURE_GRADIENT_HPP

#include "plato/Simplex.hpp"
#include "plato/PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Pressure gradient functor.

 Given a gradient matrix, b, and state array, s, compute the pressure gradient, g.

 g_{i} = s_{e,I} b_{e,I,i}

 e:  element index
 I:  local node index
 i:  dimension index

 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class PressureGradient : public Plato::Simplex<SpaceDim>
{
private:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */

public:
    /******************************************************************************//**
     * \brief Compute pressure gradient at cubature point, i.e. integration point
     * \param [in] aCellOrdinal cell (i.e. element) ordinal
     * \param [in/out] aPressureGrad pressure gradient workset
     * \param [in] aState state workset
     * \param [in] aGradient configuration gradient workset
    **********************************************************************************/
    template<typename ResultScalarType, typename StateScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void operator()(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarMultiVectorT<ResultScalarType> const& aPressureGrad,
                                       Plato::ScalarMultiVectorT<StateScalarType> const& aState,
                                       Plato::ScalarArray3DT<GradientScalarType> const& aGradient) const
    {
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            aPressureGrad(aCellOrdinal, tDimIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                aPressureGrad(aCellOrdinal, tDimIndex) += aState(aCellOrdinal, tNodeIndex)
                        * aGradient(aCellOrdinal, tNodeIndex, tDimIndex);
            }
        }
    }
};
// class PressureGradient

}
// namespace Plato

#endif

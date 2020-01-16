#pragma once

#include "Simplex.hpp"

namespace Plato
{

/******************************************************************************/
/*! Project to node functor.

 Given values at gauss points, multiply by the basis functions to project
 to the nodes.
 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumDofsPerNode = SpaceDim, Plato::OrdinalType DofOffset = 0>
class ProjectToNode : public Plato::Simplex<SpaceDim>
{
private:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */

public:
    /******************************************************************************//**
     * \brief Project state node values to cubature points (i.e. Gauss points)
     * \param [in] aCellOrdinal cell (i.e. element) ordinal
     * \param [in] aCellVolume cell (i.e. element) volume workset
     * \param [in] aBasisFunctions basis functions
     * \param [in] aStateValues 2D state values workset
     * \param [in/out] aResult output, state values at cubature points
     * \param [in] aScale scale parameter (default = 1.0)
    **********************************************************************************/
    template<typename GaussPointScalarType, typename ProjectedScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                                       const Plato::ScalarVectorT<Plato::Scalar> & aBasisFunctions,
                                       const Plato::ScalarMultiVectorT<GaussPointScalarType> & aStateValues,
                                       const Plato::ScalarMultiVectorT<ProjectedScalarType> & aResult,
                                       Plato::Scalar aScale = 1.0) const
    {
        const Plato::OrdinalType tNumDofs = aStateValues.extent(1);
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < tNumDofs; tDofIndex++)
            {
                Plato::OrdinalType tMyDofIndex = (NumDofsPerNode * tNodeIndex) + tDofIndex + DofOffset;
                aResult(aCellOrdinal, tMyDofIndex) += aScale * aBasisFunctions(tNodeIndex)
                        * aStateValues(aCellOrdinal, tDofIndex) * aCellVolume(aCellOrdinal);
            }
        }
    }

    /******************************************************************************//**
     * \brief Project state node values to cubature points (i.e. Gauss points)
     * \param [in] aCellOrdinal cell (i.e. element) ordinal
     * \param [in] aCellVolume cell (i.e. element) volume workset
     * \param [in] aBasisFunctions basis functions
     * \param [in] aStateValues 1D state values workset
     * \param [in/out] aResult output, state values at cubature points
     * \param [in] aScale scale parameter (default = 1.0)
    **********************************************************************************/
    template<typename GaussPointScalarType, typename ProjectedScalarType, typename VolumeScalarType>
    DEVICE_TYPE inline void operator()(const Plato::OrdinalType & aCellOrdinal,
                                       const Plato::ScalarVectorT<VolumeScalarType> & aCellVolume,
                                       const Plato::ScalarVectorT<Plato::Scalar> & aBasisFunctions,
                                       const Plato::ScalarVectorT<GaussPointScalarType> & aStateValues,
                                       const Plato::ScalarMultiVectorT<ProjectedScalarType> & aResult,
                                       Plato::Scalar aScale = 1.0) const
    {
        for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
        {
            Plato::OrdinalType tMyDofIndex = (NumDofsPerNode * tNodeIndex) + DofOffset;
            aResult(aCellOrdinal, tMyDofIndex) += aScale * aBasisFunctions(tNodeIndex)
                    * aStateValues(aCellOrdinal) * aCellVolume(aCellOrdinal);
        }
    }
};
// class ProjectToNode

}
// namespace Plato

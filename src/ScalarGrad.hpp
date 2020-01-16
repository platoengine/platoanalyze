#ifndef SCALAR_GRAD_HPP
#define SCALAR_GRAD_HPP

#include "PlatoStaticsTypes.hpp"
#include <Omega_h_vector.hpp>

namespace Plato
{

/******************************************************************************/
/*! \brief Scalar gradient functor.
 *
 *  Given a gradient matrix and scalar field, compute the scalar gradient.
 *
 *  \tparam SpaceDim spatial dimensions
 *
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ScalarGrad
{
private:
    static constexpr auto mNumNodesPerCell = SpaceDim + 1; /*!< number of nodes per cell */

public:
    /***********************************************************************************
     * \brief Compute scalar field gradient
     * \param [in] aCellOrdinal cell ordinal
     * \param [in/out] aOutput scalar field gradient workset
     * \param [in] aScalarField scalar field workset
     * \param [in] aGradient configuration gradient workset
     **********************************************************************************/
    template<typename ScalarType>
    DEVICE_TYPE inline void
    operator()(Plato::OrdinalType aCellOrdinal,
               Kokkos::View<ScalarType**, Kokkos::LayoutRight, Plato::MemSpace> aOutput,
               Kokkos::View<ScalarType**, Kokkos::LayoutRight, Plato::MemSpace> aScalarField,
               Omega_h::Vector<SpaceDim>* aConfigGrad) const
    {
        // compute scalar gradient
        //
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            aOutput(aCellOrdinal, tDimIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                aOutput(aCellOrdinal, tDimIndex) += aScalarField(aCellOrdinal, tNodeIndex) * aConfigGrad[tNodeIndex][tDimIndex];
            }
        }
    }

    /***********************************************************************************
     * \brief Compute scalar field gradient
     * \param [in] aCellOrdinal cell ordinal
     * \param [in/out] aOutput scalar field gradient workset
     * \param [in] aScalarField scalar field workset
     * \param [in] aGradient configuration gradient workset
     **********************************************************************************/
    template<typename ScalarGradType, typename ScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void
    operator()(Plato::OrdinalType aCellOrdinal,
               Plato::ScalarMultiVectorT<ScalarGradType> aOutput,
               Plato::ScalarMultiVectorT<ScalarType> aScalarField,
               Plato::ScalarArray3DT<GradientScalarType> aGradient) const
    {
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            aOutput(aCellOrdinal, tDimIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                aOutput(aCellOrdinal, tDimIndex) += aScalarField(aCellOrdinal, tNodeIndex)
                        * aGradient(aCellOrdinal, tNodeIndex, tDimIndex);
            }
        }
    }

    /***********************************************************************************
     * \brief Compute scalar field gradient
     *
     * \tparam ScalarGradType     POD type for 2-D Kokkos::View
     * \tparam ScalarType         POD type for 2-D Kokkos::View
     * \tparam GradientScalarType POD type for 3-D Kokkos::View
     *
     * \param [in]     aCellOrdinal    cell ordinal
     * \param [in]     aNumDofsPerNode number of degree of freedom per node
     * \param [in]     aScalarOffset   scalar degree of freedom offset
     * \param [in]     aScalarField    scalar field workset
     * \param [in]     aConfigGradient configuration gradient workset
     * \param [in/out] aScalarGradient scalar field gradient workset
     *
     **********************************************************************************/
    template<typename ScalarGradType, typename ScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::OrdinalType & aNumDofsPerNode,
               const Plato::OrdinalType & aScalarOffset,
               const Plato::ScalarMultiVectorT<ScalarType> & aScalarField,
               const Plato::ScalarArray3DT<GradientScalarType> & aConfigGradient,
               const Plato::ScalarMultiVectorT<ScalarGradType> & aScalarGradient) const
    {
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            aScalarGradient(aCellOrdinal, tDimIndex) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                Plato::OrdinalType tLocalOrdinal = tNodeIndex * aNumDofsPerNode + aScalarOffset;
                aScalarGradient(aCellOrdinal, tDimIndex) += aScalarField(aCellOrdinal, tLocalOrdinal)
                        * aConfigGradient(aCellOrdinal, tNodeIndex, tDimIndex);
            }
        }
    }
};
// class ScalarGrad

}
// namespace Plato

#endif

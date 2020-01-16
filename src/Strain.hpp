#pragma once

#include "SimplexMechanics.hpp"
#include "PlatoStaticsTypes.hpp"

#include <Omega_h_vector.hpp>

namespace Plato
{

/******************************************************************************/
/*! Strain functor.

 Given a gradient matrix and displacement array, compute the strain.
 strain tensor in Voigt notation = {e_xx, e_yy, e_zz, e_yz, e_xz, e_xy}

 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumDofsPerNode = SpaceDim>
class Strain : public Plato::SimplexMechanics<SpaceDim>
{
private:

    using Plato::SimplexMechanics<SpaceDim>::mNumVoigtTerms;   /*!< number of Voigt terms */
    using Plato::SimplexMechanics<SpaceDim>::mNumDofsPerCell;  /*!< number of degrees of freedom per cell */
    using Plato::SimplexMechanics<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */

public:
    /***************************************************************************//**
     * \brief Compute Cauchy strain tensor - Voigt notation used herein.
     * \param [in]     aCellOrdinal    cell ordinal
     * \param [in/out] aStrain         Cauchy strain tensor
     * \param [in]     aState          cell states
     * \param [in]     aGradientMatrix spatial gradient matrix
    *******************************************************************************/
    template<typename ScalarType>
    DEVICE_TYPE inline void operator()(Plato::OrdinalType aCellOrdinal,
                                       Kokkos::View<ScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& aStrain,
                                       Kokkos::View<ScalarType**, Kokkos::LayoutRight, Plato::MemSpace> const& aState,
                                       Omega_h::Vector<mNumVoigtTerms> const* aGradientMatrix) const
    {

        // compute strain
        //
        for(Plato::OrdinalType tVoigtIndex = 0; tVoigtIndex < mNumVoigtTerms; tVoigtIndex++)
        {
            aStrain(aCellOrdinal, tVoigtIndex) = 0.0;
            for(Plato::OrdinalType tDofIndex = 0; tDofIndex < mNumDofsPerCell; tDofIndex++)
            {
                aStrain(aCellOrdinal, tVoigtIndex) += aState(aCellOrdinal, tDofIndex) * aGradientMatrix[tDofIndex][tVoigtIndex];
            }
        }
    }

    template<typename StrainScalarType, typename DispScalarType, typename GradientScalarType>
    DEVICE_TYPE inline void operator()(Plato::OrdinalType aCellOrdinal,
                                       Plato::ScalarMultiVectorT<StrainScalarType> const& aStrain,
                                       Plato::ScalarMultiVectorT<DispScalarType> const& aState,
                                       Plato::ScalarArray3DT<GradientScalarType> const& aGradient) const
    {
        /***************************************************************************//**
         * \brief Compute Cauchy strain tensor - Voigt notation used herein.
         * \param [in]     aCellOrdinal cell ordinal
         * \param [in/out] aStrain      Cauchy strain tensor
         * \param [in]     aState       cell states
         * \param [in]     aGradient    spatial gradient matrix
        *******************************************************************************/
        Plato::OrdinalType tVoigtTerm = 0;
        for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
        {
            aStrain(aCellOrdinal, tVoigtTerm) = 0.0;
            for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
            {
                auto tLocalOrdinal = tNodeIndex * NumDofsPerNode + tDimIndex;
                aStrain(aCellOrdinal, tVoigtTerm) +=
                        aState(aCellOrdinal, tLocalOrdinal) * aGradient(aCellOrdinal, tNodeIndex, tDimIndex);
            }
            tVoigtTerm++;
        }

        for(Plato::OrdinalType tDofIndexJ = SpaceDim - 1; tDofIndexJ >= 1; tDofIndexJ--)
        {
            for(Plato::OrdinalType tDofIndexI = tDofIndexJ - 1; tDofIndexI >= 0; tDofIndexI--)
            {
                for(Plato::OrdinalType tNodeIndex = 0; tNodeIndex < mNumNodesPerCell; tNodeIndex++)
                {
                    auto tLocalOrdinalI = tNodeIndex * NumDofsPerNode + tDofIndexI;
                    auto tLocalOrdinalJ = tNodeIndex * NumDofsPerNode + tDofIndexJ;
                    aStrain(aCellOrdinal, tVoigtTerm) += (aState(aCellOrdinal, tLocalOrdinalJ) * aGradient(aCellOrdinal, tNodeIndex, tDofIndexI)
                            + aState(aCellOrdinal, tLocalOrdinalI) * aGradient(aCellOrdinal, tNodeIndex, tDofIndexJ));
                }
                tVoigtTerm++;
            }
        }
    }
};
// class Strain

}
// namespace Plato

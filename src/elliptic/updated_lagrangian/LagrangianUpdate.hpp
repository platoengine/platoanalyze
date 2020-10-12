#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
template<int SpaceDim>
class LagrangianUpdate
/******************************************************************************/
{
public:
    /******************************************************************************/
    explicit 
    LagrangianUpdate()
    /******************************************************************************/
    {
    }
    /******************************************************************************/
    ~LagrangianUpdate()
    /******************************************************************************/
    {
    }

    /******************************************************************************/
    void
    operator()(
        const Plato::DataMap      & aDataMap,
        const Plato::ScalarVector & aPreviousStrain,
        const Plato::ScalarVector & aUpdatedStrain
    )
    /******************************************************************************/
    {
        using ordT = Plato::ScalarVector::size_type;

        auto tStrainInc = aDataMap.scalarMultiVectors.at("strain increment");

        auto tNumCells = tStrainInc.extent(0);
        auto tNumTerms = tStrainInc.extent(1);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            for (ordT iTerm=0; iTerm<tNumTerms; iTerm++)
            {
                auto tOrdinal = aCellOrdinal * tNumTerms + iTerm;
                aUpdatedStrain(tOrdinal) = aPreviousStrain(tOrdinal) + tStrainInc(aCellOrdinal, iTerm);
            }
        }, "Update local state");
    }
};

} // namespace Plato

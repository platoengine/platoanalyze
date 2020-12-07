/*
 * PlatoMask.cpp
 *
 *  Created on: Nov 18, 2020
 */

#include "PlatoMask.hpp"

namespace Plato {

    BrickPrimitive::BrickPrimitive(
        Teuchos::ParameterList& aParams
    ) : mLimits(aParams)
    {
        if (!aParams.isType<std::string>("Operation")) {
            THROWERR("Primitive definition is missing required parameter 'Operation'");
        }

        auto tOperation = aParams.get<std::string>("Operation");
        if (tOperation == "Add")
        {
            mOperation = 1;
        }
        else
        if (tOperation == "Subtract")
        {
            mOperation = 0;
        }
        else
        {
            THROWERR("Primitive definition: 'Operation' must be either 'Add' or 'Subtract'");
        }
    }

    void
    BrickPrimitive::apply(
        Plato::LocalOrdinalVector aCellMask,
        Plato::ScalarMultiVector aCellCenters
    ) const
    {
        auto tLimits = mLimits;
        auto tOperation = mOperation;

        auto tNumCells = aCellCenters.extent(0);
        auto tNumDims = aCellCenters.extent(1);
        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            bool tInside = true;
            for (Plato::OrdinalType tDim=0; tDim<tNumDims; tDim++)
            {
                auto tVal = aCellCenters(aCellOrdinal, tDim);
                tInside = tInside && (tVal < tLimits.mMaximum[tDim]);
                tInside = tInside && (tVal > tLimits.mMinimum[tDim]);
            }
            if (tInside) aCellMask(aCellOrdinal) = tOperation;
        }, "cell mask");
    }

} // namespace Plato

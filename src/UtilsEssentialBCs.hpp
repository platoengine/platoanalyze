/*
 * UtilsEssentialBCs.hpp
 *
 *  Created on: Apr 23, 2021
 */

#pragma once

#include "BLAS1.hpp"

namespace Plato
{

struct Duplicates
{
    Plato::OrdinalType mNumDuplicates;
    Plato::LocalOrdinalVector mDofs;
    Plato::LocalOrdinalVector mPositions;
};
// struct Duplicates

/******************************************************************************//**
 * \fn Plato::Duplicates find_duplicate_dofs
 *
 * \brief Find duplicate degrees of freedom (dofs). Assumes all the dof ids are 
 * positive, i.e. dofs ids cannot be negative. 
 * \param [in] aDofs dofs array
 * \return database with duplicate dofs and respective locations
 *
 **********************************************************************************/
inline Plato::Duplicates 
find_duplicate_dofs
(const Plato::LocalOrdinalVector & aDofs)
{
    auto tNumElems = aDofs.size();
    Plato::LocalOrdinalVector tPositions("positions", tNumElems);
    Plato::blas1::fill(-1.0, tPositions);
    Plato::LocalOrdinalVector tDuplicates("duplicates", tNumElems);
    Plato::blas1::fill(-1.0, tDuplicates);
    Plato::LocalOrdinalVector tSum("number of duplicates", 1);
    
    auto tRange = tNumElems - static_cast<decltype(tNumElems)>(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tRange), LAMBDA_EXPRESSION(const decltype(tRange)& aIndexI)
    {
        for(decltype(tNumElems) tIndexJ = aIndexI + static_cast<decltype(tNumElems)>(1); tIndexJ < tNumElems; tIndexJ++)
        {
            if ( aDofs(aIndexI) == aDofs(tIndexJ) )
            {
                bool tFoundNumber = false;
                for(decltype(tNumElems) tIndex = 0; tIndex < tNumElems; tIndex++)
                {
                    if(tDuplicates(tIndex) == aDofs(aIndexI))
                    {
                        tFoundNumber = true;
                        break;
                    }
                }
                
                if(!tFoundNumber)
                {
                    tPositions(aIndexI) = aIndexI;
                    tDuplicates(aIndexI) = aDofs(aIndexI);
                    tSum(0) += 1;
                }
            }
        }        
    }, "find_duplicate_dofs");

    auto tHostSum = Kokkos::create_mirror(tSum);
    Kokkos::deep_copy(tHostSum, tSum);

    Plato::Duplicates tOutput;
    tOutput.mNumDuplicates = tHostSum(0);
    tOutput.mDofs = tDuplicates;
    tOutput.mPositions = tPositions;

    return tOutput;
}
// function find_duplicate_dofs

inline void 
post_process_duplicate_dofs
(Plato::Duplicates& aDuplicates)
{
    auto tNumElems = aDuplicates.mDofs.size();
    Plato::LocalOrdinalVector tPositions("positions", aDuplicates.mNumDuplicates);
    Plato::LocalOrdinalVector tDuplicates("duplicates", aDuplicates.mNumDuplicates);

    Plato::LocalOrdinalVector tCounter("counter", 1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumElems), LAMBDA_EXPRESSION(const decltype(tNumElems)& aIndex)
    {
        if(aDuplicates.mDofs(aIndex) != -1)
        {
            tDuplicates(tCounter(0)) = aDuplicates.mDofs(aIndex);
            tPositions(tCounter(0)) = aDuplicates.mPositions(aIndex);
            tCounter(0) += 1;
        }
    },"post_process_duplicate_dofs");

    aDuplicates.mDofs = tDuplicates;
    aDuplicates.mPositions = tPositions;
}
// post_process_duplicate_dofs

inline void 
set_unique_dof_vals
(const Plato::Duplicates& aDuplicates,
 const Plato::LocalOrdinalVector& aDofs,
       Plato::ScalarVector& aDofVals)
{
    auto tNumDofs = aDofVals.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const decltype(tNumDofs)& aIndex)
    {
        for(decltype(tNumDofs) i = 0; i < aDuplicates.mNumDuplicates; i++)
        {
            if(aDofs(aIndex) == aDuplicates.mDofs(i))
            {
                aDofVals(aIndex) = aDofVals(aDuplicates.mPositions(i));
            }
        }
    },"set_unique_dof_vals");
}

inline void insertion_sort
(Plato::LocalOrdinalVector & aArray)
{
    auto tRange = aArray.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tRange), LAMBDA_EXPRESSION(const decltype(tRange)& aIndexI)
    {
        auto tKey = aArray(aIndexI);
        Plato::OrdinalType tIndexJ = aIndexI - static_cast<decltype(tRange)>(1);
 
        /* Move elements of arr[0..i-1], that are greater than key, to one position ahead of their current position */
        while (tIndexJ >= 0 && (aArray(tIndexJ) > tKey) )
        {
            auto tIndex = tIndexJ + static_cast<decltype(tIndexJ)>(1);
            aArray(tIndex) = aArray(tIndexJ);
            tIndexJ = tIndexJ - static_cast<decltype(tIndexJ)>(1);
        }

        auto tIndex = tIndexJ + static_cast<decltype(tIndexJ)>(1);
        aArray(tIndex) = tKey;
        
    }, "outer loop");
}

inline Plato::LocalOrdinalVector 
post_process_remove_duplicates
(const Plato::OrdinalType aNumUnique,
 const Plato::LocalOrdinalVector & aArray)
{
    Plato::LocalOrdinalVector tOut("output", aNumUnique);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumUnique), LAMBDA_EXPRESSION(const Plato::OrdinalType& aIndex)
    {
        tOut(aIndex) = aArray(aIndex);
    }, "set_output");
    return tOut;
}

inline Plato::OrdinalType
remove_duplicates
(Plato::LocalOrdinalVector & aArray)
{
    Plato::LocalOrdinalVector tJ("index j", 1);

    auto tLength = aArray.size();
    auto tRange = tLength - static_cast<decltype(tLength)>(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tRange), LAMBDA_EXPRESSION(const decltype(tRange)& aIndex)
    {
        if (aArray[aIndex] != aArray[aIndex + static_cast<decltype(tRange)>(1)])
        {
            aArray[tJ(0)++] = aArray[aIndex];
        }

        if(aIndex + static_cast<decltype(tRange)>(1) == tRange)
        {
            aArray[tJ(0)++] = aArray[tRange];
        }
    }, "remove_duplicates");

    auto tHostJ = Kokkos::create_mirror(tJ);
    Kokkos::deep_copy(tHostJ, tJ);
    return tHostJ(0);
}

inline void 
post_process_dirichlet_dofs
(const Plato::LocalOrdinalVector & aDofs,
       Plato::ScalarVector       & aDofVals)
{
    auto tDuplicates = Plato::find_duplicate_dofs(aDofs);
    Plato::post_process_duplicate_dofs(tDuplicates);
    Plato::set_unique_dof_vals(tDuplicates, aDofs, aDofVals);
}

}
// namespace Plato
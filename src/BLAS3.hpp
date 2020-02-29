/*
 * BLAS3.hpp
 *
 *  Created on: Feb 29, 2020
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Serial_Impl.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Serial_Impl.hpp"

#include <Kokkos_Concepts.hpp>
#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosSparse_spgemm.hpp"
#include "KokkosSparse_spadd.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include <KokkosKernels_IOUtils.hpp>

namespace Plato
{

/************************************************************************//**
 *
 * \brief Build a workset of identity matrices
 *
 * \tparam NumRowsPerCell number of rows per cell
 * \tparam NumColsPerCell number of columns per cell
 *
 * \param aNumCells [in]     number of cells
 * \param aIdentity [in/out] 3-D view, workset of identity matrices
 *
********************************************************************************/
template<Plato::OrdinalType NumRowsPerCell, Plato::OrdinalType NumColumnsPerCell>
inline void identity_workset(const Plato::OrdinalType& aNumCells, Plato::ScalarArray3D& aIdentity)
{
    if(aIdentity.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 3-D view is empty, i.e. size <= 0.\n")
    }
    if(aIdentity.extent(0) != aNumCells)
    {
        THROWERR("\nNumber of cell mismatch. Input array has different number of cells than input number of cell argument.\n")
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < NumRowsPerCell; tRowIndex++)
        {
            for(Plato::OrdinalType tColumnIndex = 0; tColumnIndex < NumColumnsPerCell; tColumnIndex++)
            {
                aIdentity(aCellOrdinal, tRowIndex, tColumnIndex) = tRowIndex == tColumnIndex ? 1.0 : 0.0;
            }
        }
    }, "identity workset");
}
// function identity_workset

/************************************************************************//**
 *
 * \brief Compute workset of matrix inverse
 *
 * \tparam NumRowsPerCell number of rows per cell
 * \tparam NumColsPerCell number of columns per cell
 * \tparam AViewType      Input matrix, as a 3-D Kokkos::View
 * \tparam BViewType      Output matrix, as a 3-D Kokkos::View
 *
 * \param aNumCells [in]     number of cells
 * \param aA        [in]     3-D view, matrix workset
 * \param aInverse  [in/out] 3-D view, matrix inverse workset
 *
********************************************************************************/
template<Plato::OrdinalType NumRowsPerCell, Plato::OrdinalType NumColumnsPerCell, class AViewType, class BViewType>
inline void inverse_matrix_workset(const Plato::OrdinalType& aNumCells, AViewType& aA, BViewType& aInverse)
{
    if(aA.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 3D array, i.e. matrix workset, size is zero.\n")
    }
    if(aInverse.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput 3D array, i.e. matrix workset, size is zero.\n")
    }
    if(aA.size() != aInverse.size())
    {
        THROWERR("\nInput and output views dimensions are different, i.e. Input.size != Output.size.\n")
    }

    Plato::identity_workset<NumRowsPerCell, NumColumnsPerCell>(aNumCells, aInverse);

    using namespace KokkosBatched;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tA = Kokkos::subview(aA, aCellOrdinal, Kokkos::ALL(), Kokkos::ALL());
        auto tAinv = Kokkos::subview(aInverse, aCellOrdinal, Kokkos::ALL(), Kokkos::ALL());

        const Plato::Scalar tAlpha = 1.0;
        SerialLU<Algo::LU::Blocked>::invoke(tA);
        SerialTrsm<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit   ,Algo::Trsm::Blocked>::invoke(tAlpha, tA, tAinv);
        SerialTrsm<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsm::Blocked>::invoke(tAlpha, tA, tAinv);
    }, "compute matrix inverse 3DView");
}
// function inverse_matrix_workset

}
// namespace Plato

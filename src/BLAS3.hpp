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

/******************************************************************************//**
 * \brief Set entries in 3-D array to a single value.
 *
 * \tparam NumRowsPerCell matrix number of rows
 * \tparam NumColumnsPerCell matrix number of columns
 * \tparam AViewType Output workset, as a 3-D Kokkos::View
 *
 * \param [in] aNumCells number of cells, i.e. elements
 * \param [in] aAlpha scalar multiplier
 * \param [in/out] aOutput 3-D matrix workset (NumCells, NumRowsPerCell, NumColumnsPerCell)
**********************************************************************************/
template<Plato::OrdinalType NumRowsPerCell, Plato::OrdinalType NumColumnsPerCell, class AViewType>
inline void fill_array_3D(const Plato::OrdinalType& aNumCells,
                                typename AViewType::const_value_type & aAlpha,
                                AViewType& aOutput)
{
    if(aOutput.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 3D array is empty, i.e. size <= 0.\n")
    }
    if(aNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInvalid number of input cells, i.e. elements. Value is <= 0.\n")
    }

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < NumRowsPerCell; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < NumColumnsPerCell; tColIndex++)
            {
                aOutput(aCellOrdinal, tRowIndex, tColIndex) = aAlpha;
            }
        }
    }, "fill matrix identity 3DView");
}
// function fill_array_3D

/******************************************************************************//**
 * \brief Add workset of matrices
 *
 * \tparam AViewType Input matrix, as a 3-D Kokkos::View
 * \tparam BViewType Output matrix, as a 3-D Kokkos::View
 *
 * \param [in]     aNumCells number of cells, i.e. elements
 * \param [in]     aAlpha    scalar multiplier
 * \param [in]     aA        3-D matrix workset (NumCells, NumRowsPerCell, NumColumnsPerCell)
 * \param [in]     aBeta     scalar multiplier
 * \param [in/out] aB        3-D matrix workset (NumCells, NumRowsPerCell, NumColumnsPerCell)
**********************************************************************************/
template<class AViewType, class BViewType>
inline void update_array_3D(const Plato::OrdinalType& aNumCells,
                            typename AViewType::const_value_type& aAlpha,
                            const AViewType& aA,
                            typename BViewType::const_value_type& aBeta,
                            const BViewType& aB)
{
    if(aA.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput 3D array is empty, i.e. size <= 0\n")
    }
    if(aB.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput 3D array is empty, i.e. size <= 0\n")
    }
    if(aA.extent(1) != aB.extent(1))
    {
        THROWERR("\nDimension mismatch, number of rows do not match.\n")
    }
    if(aA.extent(2) != aB.extent(2))
    {
        THROWERR("\nDimension mismatch, number of columns do not match.\n")
    }
    if(aNumCells <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nNumber of input cells, i.e. elements, is less or equal to zero.\n");
    }

    const auto tNumRows = aA.extent(1);
    const auto tNumCols = aA.extent(2);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                aB(aCellOrdinal, tRowIndex, tColIndex) = aAlpha * aA(aCellOrdinal, tRowIndex, tColIndex) +
                        aBeta * aB(aCellOrdinal, tRowIndex, tColIndex);
            }
        }
    }, "update matrix workset");
}
// function update_array_3D

/************************************************************************//**
 *
 * \brief Dense matrix-matrix multiplication:
 *     \f$ C = \beta*C + \alpha*op(A)*op(B) \f$
 * NOTE: Function does not support transpose operations
 *
 * \tparam AViewType Input matrix, as a 3-D Kokkos::View
 * \tparam BViewType Input matrix, as a 3-D Kokkos::View
 * \tparam CViewType Output matrix, as a nonconst 3-D Kokkos::View
 *
 * \param aNumCells [in]     Input number of cells, i.e. elements
 * \param aAlpha    [in]     Input coefficient of A
 * \param aA        [in]     Input matrix, as a 3-D Kokkos::View
 * \param aB        [in]     Input matrix, as a 3-D Kokkos::View
 * \param aBeta     [in]     Input coefficient of C
 * \param aC        [in/out] Output matrix, as a nonconst 3-D Kokkos::View
 *
****************************************************************************/
template<class AViewType, class BViewType, class CViewType>
inline void multiply_matrix_workset(const Plato::OrdinalType& aNumCells,
                                    typename AViewType::const_value_type& aAlpha,
                                    const AViewType& aA,
                                    const BViewType& aB,
                                    typename CViewType::const_value_type& aBeta,
                                    CViewType& aC)
{
    std::ostringstream tError;
    if(aA.size() <= static_cast<Plato::OrdinalType>(0))
    {
        tError << "\nInput 3D array A is empty, i.e. size <= 0.\n";
        THROWERR(tError.str())
    }
    if(aB.size() <= static_cast<Plato::OrdinalType>(0))
    {
        tError << "\nInput 3D array B is empty, i.e. size <= 0.\n";
        THROWERR(tError.str())
    }
    if(aC.size() <= static_cast<Plato::OrdinalType>(0))
    {
        tError << "\nOutput 3D array C is empty, i.e. size <= 0.\n";
        THROWERR(tError.str())
    }
    if(aA.extent(2) != aB.extent(1))
    {
        tError << "\nDimension mismatch: The number of columns in A matrix workset does not match "
            << "the number of rows in B matrix workset. " << "A has " << aA.extent(2) << " columns and B has "
            << aB.extent(1) << " rows.\n";
        THROWERR(tError.str())
    }
    if(aA.extent(1) != aC.extent(1))
    {
        tError << "\nDimension mismatch. Mismatch in input (A) and output (C) matrices row count. "
            << "A has " << aA.extent(1) << " rows and C has " << aC.extent(1) << " rows.\n";
        THROWERR(tError.str())
    }
    if(aB.extent(2) != aC.extent(2))
    {
        tError << "\nDimension mismatch. Mismatch in input (B) and output (C) matrices column count. "
            << "A has " << aA.extent(2) << " columns and C has " << aC.extent(2) << " columns.\n";
        THROWERR(tError.str())
    }
    if(aA.extent(0) != aNumCells)
    {
        tError << "\nDimension mismatch, number of cells of matrix A does not match input number of cells. "
            << "A has " << aA.extent(0) << " and the input number of cells is set to " << aNumCells << "\n.";
        THROWERR(tError.str())
    }
    if(aB.extent(0) != aNumCells)
    {
        tError << "\nDimension mismatch, number of cells of matrix B does not match input number of cells. "
            << "B has " << aB.extent(0) << " and the input number of cells is set to " << aNumCells << "\n.";
        THROWERR(tError.str())
    }
    if(aC.extent(0) != aNumCells)
    {
        tError << "\nDimension mismatch, number of cells of matrix C does not match input number of cells. "
            << "C has " << aC.extent(0) << " and the input number of cells is set to " << aNumCells << "\n.";
        THROWERR(tError.str())
    }

    const auto tNumOutRows = aC.extent(1);
    const auto tNumOutCols = aC.extent(2);
    const auto tNumInnerCols = aA.extent(2);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumOutRows; tRowIndex++)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumOutCols; tColIndex++)
            {
                aC(aCellOrdinal, tRowIndex, tColIndex) = aBeta * aC(aCellOrdinal, tRowIndex, tColIndex);
            }
        }

        for(Plato::OrdinalType tOutRowIndex = 0; tOutRowIndex < tNumOutRows; tOutRowIndex++)
        {
            for(Plato::OrdinalType tOutColIndex = 0; tOutColIndex < tNumOutCols; tOutColIndex++)
            {
                Plato::Scalar tValue = 0.0;
                for(Plato::OrdinalType tCommonIndex = 0; tCommonIndex < tNumInnerCols; tCommonIndex++)
                {
                    tValue += aAlpha * aA(aCellOrdinal, tOutRowIndex, tCommonIndex) * aB(aCellOrdinal, tCommonIndex, tOutColIndex);
                }
                aC(aCellOrdinal, tOutRowIndex, tOutColIndex) += tValue;
            }
        }
    }, "multiply matrix workset");
}
// function multiply_matrix_workset

}
// namespace Plato

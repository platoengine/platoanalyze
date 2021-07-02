/*
 * PlatoMathHelpers.hpp
 *
 *  Created on: April 19, 2018
 */

#ifndef PLATOMATHHELPERS_HPP_
#define PLATOMATHHELPERS_HPP_

#include <sstream>
#include <cassert>

#include <Kokkos_Macros.hpp>
#include <KokkosKernels_SparseUtils.hpp>
#include <KokkosSparse_spgemm.hpp>
#include <KokkosSparse_spadd.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

#include "PlatoStaticsTypes.hpp"
#include "PlatoMathFunctors.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Device only function used to compare two values (conditional values)
 * between themselves and return the decision (consequent value). The conditional
 * expression evaluated in this function is defined as if(X > Y) A = B.
 * \param [in] aConditionalValOne conditional value given by X
 * \param [in] aConditionalValTwo conditional value given by Y
 * \param [in] aConsequentValOne consequent value given by A
 * \param [in] aConsequentValTwo consequent value given by B
 * \return result/decision
**********************************************************************************/
DEVICE_TYPE inline Plato::Scalar
conditional_expression(const Plato::Scalar & aX,
                       const Plato::Scalar & aY,
                       const Plato::Scalar & aA,
                       const Plato::Scalar & aB)
{
    auto tConditionalExpression = aX - aY - static_cast<Plato::Scalar>(1.0);
    tConditionalExpression = exp(tConditionalExpression);
    Plato::OrdinalType tCoeff = fmin(static_cast<Plato::Scalar>(1.0), tConditionalExpression);
    Plato::Scalar tScalarCoeff = tCoeff;
    auto tOutput = tScalarCoeff * aB + (static_cast<Plato::Scalar>(1.0) - tScalarCoeff) * aA;
    return (tOutput);
}
// function conditional_expression

/******************************************************************************//**
 * \brief Matrix times vector plus vector
 * \param [in] aMatrix multiplier of 1D container A
 * \param [in] aInput input 1D container
 * \param [out] aOutput output 1D container
**********************************************************************************/
template<typename ScalarT>
void MatrixTimesVectorPlusVector(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
                                 const Plato::ScalarVectorT<ScalarT> & aInput,
                                 const Plato::ScalarVectorT<ScalarT> & aOutput)
{
    if(aMatrix->numCols() != aInput.size())
    {
        std::ostringstream tMsg;
        tMsg << "DIMENSION MISMATCH.  INPUT VECTOR LENGTH DOES NOT MATCH THE NUMBER OF COLUMNS IN MATRIX A.  "
             << "INPUT VECTOR LENGTH = '" <<  aInput.size() << "' AND THE NUMBER OF COLUMNS IN MATRIX A = '"
             << aMatrix->numCols() << "'. INPUT VECTOR LABEL IS '" << aInput.label()
             << "' AND OUTPUT VECTOR LABEL IS '" << aOutput.label() << "'.";
        THROWERR(tMsg.str());
    }
    if(aMatrix->numRows() != aOutput.size())
    {
        std::ostringstream tMsg;
        tMsg << "DIMENSION MISMATCH.  OUTPUT VECTOR LENGTH DOES NOT MATCH THE NUMBER OF ROWS IN MATRIX A.  "
             << "OUTPUT VECTOR LENGTH = '" <<  aOutput.size() << "' AND THE NUMBER OF ROWS IN MATRIX A = '"
             << aMatrix->numRows() << "'. OUTPUT VECTOR LABEL IS '" << aOutput.label() << "'.";
        THROWERR(tMsg.str());
    }

    if(aMatrix->isBlockMatrix())
    {
        auto tNodeRowMap = aMatrix->rowMap();
        auto tNodeColIndices = aMatrix->columnIndices();
        auto tNumRowsPerBlock = aMatrix->numRowsPerBlock();
        auto tNumColsPerBlock = aMatrix->numColsPerBlock();
        auto tEntries = aMatrix->entries();
        auto tNumNodeRows = tNodeRowMap.size() - 1;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodeRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeRowOrdinal)
        {
            auto tRowStartIndex = tNodeRowMap(aNodeRowOrdinal);
            auto tRowEndIndex = tNodeRowMap(aNodeRowOrdinal + 1);
            for (auto tCrsIndex = tRowStartIndex; tCrsIndex < tRowEndIndex; tCrsIndex++)
            {
                auto tNodeColumnIndex = tNodeColIndices(tCrsIndex);

                auto tFromDofColIndex = tNumColsPerBlock*tNodeColumnIndex;
                auto tToDofColIndex = tFromDofColIndex + tNumColsPerBlock;

                auto tFromDofRowIndex = tNumRowsPerBlock*aNodeRowOrdinal;
                auto tToDofRowIndex = tFromDofRowIndex + tNumRowsPerBlock;

                auto tMatrixEntryIndex = tNumRowsPerBlock*tNumColsPerBlock*tCrsIndex;
                for ( auto tDofRowIndex = tFromDofRowIndex; tDofRowIndex < tToDofRowIndex; tDofRowIndex++ )
                {
                    ScalarT tSum = 0.0;
                    for ( auto tDofColIndex = tFromDofColIndex; tDofColIndex < tToDofColIndex; tDofColIndex++ )
                    {
                        tSum += tEntries(tMatrixEntryIndex) * aInput(tDofColIndex);
                        tMatrixEntryIndex += 1;
                    }
                    aOutput(tDofRowIndex) += tSum;
                }
            }
        }, "BlockMatrix * Vector_a + Vector_b");
    }
    else
    {
        auto tRowMap = aMatrix->rowMap();
        auto tColIndices = aMatrix->columnIndices();
        auto tEntries = aMatrix->entries();
        auto tNumRows = tRowMap.size() - 1;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aRowOrdinal)
        {
            auto tRowStart = tRowMap(aRowOrdinal);
            auto tRowEnd = tRowMap(aRowOrdinal + 1);
            ScalarT tSum = 0.0;
            for (auto tEntryIndex = tRowStart; tEntryIndex < tRowEnd; tEntryIndex++)
            {
                auto tColumnIndex = tColIndices(tEntryIndex);
                tSum += tEntries(tEntryIndex) * aInput(tColumnIndex);
            }
            aOutput(aRowOrdinal) += tSum;
        },"Matrix * Vector_a + Vector_b");
    }
}
// function MatrixTimesVectorPlusVector

/******************************************************************************//**
 * \brief Vector times Matrix plus vector
 * \param [in] aInput input 1D container
 * \param [in] aMatrix multiplier of 1D container A
 * \param [out] aOutput output 1D container
**********************************************************************************/
template<typename ScalarT>
void VectorTimesMatrixPlusVector(
    const Plato::ScalarVectorT<ScalarT>      & aInput,
    const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
    const Plato::ScalarVectorT<ScalarT>      & aOutput)
{
    if(aMatrix->numRows() != aInput.size())
    {
        std::ostringstream tMsg;
        tMsg << "DIMENSION MISMATCH.  INPUT VECTOR LENGTH DOES NOT MATCH THE NUMBER OF ROWS IN MATRIX A.  "
             << "INPUT VECTOR LENGTH = '" <<  aInput.size() << "' AND THE NUMBER OF ROWS IN MATRIX A = '"
             << aMatrix->numCols() << "'. INPUT VECTOR LABEL IS '" << aInput.label()
             << "' AND OUTPUT VECTOR LABEL IS '" << aOutput.label() << "'.";
        THROWERR(tMsg.str());
    }
    if(aMatrix->numCols() != aOutput.size())
    {
        std::ostringstream tMsg;
        tMsg << "DIMENSION MISMATCH.  OUTPUT VECTOR LENGTH DOES NOT MATCH THE NUMBER OF COLUMNS IN MATRIX A.  "
             << "OUTPUT VECTOR LENGTH = '" <<  aOutput.size() << "' AND THE NUMBER OF COLUMNS IN MATRIX A = '"
             << aMatrix->numRows() << "'. OUTPUT VECTOR LABEL IS '" << aOutput.label() << "'.";
        THROWERR(tMsg.str());
    }

    if(aMatrix->isBlockMatrix())
    {
        auto tNodeRowMap = aMatrix->rowMap();
        auto tNodeColIndices = aMatrix->columnIndices();
        auto tNumRowsPerBlock = aMatrix->numRowsPerBlock();
        auto tNumColsPerBlock = aMatrix->numColsPerBlock();
        auto tEntries = aMatrix->entries();
        auto tNumNodeRows = tNodeRowMap.size() - 1;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodeRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeRowOrdinal)
        {
            auto tRowStartIndex = tNodeRowMap(aNodeRowOrdinal);
            auto tRowEndIndex = tNodeRowMap(aNodeRowOrdinal + 1);
            for (auto tCrsIndex = tRowStartIndex; tCrsIndex < tRowEndIndex; tCrsIndex++)
            {
                auto tNodeColumnIndex = tNodeColIndices(tCrsIndex);

                auto tFromDofColIndex = tNumColsPerBlock*tNodeColumnIndex;
                auto tToDofColIndex = tFromDofColIndex + tNumColsPerBlock;

                auto tFromDofRowIndex = tNumRowsPerBlock*aNodeRowOrdinal;
                auto tToDofRowIndex = tFromDofRowIndex + tNumRowsPerBlock;

                auto tMatrixEntryIndex = tNumRowsPerBlock*tNumColsPerBlock*tCrsIndex;
                for ( auto tDofRowIndex = tFromDofRowIndex; tDofRowIndex < tToDofRowIndex; tDofRowIndex++ )
                {
                    for ( auto tDofColIndex = tFromDofColIndex; tDofColIndex < tToDofColIndex; tDofColIndex++ )
                    {
                        Kokkos::atomic_add(&aOutput(tDofColIndex), tEntries(tMatrixEntryIndex) * aInput(tDofRowIndex));
                        tMatrixEntryIndex += 1;
                    }
                }
            }
        }, "Vector_a * BlockMatrix + Vector_b");
    }
    else
    {
        auto tRowMap = aMatrix->rowMap();
        auto tColIndices = aMatrix->columnIndices();
        auto tEntries = aMatrix->entries();
        auto tNumRows = tRowMap.size() - 1;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aRowOrdinal)
        {
            auto tRowStart = tRowMap(aRowOrdinal);
            auto tRowEnd = tRowMap(aRowOrdinal + 1);
            for (auto tEntryIndex = tRowStart; tEntryIndex < tRowEnd; tEntryIndex++)
            {
                auto tColumnIndex = tColIndices(tEntryIndex);
                Kokkos::atomic_add(&aOutput(tColumnIndex), tEntries(tEntryIndex) * aInput(aRowOrdinal));
            }
        },"Vector_a * Matrix + Vector_b");
    }
}
// function MatrixTimesVectorPlusVector
/******************************************************************************//**
 * \brief Compute row sum inverse product
 * \param [in] aA
 * \param [in/out] aB
 *
 * aB = RowSum(aA)^{-1} . aB
 *
**********************************************************************************/
inline void
RowSummedInverseMultiply( const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
                                Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo)
{
    auto tNumBlockRows = aInMatrixOne->rowMap().size()-1;
    auto tNumRows = aInMatrixOne->numRows();
    Plato::RowSum tRowSumFunctor(aInMatrixOne);
    Plato::DiagonalInverseMultiply tDiagInverseMultiplyFunctor(aInMatrixTwo);
    Plato::ScalarVector tRowSum("row sum", tNumRows);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumBlockRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tBlockRowOrdinal) {
      tRowSumFunctor(tBlockRowOrdinal, tRowSum);
      tDiagInverseMultiplyFunctor(tBlockRowOrdinal, tRowSum);
    });
}


/******************************************************************************/
/*! 
  \brief Extract block matrix graph and values into a non-block matrix format
  \param [in] aMatrix Block matrix from which graph and values will be extracted
  \param [out] aMatrixRowMap Non-block row map: (rowIndex) => {offsetIndex}
  \param [out] aMatrixColMap Non-block column map: (offsetIndex) => {columnIndex}
  \param [out] aMatrixValues Non-block matrix entries: (offsetIndex) => {matrix value}
  \param [in] aRowStride If greater than one, the input matrix is considered a sub-block matrix.
  \param [in] aRowOffset Offset of the submatrix.  Ignored if aRowStride equals one.

  The column map for the block matrix provides the node column index.  The matrix 
  values are indexed by offsetIndex*numColsPerBlock*numRowsPerBlock + blockRowIndex*numColsPerBlock + blockColIndex.
  The output matrix values are indexed by offsetIndex.

  If aRowStride is greater than one, the input matrix is assumed to be a sub-block matrix, i.e., 
  the block data provided for each node is only one row of the full block matrix.  The value of
  aRowStride is the number of rows in the full block matrix, and aRowOffset is the index of the
  sub-matrix row within the full block matrix.  The resulting non-block maps are for the full
  block matrix.
*/
/******************************************************************************/
inline void
getDataAsNonBlock( const Teuchos::RCP<Plato::CrsMatrixType>       & aMatrix,
                         Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixRowMap,
                         Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixColMap,
                         Plato::ScalarVectorT<Plato::Scalar>      & aMatrixValues,
                         int aRowStride=1, int aRowOffset=0)
{
    const auto& tRowMap = aMatrix->rowMap();
    const auto& tColMap = aMatrix->columnIndices();
    const auto& tValues = aMatrix->entries();

    auto tNumMatrixRows = aMatrix->numRows();
    tNumMatrixRows *= aRowStride;

    auto tNumRowsPerBlock = aMatrix->numRowsPerBlock();
    auto tNumColsPerBlock = aMatrix->numColsPerBlock();
    auto tBlockSize = tNumRowsPerBlock*tNumColsPerBlock;


    // generate non block row map
    //
    aMatrixRowMap = Plato::ScalarVectorT<Plato::OrdinalType>("non block row map", tNumMatrixRows+1);

    if (aRowStride == 1)
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMatrixRowIndex) {
            auto tBlockRowIndex = tMatrixRowIndex / tNumRowsPerBlock;
            auto tLocalRowIndex = tMatrixRowIndex % tNumRowsPerBlock;
            auto tFrom = tRowMap(tBlockRowIndex);
            auto tTo   = tRowMap(tBlockRowIndex+1);
            auto tBlockRowSize = tTo - tFrom;
            aMatrixRowMap(tMatrixRowIndex) = tFrom * tBlockSize + tLocalRowIndex * tBlockRowSize * tNumColsPerBlock;
            aMatrixRowMap(tMatrixRowIndex+1) = tFrom * tBlockSize + (tLocalRowIndex+1) * tBlockRowSize * tNumColsPerBlock;
        });
    }
    else 
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMatrixRowIndex) {
            auto tBlockRowIndex = tMatrixRowIndex / aRowStride;
            auto tLocalRowIndex = tMatrixRowIndex % aRowStride;
            auto tFrom = tRowMap(tBlockRowIndex);
            auto tTo   = tRowMap(tBlockRowIndex+1);
            auto tBlockRowSize = tTo - tFrom;
            aMatrixRowMap(tMatrixRowIndex) = tFrom * aRowStride * tNumColsPerBlock + tLocalRowIndex * tBlockRowSize * tNumColsPerBlock;
            aMatrixRowMap(tMatrixRowIndex+1) = tFrom * aRowStride * tNumColsPerBlock + (tLocalRowIndex+1) * tBlockRowSize * tNumColsPerBlock;
        });
    }

    // generate non block col map and non block values
    //
    auto tNumMatrixColEntries = tColMap.extent(0)*tBlockSize*aRowStride;
    aMatrixColMap = Plato::ScalarVectorT<Plato::OrdinalType>("non block col map", tNumMatrixColEntries);
    aMatrixValues = Plato::ScalarVectorT<Plato::Scalar>     ("non block values",  tNumMatrixColEntries);

    if (aRowStride == 1)
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMatrixRowIndex) {
            auto tBlockRowIndex = tMatrixRowIndex / tNumRowsPerBlock;
            auto tLocalRowIndex = tMatrixRowIndex % tNumRowsPerBlock;
            auto tFrom = tRowMap(tBlockRowIndex);
            auto tTo   = tRowMap(tBlockRowIndex+1);
            Plato::OrdinalType tMatrixRowFrom = aMatrixRowMap(tMatrixRowIndex);
            for( auto tColMapIndex=tFrom; tColMapIndex<tTo; ++tColMapIndex )
            {
                for( Plato::OrdinalType tBlockColOffset=0; tBlockColOffset<tNumColsPerBlock; ++tBlockColOffset )
                {
                    auto tMapIndex = tColMap(tColMapIndex)*tNumColsPerBlock+tBlockColOffset;
                    auto tValIndex = tColMapIndex*tBlockSize+tLocalRowIndex*tNumColsPerBlock+tBlockColOffset;
                    aMatrixColMap(tMatrixRowFrom) = tMapIndex;
                    aMatrixValues(tMatrixRowFrom) = tValues(tValIndex);
                    tMatrixRowFrom++;
                }
            }
        });
    }
    else
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMatrixRowIndex) {
            auto tBlockRowIndex = tMatrixRowIndex / aRowStride;
            auto tLocalRowIndex = tMatrixRowIndex % aRowStride;
            auto tFrom = tRowMap(tBlockRowIndex);
            auto tTo   = tRowMap(tBlockRowIndex+1);
            Plato::OrdinalType tMatrixRowFrom = aMatrixRowMap(tMatrixRowIndex);
            for( auto tColMapIndex=tFrom; tColMapIndex<tTo; ++tColMapIndex )
            {
                for( Plato::OrdinalType tBlockColOffset=0; tBlockColOffset<tNumColsPerBlock; ++tBlockColOffset )
                {
                    auto tMapIndex = tColMap(tColMapIndex)*tNumColsPerBlock+tBlockColOffset;
                    auto tValIndex = tColMapIndex*tNumColsPerBlock+tBlockColOffset;
                    aMatrixColMap(tMatrixRowFrom) = tMapIndex;
                    aMatrixValues(tMatrixRowFrom) = ((tLocalRowIndex == aRowOffset) ? tValues(tValIndex) : 0.0);
                    tMatrixRowFrom++;
                }
            }
        });
    }
}

/******************************************************************************//**
 * \brief Sort the column indices within each row to ascending order and apply
          the same map to the matrix entries array.
 * \param [in] aMatrixRowMap matrix row map
 * \param [in/out] aMatrixColMap matrix column map to be sorted
 * \param [in/out] aMatrixValues matrix values
 *
**********************************************************************************/
inline void
sortColumnEntries( const Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixRowMap,
                         Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixColMap,
                         Plato::ScalarVectorT<Plato::Scalar>      & aMatrixValues)
{
    auto tNumMatrixRows = aMatrixRowMap.extent(0) - 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMatrixRowIndex) {
        auto tFrom = aMatrixRowMap(tMatrixRowIndex);
        auto tTo   = aMatrixRowMap(tMatrixRowIndex+1);
        for( auto tColMapEntryIndex_I=tFrom; tColMapEntryIndex_I<tTo; ++tColMapEntryIndex_I )
        {
            for( auto tColMapEntryIndex_J=tFrom; tColMapEntryIndex_J<tTo; ++tColMapEntryIndex_J )
            {
                if( aMatrixColMap[tColMapEntryIndex_I] < aMatrixColMap[tColMapEntryIndex_J] )
                {
                    auto tColIndex = aMatrixColMap[tColMapEntryIndex_J];
                    aMatrixColMap[tColMapEntryIndex_J] = aMatrixColMap[tColMapEntryIndex_I];
                    aMatrixColMap[tColMapEntryIndex_I] = tColIndex;
                    auto tValue = aMatrixValues[tColMapEntryIndex_J];
                    aMatrixValues[tColMapEntryIndex_J] = aMatrixValues[tColMapEntryIndex_I];
                    aMatrixValues[tColMapEntryIndex_I] = tValue;
                }
            }
        }
    });
}

/******************************************************************************//**
 * \brief Set block matrix data from non-block data
 * \param [in/out] aMatrix
 * \param [in] aMatrixRowMap Non-block matrix row map
 * \param [in] aMatrixColMap Non-block matrix col map
 * \param [in] aMatrixValues Non-block matrix values
 *
**********************************************************************************/
inline void
setDataFromNonBlock(      Teuchos::RCP<Plato::CrsMatrixType>       & aMatrix,
                    const Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixRowMap,
                          Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixColMap,
                          Plato::ScalarVectorT<Plato::Scalar>      & aMatrixValues)
{

    sortColumnEntries(aMatrixRowMap, aMatrixColMap, aMatrixValues);

    auto tNumMatrixRows = aMatrix->numRows();
    auto tNumRowsPerBlock = aMatrix->numRowsPerBlock();
    auto tNumColsPerBlock = aMatrix->numColsPerBlock();
    auto tBlockSize = tNumRowsPerBlock*tNumColsPerBlock;

    // generate block row map
    //
    auto tNumNodeRows = tNumMatrixRows/tNumRowsPerBlock;
    Plato::ScalarVectorT<Plato::OrdinalType> tRowMap("block row map", tNumNodeRows+1);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows+1), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMatrixRowIndex) {
        auto tBlockRowIndex = tMatrixRowIndex / tNumRowsPerBlock;
        auto tLocalRowIndex = tMatrixRowIndex % tNumRowsPerBlock;
        if(tLocalRowIndex == 0)
          tRowMap(tBlockRowIndex) = aMatrixRowMap(tMatrixRowIndex)/tBlockSize;
    });
    aMatrix->setRowMap(tRowMap);

    // generate block col map and block values
    //
    auto tNumBlockMatEntries = aMatrixValues.extent(0);
    auto tNumBlockColEntries = tNumBlockMatEntries / tBlockSize;
    Plato::ScalarVectorT<Plato::OrdinalType> tColMap("block col map", tNumBlockColEntries);
    Plato::ScalarVectorT<Plato::Scalar>      tValues("block values",  tNumBlockMatEntries);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumMatrixRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMatrixRowIndex) {
        auto tBlockRowIndex = tMatrixRowIndex / tNumRowsPerBlock;
        auto tLocalRowIndex = tMatrixRowIndex % tNumRowsPerBlock;
        auto tFrom = tRowMap(tBlockRowIndex);
        auto tTo   = tRowMap(tBlockRowIndex+1);
        Plato::OrdinalType tMatrixRowFrom = aMatrixRowMap(tMatrixRowIndex);
        for( auto tColMapIndex=tFrom; tColMapIndex<tTo; ++tColMapIndex )
        {
            if(tLocalRowIndex == 0)
            {
                tColMap(tColMapIndex) = aMatrixColMap(tMatrixRowFrom)/tNumColsPerBlock;
            }
            for( Plato::OrdinalType tBlockColOffset=0; tBlockColOffset<tNumColsPerBlock; ++tBlockColOffset )
            {
                auto tValIndex = tColMapIndex*tBlockSize + tLocalRowIndex*tNumColsPerBlock + tBlockColOffset;
                tValues(tValIndex) = aMatrixValues(tMatrixRowFrom++);
            }
        }
    });
    aMatrix->setColumnIndices(tColMap);
    aMatrix->setEntries(tValues);
}

using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

/******************************************************************************//**
 * \brief Compute the matrix product
 * \param [in] aM1
 * \param [in] aM2
 * \param [out] aProduct
 *
 * aProduct = aM1 . aM2
 *
**********************************************************************************/
inline void
MatrixMatrixMultiply( const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
                      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo,
                            Teuchos::RCP<Plato::CrsMatrixType> & aOutMatrix,
                            SPGEMMAlgorithm aAlgorithm = SPGEMM_KK_SPEED)
{
    using Plato::OrdinalType;
    using Plato::Scalar;

    typedef Plato::ScalarVectorT<OrdinalType> OrdinalView;
    typedef Plato::ScalarVectorT<Scalar>  ScalarView;

    typedef KokkosKernels::Experimental::KokkosKernelsHandle
        <OrdinalType, OrdinalType, Scalar,
        typename Plato::ExecSpace, 
        typename Plato::MemSpace,
        typename Plato::MemSpace > KernelHandle;

    KernelHandle tKernel;
    tKernel.set_team_work_size(1);
    tKernel.set_dynamic_scheduling(false);

    tKernel.create_spgemm_handle(aAlgorithm);

    const auto& tMatOne = *aInMatrixOne;
    const auto& tMatTwo = *aInMatrixTwo;
    auto& tOutMat = *aOutMatrix;
    const OrdinalType tNumRowsOne = tMatOne.numRows();
    const OrdinalType tNumColsOne = tMatOne.numCols();
    const OrdinalType tNumRowsTwo = tMatTwo.numRows();
    const OrdinalType tNumColsTwo = tMatTwo.numCols();
    const OrdinalType tNumRowsOut = tOutMat.numRows();
    const OrdinalType tNumColsOut = tOutMat.numCols();

    // C = M1 x M2
    //
    // numCols(M1) === numRows(M2)
    if (tNumRowsTwo != tNumColsOne) { THROWERR("input matrices have incompatible shapes"); }

    // numRows(C)  === numRows(M1)
    if (tNumRowsOut != tNumRowsOne) { THROWERR("output matrix has incorrect shape"); }

    // numCols(C)  === numCols(M2)
    if (tNumColsOut != tNumColsTwo) { THROWERR("output matrix has incorrect shape"); }

    ScalarView tMatOneValues;
    OrdinalView tMatOneRowMap, tMatOneColMap;
    Plato::getDataAsNonBlock(aInMatrixOne, tMatOneRowMap, tMatOneColMap, tMatOneValues);

    ScalarView tMatTwoValues;
    OrdinalView tMatTwoRowMap, tMatTwoColMap;
    Plato::getDataAsNonBlock(aInMatrixTwo, tMatTwoRowMap, tMatTwoColMap, tMatTwoValues);

    OrdinalView tOutRowMap ("output row map", tNumRowsOne + 1);
    spgemm_symbolic ( &tKernel, tNumRowsOne, tNumRowsTwo, tNumColsTwo,
        tMatOneRowMap, tMatOneColMap, /*transpose=*/false,
        tMatTwoRowMap, tMatTwoColMap, /*transpose=*/false,
        tOutRowMap
    );

    OrdinalView tOutColMap;
    ScalarView  tOutValues;
    size_t tNumOutValues = tKernel.get_spgemm_handle()->get_c_nnz();
    if (tNumOutValues){
      tOutColMap = OrdinalView(Kokkos::ViewAllocateWithoutInitializing("out column map"), tNumOutValues);
      tOutValues = ScalarView (Kokkos::ViewAllocateWithoutInitializing("out values"),  tNumOutValues);
    }
    spgemm_numeric( &tKernel, tNumRowsOne, tNumRowsTwo, tNumColsTwo,
        tMatOneRowMap, tMatOneColMap, tMatOneValues, /*transpose=*/false,
        tMatTwoRowMap, tMatTwoColMap, tMatTwoValues, /*transpose=*/false,
        tOutRowMap, tOutColMap, tOutValues
    );

    Plato::setDataFromNonBlock(aOutMatrix, tOutRowMap, tOutColMap, tOutValues);
    tKernel.destroy_spgemm_handle();
}

/******************************************************************************//**
 * \brief matrix minus matrix
 * \param [in/out] aM1
 * \param [in] aM2
 *
 * aM1 = aM1 - aM2
 *
**********************************************************************************/
inline void
MatrixMinusMatrix(      Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
                  const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo,
                        Plato::OrdinalType aOffset)
{
    using Plato::OrdinalType;
    using Plato::Scalar;

    typedef Plato::ScalarVectorT<OrdinalType> OrdinalView;
    typedef Plato::ScalarVectorT<Scalar>  ScalarView;

    typedef KokkosKernels::Experimental::KokkosKernelsHandle
        <OrdinalType, OrdinalType, Scalar,
        typename Plato::ExecSpace, 
        typename Plato::MemSpace,
        typename Plato::MemSpace > KernelHandle;

    const auto& tMatOne = *aInMatrixOne;
    const auto& tMatTwo = *aInMatrixTwo;
    const OrdinalType tNumRowsOne = tMatOne.numRows();
    const OrdinalType tNumColsOne = tMatOne.numCols();
    const OrdinalType tNumRowsTwo = tMatTwo.numRows();
    const OrdinalType tNumColsTwo = tMatTwo.numCols();

    auto tNumRowsPerBlock = tMatOne.numRowsPerBlock();

    if (tNumColsOne != tNumColsTwo) { THROWERR("matrices have incompatible shape"); }

    ScalarView tMatOneValues;
    OrdinalView tMatOneRowMap, tMatOneColMap;
    Plato::getDataAsNonBlock(aInMatrixOne, tMatOneRowMap, tMatOneColMap, tMatOneValues);

    ScalarView tMatTwoValues;
    OrdinalView tMatTwoRowMap, tMatTwoColMap;
    Plato::getDataAsNonBlock(aInMatrixTwo, tMatTwoRowMap, tMatTwoColMap, tMatTwoValues, tNumRowsPerBlock, aOffset);

    OrdinalView tOutRowMap ("output row map", tNumRowsOne + 1);

    KernelHandle tKernel;
    tKernel.create_spadd_handle(/*sort rows=*/ false);
    auto tAddHandle = tKernel.get_spadd_handle();
    KokkosSparse::Experimental::spadd_symbolic< KernelHandle,
      OrdinalView, OrdinalView,
      OrdinalView, OrdinalView,
      OrdinalView, OrdinalView
    >
    ( &tKernel,
      tMatOneRowMap, tMatOneColMap,
      tMatTwoRowMap, tMatTwoColMap,
      tOutRowMap
    );

    auto t_nnz = tAddHandle->get_c_nnz();

    OrdinalView tOutColMap("output graph", t_nnz);
    ScalarView  tOutValues("output values", t_nnz);
    KokkosSparse::Experimental::spadd_numeric< KernelHandle,
      OrdinalView, OrdinalView, Scalar, ScalarView,
      OrdinalView, OrdinalView, Scalar, ScalarView,
      OrdinalView, OrdinalView, ScalarView
    >
    ( &tKernel,
      tMatOneRowMap, tMatOneColMap, tMatOneValues, 1.0,
      tMatTwoRowMap, tMatTwoColMap, tMatTwoValues, -1.0,
      tOutRowMap,    tOutColMap,    tOutValues
    );

    Plato::setDataFromNonBlock(aInMatrixOne, tOutRowMap, tOutColMap, tOutValues);
    tKernel.destroy_spadd_handle();
}

/******************************************************************************//**
 * \brief Condense into reduced matrix
 * \param [in/out] aA
 * \param [in] aB
 * \param [in] aC
 * \param [in] aD
 *
 * aA = aA - aB . RowSum(aC)^{-1} . aD
 *
**********************************************************************************/
inline void Condense(       Teuchos::RCP<Plato::CrsMatrixType> & aA,
                      const Teuchos::RCP<Plato::CrsMatrixType> & aB,
                      const Teuchos::RCP<Plato::CrsMatrixType> & aC,
                            Teuchos::RCP<Plato::CrsMatrixType> & aD,
                            Plato::OrdinalType aOffset )
{
  RowSummedInverseMultiply ( aC, aD );

  auto tNumRows = aB->numRows();
  auto tNumCols = aD->numCols();
  auto tNumRowsPerBlock = aB->numRowsPerBlock();
  auto tNumColsPerBlock = aD->numColsPerBlock();
  auto tBD = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock ) );
  MatrixMatrixMultiply     ( aB, aD, tBD );

  MatrixMinusMatrix        ( aA, tBD, aOffset );
}

} // namespace Plato

#endif /* PLATOMATHHELPERS_HPP_ */

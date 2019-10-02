#pragma once

namespace Plato
{

/******************************************************************************//**
 * @brief Functor for computing the row sum of a given matrix
 **********************************************************************************/
class RowSum
{
private:
    const typename Plato::CrsMatrixType::RowMapVector mRowMap;
    const typename Plato::CrsMatrixType::ScalarVector mEntries;
    const Plato::OrdinalType mNumDofsPerNode_I;
    const Plato::OrdinalType mNumDofsPerNode_J;

public:
    /**********************************************************************//**
     * @brief Constructor
     * @param [in] aMatrix Matrix for witch the row sum will be computed
     **************************************************************************/
    RowSum(Teuchos::RCP<Plato::CrsMatrixType> aMatrix) :
            mRowMap(aMatrix->rowMap()),
            mEntries(aMatrix->entries()),
            mNumDofsPerNode_I(aMatrix->numRowsPerBlock()),
            mNumDofsPerNode_J(aMatrix->numColsPerBlock())
    {
    }

    /**********************************************************************//**
     * @brief Functor
     * @param [in]  blockRowOrdinal Ordinal for the row for which the sum is to be computed
     * @param [out] aRowSum Row sum vector (assumed initialized to zero)
     **************************************************************************/
    DEVICE_TYPE
    inline void operator()(Plato::OrdinalType aBlockRowOrdinal, Plato::ScalarVector aRowSum) const
    {
        using Plato::OrdinalType;

        // for each entry in this block row
        OrdinalType tRowStart = mRowMap(aBlockRowOrdinal);
        OrdinalType tRowEnd = mRowMap(aBlockRowOrdinal + 1);

        OrdinalType tTotalBlockSize = mNumDofsPerNode_I * mNumDofsPerNode_J;

        for(OrdinalType tColNodeOrd = tRowStart; tColNodeOrd < tRowEnd; tColNodeOrd++)
        {

            OrdinalType tEntryOrdinalOffset = tTotalBlockSize * tColNodeOrd;

            // for each row in this block
            for(OrdinalType tIdim = 0; tIdim < mNumDofsPerNode_I; tIdim++)
            {
                // for each col in this block
                OrdinalType tVectorOrdinal = aBlockRowOrdinal * mNumDofsPerNode_I + tIdim;
                OrdinalType tMatrixOrdinal = tEntryOrdinalOffset + mNumDofsPerNode_J * tIdim;
                for(OrdinalType tJdim = 0; tJdim < mNumDofsPerNode_J; tJdim++)
                {
                    aRowSum(tVectorOrdinal) += mEntries(tMatrixOrdinal + tJdim);
                }
            }
        }
    }
};

/******************************************************************************//**
 * @brief Functor for computing the weighted inverse
 **********************************************************************************/
template<Plato::OrdinalType NumDofsPerNode_I, Plato::OrdinalType NumDofsPerNode_J = NumDofsPerNode_I>
class InverseWeight
{

public:
    /**********************************************************************//**
     * @brief Functor
     * @param [in]  blockRowOrdinal Ordinal for the row for which the sum is to be computed
     * @param [in]  Row sum vector, R
     * @param [in]  Input vector, b
     * @param [out] Output vector, x
     *
     *  x[i] = b[i] / R[i]
     *
     **************************************************************************/
    DEVICE_TYPE
    inline void operator()(Plato::OrdinalType aBlockRowOrdinal,
                           Plato::ScalarVector aRowSum,
                           Plato::ScalarVector aRHS,
                           Plato::ScalarVector aLHS,
                           Plato::Scalar aScale = 1.0) const
    {
        // for each row in this block
        for(Plato::OrdinalType tIdim = 0; tIdim < NumDofsPerNode_I; tIdim++)
        {
            // for each col in this block
            Plato::OrdinalType tVectorOrdinal = aBlockRowOrdinal * NumDofsPerNode_I + tIdim;
            aLHS(tVectorOrdinal) = (aRHS(tVectorOrdinal) / aRowSum(tVectorOrdinal)) * aScale;
        }
    }
};

/******************************************************************************//**
 * @brief Functor for computing the row sum of a given matrix
 **********************************************************************************/
class DiagonalInverseMultiply
{
private:
    const typename Plato::CrsMatrixType::RowMapVector mRowMap;
    const typename Plato::CrsMatrixType::ScalarVector mEntries;
    const Plato::OrdinalType mNumDofsPerNode_I;
    const Plato::OrdinalType mNumDofsPerNode_J;

public:
    /**********************************************************************//**
     * @brief Constructor
     * @param [in] aMatrix Matrix to witch the inverse diagonal multiply will be applied
     **************************************************************************/
    DiagonalInverseMultiply(Teuchos::RCP<Plato::CrsMatrixType> aMatrix) :
            mRowMap(aMatrix->rowMap()),
            mEntries(aMatrix->entries()),
            mNumDofsPerNode_I(aMatrix->numRowsPerBlock()),
            mNumDofsPerNode_J(aMatrix->numColsPerBlock())
    {
    }

    /**********************************************************************//**
     * @brief Functor
     * @param [in]  blockRowOrdinal Ordinal for the block row to which the inverse diagonal multiply is applied
     * @param [out] aDiagonals Vector of diagonal entries
     **************************************************************************/
    DEVICE_TYPE
    inline void operator()(Plato::OrdinalType aBlockRowOrdinal, Plato::ScalarVector aDiagonals) const
    {
        // for each entry in this block row
        Plato::OrdinalType tRowStart = mRowMap(aBlockRowOrdinal);
        Plato::OrdinalType tRowEnd = mRowMap(aBlockRowOrdinal + 1);

        Plato::OrdinalType tTotalBlockSize = mNumDofsPerNode_I * mNumDofsPerNode_J;

        for(Plato::OrdinalType tColNodeOrd = tRowStart; tColNodeOrd < tRowEnd; tColNodeOrd++)
        {

            Plato::OrdinalType tEntryOrdinalOffset = tTotalBlockSize * tColNodeOrd;

            // for each row in this block
            for(Plato::OrdinalType tIdim = 0; tIdim < mNumDofsPerNode_I; tIdim++)
            {
                // for each col in this block
                Plato::OrdinalType tVectorOrdinal = aBlockRowOrdinal * mNumDofsPerNode_I + tIdim;
                Plato::OrdinalType tMatrixOrdinal = tEntryOrdinalOffset + mNumDofsPerNode_J * tIdim;
                for(Plato::OrdinalType tJdim = 0; tJdim < mNumDofsPerNode_J; tJdim++)
                {
                    mEntries(tMatrixOrdinal + tJdim) /= aDiagonals(tVectorOrdinal);
                }
            }
        }
    }
};

}
// namespace Plato

/*
 * BLAS2.hpp
 *
 *  Created on: Feb 29, 2020
 */

#pragma once

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace blas2
{

/******************************************************************************//**
 * \tparam Length    number of elements along summation dimension
 * \tparam aAlpha    multiplication/scaling factor
 * \tparam aInputWS  2D scalar view
 * \tparam aBeta     multiplication/scaling factor
 * \tparam aOutputWS 2D scalar view
 *
 * \fn device_type inline void update
 *
 * \brief Add two-dimensional 2D views, b = \alpha a + \beta b
 *
 * \param [in/out] aVector 1D scalar view
**********************************************************************************/
template<Plato::OrdinalType Length,
         typename ScalarT,
         typename AViewTypeT,
         typename BViewTypeT>
DEVICE_TYPE inline void
update
(const Plato::OrdinalType & aCellOrdinal,
 const ScalarT & aAlpha,
 const Plato::ScalarMultiVectorT<AViewTypeT> & aInputWS,
 const ScalarT & aBeta,
 const Plato::ScalarMultiVectorT<BViewTypeT> & aOutputWS)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutputWS(aCellOrdinal, tIndex) =
            aBeta * aOutputWS(aCellOrdinal, tIndex) + aAlpha * aInputWS(aCellOrdinal, tIndex);
    }
}
// update function

/******************************************************************************//**
 * \tparam Length   number of elements along summation dimension
 * \tparam ScalarT  multiplication/scaling factor
 * \tparam ResultT  2D scalar view
 *
 * \fn device_type inline void update
 *
 * \brief Scale all the elements by input scalar value, \mathbf{a} = \alpha\mathbf{a}
 *
 * \param [in]  aCellOrdinal cell/element ordinal
 * \param [in]  aScalar      acalar multiplication factor
 * \param [out] aOutputWS    output 2D scalar view
**********************************************************************************/
template<Plato::OrdinalType Length,
         typename ScalarT,
         typename ResultT>
DEVICE_TYPE inline void
scale
(const Plato::OrdinalType & aCellOrdinal,
 const ScalarT & aScalar,
 const Plato::ScalarMultiVectorT<ResultT> & aOutputWS)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutputWS(aCellOrdinal, tIndex) *= aScalar;
    }
}
// function scale

/******************************************************************************//**
 * \tparam Length   number of elements along summation dimension
 * \tparam ScalarT  multiplication/scaling factor
 * \tparam ResultT  2D scalar view
 *
 * \fn device_type inline void update
 *
 * \brief Scale all the elements by input scalar value, \mathbf{b} = \alpha\mathbf{a}
 *
 * \param [in]  aCellOrdinal cell/element ordinal
 * \param [in]  aScalar      acalar multiplication factor
 * \param [in]  aInputWS     input 2D scalar view
 * \param [out] aOutputWS    output 2D scalar view
**********************************************************************************/
template<Plato::OrdinalType Length,
         typename ScalarT,
         typename AViewTypeT,
         typename BViewTypeT>
DEVICE_TYPE inline void
scale
(const Plato::OrdinalType & aCellOrdinal,
 const ScalarT & aScalar,
 const Plato::ScalarMultiVectorT<AViewTypeT> & aInputWS,
 const Plato::ScalarMultiVectorT<BViewTypeT> & aOutputWS)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutputWS(aCellOrdinal, tIndex) = aScalar * aInputWS(aCellOrdinal, tIndex);
    }
}
// function scale

/******************************************************************************//**
 * \tparam Length     number of elements in the summing direction
 * \tparam AViewTypeT view type
 * \tparam BViewTypeT view type
 * \tparam CViewTypeT view type
 *
 * \fn device_type inline void dot
 *
 * \brief Compute two-dimensional tensor dot product for each cell.
 *
 * \param [in]  aCellOrdinal cell/element ordinal
 * \param [in]  aViewA       input 2D scalar view
 * \param [in]  aViewB       input 2D scalar view
 * \param [out] aOutput      output 1D scalar view
**********************************************************************************/
template<Plato::OrdinalType Length,
         typename AViewType,
         typename BViewType,
         typename CViewType>
DEVICE_TYPE inline void dot
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::ScalarMultiVectorT<AViewType> & aViewA,
 const Plato::ScalarMultiVectorT<BViewType> & aViewB,
 const Plato::ScalarVectorT<CViewType>      & aOutput)
{
    for(Plato::OrdinalType tIndex = 0; tIndex < Length; tIndex++)
    {
        aOutput(aCellOrdinal) += aViewA(aCellOrdinal, tIndex) * aViewB(aCellOrdinal, tIndex);
    }
}
// function dot

/******************************************************************************//**
 * \brief Extract a sub 2D array from full 2D array
 *
 * \tparam NumStride stride, e.g. number of degree of freedom per node
 * \tparam NumOffset offset, e.g. degree of freedom offset
 *
 * \param [in] aFrom  input 2D array
 * \param [out] aTo   extracted 2D sub-array
 *
 * aToVector(i,j) = aFromVector(i,j*NumStride+NumOffset)
 *
**********************************************************************************/
template<Plato::OrdinalType NumStride, Plato::OrdinalType NumOffset>
inline void extract(const Plato::ScalarMultiVector& aFrom, Plato::ScalarMultiVector& aTo)
{
    auto tDim0 = aFrom.extent(0);
    for(Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        auto tToSubView = Kokkos::subview(aTo, tIndexI, Kokkos::ALL());
        auto tFromSubView = Kokkos::subview(aFrom, tIndexI, Kokkos::ALL());

        auto tLength = tToSubView.extent(0);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
        {
            tToSubView(aOrdinal) = tFromSubView(aOrdinal*NumStride + NumOffset);
        }, "blas2::extract");
    }
}
// function extract

/******************************************************************************//**
 * \brief Extract a sub 2D array from full 2D array
 *
 * \tparam  NumStride   stride, e.g. number of degree of freedom per node
 * \tparam  NumDim      number of dimensions to extract from an ordinal, e.g. number of displacement components at a vertex
 * \tparam  NumOffset   offset, e.g. degree of freedom offset (default = 0)
 *
 * \param  [in]  aNumOrdinal  number of ordinal, e.g. number of vertices in the mesh
 * \param  [in]  aFrom        input 2D array
 * \param  [out] aTo          extracted 2D sub-array
 *
 * aToVector(i,j*NumDim+Dim) = aFromVector(i,j*NumStride+Dim+NumOffset)
 *
**********************************************************************************/
template<Plato::OrdinalType NumStride, Plato::OrdinalType NumDim, Plato::OrdinalType NumOffset = 0>
inline void extract(const Plato::OrdinalType& aNumOrdinal, const Plato::ScalarMultiVector& aFrom, Plato::ScalarMultiVector& aTo)
{
    auto tDim0 = aFrom.extent(0);
    for(Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        auto tToSubView = Kokkos::subview(aTo, tIndexI, Kokkos::ALL());
        auto tFromSubView = Kokkos::subview(aFrom, tIndexI, Kokkos::ALL());
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumOrdinal), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
        {
            for(Plato::OrdinalType tDim = 0; tDim < NumDim; tDim++)
            {
                tToSubView(aOrdinal*NumDim + tDim) = tFromSubView(aOrdinal*NumStride+tDim+NumOffset);
            }

        }, "blas2::extract");
    }
}
// function extract

/******************************************************************************//**
 * \brief Fill 2-D array with a given input value, \f$ X(i,j) = \alpha\ \forall\ i,j \f$ indices.
 *
 * \tparam XViewType Input matrix, as a 2-D Kokkos::View
 *
 * \param [in]     aAlpha  scalar value
 * \param [in/out] aXvec   2-D Kokkos view
**********************************************************************************/
template<class XViewType>
inline void fill(typename XViewType::const_value_type& aAlpha, XViewType& aXvec)
{
    if(static_cast<Plato::OrdinalType>(aXvec.size()) <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nINPUT VECTOR IS EMPTY.\n");
    }

    const Plato::OrdinalType tNumEntriesDim0 = aXvec.extent(0);
    const Plato::OrdinalType tNumEntriesDim1 = aXvec.extent(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumEntriesDim0), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tIndex = 0; tIndex < tNumEntriesDim1; tIndex++)
        {
            aXvec(aCellOrdinal, tIndex) = aAlpha;
        }
    }, "blas2::fill");
}
// function fill


/******************************************************************************//**
 * \brief Scale 2-D array, \f$ X = \alpha*X \f$
 *
 * \tparam XViewType Input matrix, as a 2-D Kokkos::View
 *
 * \param [in]     aAlpha  scalar multiplier
 * \param [in/out] aXvec   2-D Kokkos view
**********************************************************************************/
template<class XViewType>
inline void scale(typename XViewType::const_value_type& aAlpha, XViewType& aXvec)
{
    if(static_cast<Plato::OrdinalType>(aXvec.size()) <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nINPUT VECTOR IS EMPTY.\n");
    }

    const Plato::OrdinalType tNumEntriesDim0 = aXvec.extent(0);
    const Plato::OrdinalType tNumEntriesDim1 = aXvec.extent(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumEntriesDim0), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tIndex = 0; tIndex < tNumEntriesDim1; tIndex++)
        {
            aXvec(aCellOrdinal, tIndex) = aAlpha * aXvec(aCellOrdinal, tIndex);
        }
    }, "blas2::scale");
}
// function scale

/******************************************************************************//**
 * \brief Add 2-D arrays, \f$ Y = \beta*Y + \alpha*X \f$
 *
 * \tparam XViewType Input matrix, as a 2-D Kokkos::View
 * \tparam YViewType Output matrix, as a 2-D Kokkos::View
 *
 * \param [in] aAlpha scalar multiplier
 * \param [in] aXvec 2-D vector workset (NumCells, NumEntriesPerCell)
 * \param [in] aBeta scalar multiplier
 * \param [in/out] aYvec 2-D vector workset (NumCells, NumEntriesPerCell)
**********************************************************************************/
template<class XViewType, class YViewType>
inline void update(typename XViewType::const_value_type& aAlpha,
                   const XViewType& aXvec,
                   typename YViewType::const_value_type& aBeta,
                   const YViewType& aYvec)
{
    if(aXvec.extent(0) != aYvec.extent(0))
    {
        std::stringstream tMsg;
        tMsg << "\nDIMENSION MISMATCH. X ARRAY DIM(0) = " << aXvec.extent(0)
                << " AND Y ARRAY DIM(0) = " << aYvec.extent(0) << ".\n";
        THROWERR(tMsg.str().c_str());
    }
    if(aXvec.extent(1) != aYvec.extent(1))
    {
        std::stringstream tMsg;
        tMsg << "\nDIMENSION MISMATCH. X ARRAY DIM(1) = " << aXvec.extent(1)
                << " AND Y ARRAY DIM(1) = " << aYvec.extent(1) << ".\n";
        THROWERR(tMsg.str().c_str());
    }

    const auto tNumEntriesDim0 = aXvec.extent(0);
    const auto tNumEntriesDim1 = aXvec.extent(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumEntriesDim0), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tIndex = 0; tIndex < tNumEntriesDim1; tIndex++)
        {
            aYvec(aCellOrdinal, tIndex) = aAlpha * aXvec(aCellOrdinal, tIndex) + aBeta * aYvec(aCellOrdinal, tIndex);
        }
    }, "blas2::update");
}
// function update

/******************************************************************************//**
 * \brief Add two 2-D vector workset
 *
 * \tparam NumDofsPerNode number of degrees of freedom per node
 * \tparam DofOffset      offset
 *
 * \param [in]     aAlpha 2-D scalar multiplier
 * \param [in]     aXvec  2-D vector workset (NumCells, NumEntriesPerCell)
 * \param [in/out] aYvec  2-D vector workset (NumCells, NumEntriesPerCell)
**********************************************************************************/
template<Plato::OrdinalType NumDofsPerNode, Plato::OrdinalType DofOffset>
inline void axpy(const Plato::Scalar & aAlpha, const Plato::ScalarMultiVector& aIn, Plato::ScalarMultiVector& aOut)
{
    if(aOut.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOUT ARRAY IS EMPTY.\n");
    }
    if(aIn.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nIN ARRAY IS EMPTY.\n");
    }
    if(aOut.extent(0) != aIn.extent(0))
    {
        std::stringstream tMsg;
        tMsg << "\nDIMENSION MISMATCH. X ARRAY DIM(0) = " << aOut.extent(0)
                << " AND Y ARRAY DIM(0) = " << aIn.extent(0) << ".\n";
        THROWERR(tMsg.str().c_str());
    }

    const auto tInputVecDim0 = aIn.extent(0);
    const auto tInputVecDim1 = aIn.extent(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tInputVecDim0), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tInputVecIndex = 0; tInputVecIndex < tInputVecDim1; tInputVecIndex++)
        {
            const auto tOutputVecIndex = (NumDofsPerNode * tInputVecIndex) + DofOffset;
            aOut(aCellOrdinal, tOutputVecIndex) = aAlpha * aOut(aCellOrdinal, tOutputVecIndex) + aIn(aCellOrdinal, tInputVecIndex);
        }
    }, "blas2::axpy");
}
// function axpy

/************************************************************************//**
 *
 * \brief Dense matrix-vector multiply: y = beta*y + alpha*A*x.
 *
 * \tparam AViewType Input matrix, as a 2-D Kokkos::View
 * \tparam XViewType Input vector, as a 1-D Kokkos::View
 * \tparam YViewType Output vector, as a nonconst 1-D Kokkos::View
 * \tparam AlphaCoeffType Type of input coefficient alpha
 * \tparam BetaCoeffType Type of input coefficient beta
 *
 * \param trans [in] "N" for non-transpose, "T" for transpose.  All
 *   characters after the first are ignored.  This works just like
 *   the BLAS routines.
 * \param aAlpha [in]     Input coefficient of A*x
 * \param aAmat  [in]     Input matrix, as a 2-D Kokkos::View
 * \param aXvec  [in]     Input vector, as a 1-D Kokkos::View
 * \param aBeta  [in]     Input coefficient of y
 * \param aYvec  [in/out] Output vector, as a nonconst 1-D Kokkos::View
 *
********************************************************************************/
template<class AViewType, class XViewType, class YViewType>
inline void matrix_times_vector(const char aTransA[],
                                typename AViewType::const_value_type& aAlpha,
                                const AViewType& aAmat,
                                const XViewType& aXvec,
                                typename YViewType::const_value_type& aBeta,
                                YViewType& aYvec)
{
    // check validity of inputs' dimensions
    if(aAmat.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput matrix A is empty, i.e. size <= 0\n")
    }
    if(aXvec.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nInput vector X is empty, i.e. size <= 0\n")
    }
    if(aYvec.size() <= static_cast<Plato::OrdinalType>(0))
    {
        THROWERR("\nOutput vector Y is empty, i.e. size <= 0\n")
    }
    if(aAmat.extent(0) != aXvec.extent(0))
    {
        THROWERR("\nDimension mismatch, matrix A and vector X have different number of cells.\n")
    }
    if(aAmat.extent(0) != aYvec.extent(0))
    {
        THROWERR("\nDimension mismatch, matrix A and vector Y have different number of cells.\n")
    }

    // Check validity of transpose argument
    bool tValidTransA = (aTransA[0] == 'N') || (aTransA[0] == 'n') ||
                        (aTransA[0] == 'T') || (aTransA[0] == 't');

    if(!tValidTransA)
    {
        std::stringstream tMsg;
        tMsg << "\ntransA[0] = '" << aTransA[0] << "'. Valid values include 'N' or 'n' (No transpose) and 'T' or 't' (Transpose).\n";
        THROWERR(tMsg.str())
    }

    auto tNumCells = aAmat.extent(0);
    auto tNumRows = aAmat.extent(1);
    auto tNumCols = aAmat.extent(2);
    if((aTransA[0] == 'N') || (aTransA[0] == 'n'))
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
            {
                aYvec(aCellOrdinal, tRowIndex) = aBeta * aYvec(aCellOrdinal, tRowIndex);
            }

            for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
            {
                for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
                {
                    aYvec(aCellOrdinal, tRowIndex) = aYvec(aCellOrdinal, tRowIndex) +
                            aAlpha * aAmat(aCellOrdinal, tRowIndex, tColIndex) * aXvec(aCellOrdinal, tColIndex);
                }
            }
        }, "matrix vector multiplication - no transpose");
    }
    else
    {
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
            {
                aYvec(aCellOrdinal, tColIndex) = aBeta * aYvec(aCellOrdinal, tColIndex);
            }

            for(Plato::OrdinalType tRowIndex = 0; tRowIndex < tNumRows; tRowIndex++)
            {
                for(Plato::OrdinalType tColIndex = 0; tColIndex < tNumCols; tColIndex++)
                {
                    aYvec(aCellOrdinal, tColIndex) = aYvec(aCellOrdinal, tColIndex) +
                            aAlpha * aAmat(aCellOrdinal, tRowIndex, tColIndex) * aXvec(aCellOrdinal, tRowIndex);
                }
            }
        }, "matrix vector multiplication - transpose");
    }
}
// function matrix_times_vector

}
// namespace blas2

}
// namespace Plato

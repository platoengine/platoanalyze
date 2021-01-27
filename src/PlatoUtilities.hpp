/*
 * PlatoUtilities.hpp
 *
 *  Created on: Aug 8, 2018
 */

#ifndef SRC_PLATO_PLATOUTILITIES_HPP_
#define SRC_PLATO_PLATOUTILITIES_HPP_

#include <Omega_h_array.hpp>

#include "PlatoStaticsTypes.hpp"
#include "Plato_Solve.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Print 1D standard vector to terminal - host function
 * \param [in] aInput 1D standard vector
 * \param [in] aName  container name (default = "")
**********************************************************************************/
inline void print_standard_vector_1D
(const std::vector<Plato::Scalar> & aInput, std::string aName = "Data")
{
    std::cout << "PRINT " << aName << std::endl;
    int tSize = aInput.size();
    for(int tIndex = 0; tIndex < tSize; tIndex++)
    {
        auto tEntry = tIndex + 1;
        std::cout << "X(" << tEntry << ") = " << aInput[tIndex] << std::endl;
    }
}
// print_array_1D_device

/******************************************************************************//**
 * \brief Print input 1D container to terminal - device function
 * \param [in] aInput 1D container
 * \param [in] aName  container name (default = "")
**********************************************************************************/
template<typename ArrayT>
DEVICE_TYPE inline void print_array_1D_device
(const ArrayT & aInput, const char* aName)
{
    auto tSize = aInput.size();
    for(Plato::OrdinalType tIndex = 0; tIndex < tSize; tIndex++)
    {
        auto tEntry = tIndex + static_cast<Plato::OrdinalType>(1);
        std::cout << aName << ": X(" << tEntry << ") = " << aInput(tIndex) << std::endl;
    }
}
// print_array_1D_device

/******************************************************************************//**
 * \brief Print input 2D container to terminal - device function
 * \param [in] aLeadOrdinal leading ordinal
 * \param [in] aInput       2D container
 * \param [in] aName        container name (default = "")
**********************************************************************************/
template<typename ArrayT>
DEVICE_TYPE inline void print_array_2D_device
(const Plato::OrdinalType & aLeadOrdinal, const ArrayT & aInput, const char* aName)
{
    auto tSize = aInput.dimension_1();
    for(Plato::OrdinalType tIndex = 0; tIndex < tSize; tIndex++)
    {
        auto tEntry = tIndex + static_cast<Plato::OrdinalType>(1);
        std::cout << aName << ": X(" << aLeadOrdinal << "," << tEntry  << ") = " << aInput(aLeadOrdinal,tIndex) << std::endl;
    }
}
// print_array_2D_device

/******************************************************************************//**
 * \brief Print input 3D container to terminal - device function
 * \param [in] aLeadOrdinal leading ordinal
 * \param [in] aInput       3D container
 * \param [in] aName        container name (default = "")
**********************************************************************************/
template<typename ArrayT>
DEVICE_TYPE inline void print_array_3D_device
(const Plato::OrdinalType & aLeadOrdinal, const ArrayT & aInput, const char* aName)
{
    auto tDimOneLength = aInput.dimension_1();
    auto tDimTwoLength = aInput.dimension_2();
    for(Plato::OrdinalType tIndexI = 0; tIndexI < tDimOneLength; tIndexI++)
    {
        for(Plato::OrdinalType tIndexJ = 0; tIndexJ < tDimTwoLength; tIndexJ++)
        {
            auto tEntryI = tIndexI + static_cast<Plato::OrdinalType>(1);
            auto tEntryJ = tIndexJ + static_cast<Plato::OrdinalType>(1);
            std::cout << aName << ": X(" << aLeadOrdinal << "," << tEntryI << "," << tEntryJ << ") = "
                                    << aInput(aLeadOrdinal, tIndexI, tIndexJ) << std::endl;
        }
    }
}
// print_array_3D_device

/******************************************************************************//**
 * \brief Print input 1D container of ordinals to terminal/console - host function
 * \param [in] aInput 1D container of ordinals
 * \param [in] aName  container name (default = "")
**********************************************************************************/
inline void print_array_ordinals_1D(const Plato::LocalOrdinalVector & aInput, std::string aName = "")
{
    std::cout << "PRINT " << aName << std::endl;

    Plato::OrdinalType tSize = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tSize), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        printf("X[%lld] = %lld\n", aIndex + static_cast<Plato::OrdinalType>(1), aInput(aIndex));
#else
        printf("X[%d] = %d\n", aIndex + static_cast<Plato::OrdinalType>(1), aInput(aIndex));
#endif
    }, "print array ordinals 1D");
    std::cout << std::endl;
}
// function print


/******************************************************************************//**
 * \brief Print input sparse matrix to file for debugging
 * \param [in] aInMatrix Pointer to Crs Matrix
 * \param [in] aFilename  file name (default = "matrix.txt")
**********************************************************************************/
inline void print_sparse_matrix_to_file( Teuchos::RCP<Plato::CrsMatrixType> aInMatrix, std::string aFilename = "matrix.txt")
{
    FILE * tOutputFile;
    tOutputFile = fopen(aFilename.c_str(), "w");
    auto tNumRowsPerBlock = aInMatrix->numRowsPerBlock();
    auto tNumColsPerBlock = aInMatrix->numColsPerBlock();
    auto tBlockSize = tNumRowsPerBlock*tNumColsPerBlock;

    auto tRowMap = Kokkos::create_mirror(aInMatrix->rowMap());
    Kokkos::deep_copy(tRowMap, aInMatrix->rowMap());

    auto tColMap = Kokkos::create_mirror(aInMatrix->columnIndices());
    Kokkos::deep_copy(tColMap, aInMatrix->columnIndices());

    auto tValues = Kokkos::create_mirror(aInMatrix->entries());
    Kokkos::deep_copy(tValues, aInMatrix->entries());

    auto tNumRows = tRowMap.extent(0)-1;
    for(Plato::OrdinalType iRowIndex=0; iRowIndex<tNumRows; iRowIndex++)
    {
        auto tFrom = tRowMap(iRowIndex);
        auto tTo   = tRowMap(iRowIndex+1);
        for(auto iColMapEntryIndex=tFrom; iColMapEntryIndex<tTo; iColMapEntryIndex++)
        {
            auto tBlockColIndex = tColMap(iColMapEntryIndex);
            for(Plato::OrdinalType iLocalRowIndex=0; iLocalRowIndex<tNumRowsPerBlock; iLocalRowIndex++)
            {
                auto tRowIndex = iRowIndex * tNumRowsPerBlock + iLocalRowIndex;
                for(Plato::OrdinalType iLocalColIndex=0; iLocalColIndex<tNumColsPerBlock; iLocalColIndex++)
                {
                    auto tColIndex = tBlockColIndex * tNumColsPerBlock + iLocalColIndex;
                    auto tSparseIndex = iColMapEntryIndex * tBlockSize + iLocalRowIndex * tNumColsPerBlock + iLocalColIndex;
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
                    fprintf(tOutputFile, "%lld %lld %16.8e\n", tRowIndex, tColIndex, tValues[tSparseIndex]);
#else
                    fprintf(tOutputFile, "%d %d %16.8e\n", tRowIndex, tColIndex, tValues[tSparseIndex]);
#endif
                }
            }
        }
    }
    fclose(tOutputFile);
}

/******************************************************************************//**
 * \brief Print input 1D container to terminal - host function
 * \param [in] aInput 1D container
 * \param [in] aName  container name (default = "")
**********************************************************************************/
template<typename ArrayT>
inline void print(const ArrayT & aInput, std::string aName = "")
{
    std::cout << "PRINT " << aName << std::endl;

    Plato::OrdinalType tSize = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tSize), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        printf("X[%lld] = %e\n", aIndex + static_cast<Plato::OrdinalType>(1), aInput(aIndex));
#else
        printf("X[%d] = %e\n", aIndex + static_cast<Plato::OrdinalType>(1), aInput(aIndex));
#endif
    }, "print 1D array");
    std::cout << std::endl;
}
// function print

/******************************************************************************//**
 * \brief Print input 3D container to terminal
 * \tparam array type
 * \param [in] aInput 3D container
 * \param [in] aName  container name (default = "")
**********************************************************************************/
template<typename ArrayT>
inline void print_array_2D(const ArrayT & aInput, const std::string & aName)
{
    std::cout << "PRINT " << aName << std::endl;

    const Plato::OrdinalType tNumRows = aInput.extent(0);
    const Plato::OrdinalType tNumCols = aInput.extent(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aRow)
    {
        for(Plato::OrdinalType tCol = 0; tCol < tNumCols; tCol++)
        {
        
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
            printf("X(%lld,%lld) = %e\n", aRow + static_cast<Plato::OrdinalType>(1), tCol + static_cast<Plato::OrdinalType>(1), aInput(aRow, tCol));
#else
            printf("X(%d,%d) = %e\n", aRow + static_cast<Plato::OrdinalType>(1), tCol + static_cast<Plato::OrdinalType>(1), aInput(aRow, tCol));
#endif
        }
    }, "print 2D array");
    std::cout << std::endl;
}
// function print_array_2D

template<class ArrayT>
inline void print_array_2D_Fad(Plato::OrdinalType aNumCells, 
                               Plato::OrdinalType aNumDofsPerCell, 
                               const ArrayT & aInput, 
                               std::string aName = "")
{
    std::cout << "PRINT " << aName << std::endl;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCell)
    {
        for(Plato::OrdinalType tDof = 0; tDof < aNumDofsPerCell; tDof++)
        {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
            printf("X(%lld,%lld) = %e\n", aCell + static_cast<Plato::OrdinalType>(1), tDof + static_cast<Plato::OrdinalType>(1), aInput(aCell).dx(tDof));
#else
            printf("X(%d,%d) = %e\n", aCell + static_cast<Plato::OrdinalType>(1), tDof + static_cast<Plato::OrdinalType>(1), aInput(aCell).dx(tDof));
#endif
        }
    }, "print 2D array Fad");
    std::cout << std::endl;
}

/******************************************************************************//**
 * \brief Print input 3D container to terminal
 * \tparam array type
 * \param [in] aInput 3D container
 * \param [in] aName  container name (default = "")
**********************************************************************************/
template<typename ArrayT>
inline void print_array_3D(const ArrayT & aInput, const std::string & aName)
{
    std::cout << "PRINT " << aName << std::endl;

    const Plato::OrdinalType tNumRows = aInput.extent(1);
    const Plato::OrdinalType tNumCols = aInput.extent(2);
    const Plato::OrdinalType tNumMatrices = aInput.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumMatrices), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        for(Plato::OrdinalType tRow = 0; tRow < tNumRows; tRow++)
        {
            for(Plato::OrdinalType tCol = 0; tCol < tNumCols; tCol++)
            {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
                printf("X(%lld,%lld,%lld) = %e\n", aIndex + static_cast<Plato::OrdinalType>(1), tRow + static_cast<Plato::OrdinalType>(1), 
                                               tCol + static_cast<Plato::OrdinalType>(1), aInput(aIndex,tRow, tCol));
#else
                printf("X(%d,%d,%d) = %e\n", aIndex + static_cast<Plato::OrdinalType>(1), tRow + static_cast<Plato::OrdinalType>(1), 
                                               tCol + static_cast<Plato::OrdinalType>(1), aInput(aIndex,tRow, tCol));
#endif
            }
        }
    }, "print 3D array");
    std::cout << std::endl;
}
// function print

/******************************************************************************//**
 * \brief Copy 1D view into Omega_h 1D array
 * \param [in] aStride stride
 * \param [in] aNumVertices number of mesh vertices
 * \param [in] aInput 1D view
 * \param [out] aOutput 1D Omega_h array
**********************************************************************************/
template<const Plato::OrdinalType NumDofsPerNodeInInputArray, const Plato::OrdinalType NumDofsPerNodeInOutputArray>
inline void copy(const Plato::OrdinalType & aStride,
                 const Plato::OrdinalType & aNumVertices,
                 const Plato::ScalarVector & aInput,
                 Omega_h::Write<Omega_h::Real> & aOutput)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumVertices), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        for(Plato::OrdinalType tIndex = 0; tIndex < NumDofsPerNodeInOutputArray; tIndex++)
        {
            Plato::OrdinalType tOutputDofIndex = (aIndex * NumDofsPerNodeInOutputArray) + tIndex;
            Plato::OrdinalType tInputDofIndex = (aIndex * NumDofsPerNodeInInputArray) + (aStride + tIndex);
            aOutput[tOutputDofIndex] = aInput(tInputDofIndex);
        }
    },"PlatoDriver::copy");
}
// function copy

/******************************************************************************//**
 * \brief Copy 2D view into Omega_h 1D array
 * \param [in] aInput 2D view
 * \param [out] aOutput 1D Omega_h array
**********************************************************************************/
inline void copy_2Dview_to_write(const Plato::ScalarMultiVector & aInput, Omega_h::Write<Omega_h::Real> & aOutput)
{
    auto tNumMajorEntries      = aInput.extent(0);
    auto tNumDofsPerMajorEntry = aInput.extent(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumMajorEntries), LAMBDA_EXPRESSION(const Plato::OrdinalType & tMajorIndex)
    {
        for(Plato::OrdinalType tMinorIndex = 0; tMinorIndex < tNumDofsPerMajorEntry; tMinorIndex++)
        {
            Plato::OrdinalType tOutputDofIndex = (tMajorIndex * tNumDofsPerMajorEntry) + tMinorIndex;
            aOutput[tOutputDofIndex] = aInput(tMajorIndex, tMinorIndex);
        }
    },"PlatoDriver::compress_copy_2Dview_to_write");
}

/******************************************************************************//**
 * \brief Copy 1D view into Omega_h 1D array
 * \param [in] aInput 2D view
 * \param [out] aOutput 1D Omega_h array
**********************************************************************************/
inline void copy_1Dview_to_write(const Plato::ScalarVector & aInput, Omega_h::Write<Omega_h::Real> & aOutput)
{
    auto tNumEntries      = aInput.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumEntries), LAMBDA_EXPRESSION(const Plato::OrdinalType & tIndex)
    {
        aOutput[tIndex] = aInput(tIndex);
    },"PlatoDriver::compress_copy_1Dview_to_write");
}

} // namespace Plato

#endif /* SRC_PLATO_PLATOUTILITIES_HPP_ */

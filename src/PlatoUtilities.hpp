/*
 * PlatoUtilities.hpp
 *
 *  Created on: Aug 8, 2018
 */

#ifndef SRC_PLATO_PLATOUTILITIES_HPP_
#define SRC_PLATO_PLATOUTILITIES_HPP_

#include <Omega_h_array.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \fn tolower
 * \brief Convert uppercase word to lowercase.
 * \param [in] aInput word
 * \return lowercase word
**********************************************************************************/
inline std::string tolower(const std::string& aInput)
{
    std::locale tLocale;
    std::ostringstream tOutput;
    for (auto& tChar : aInput)
    {
        tOutput << std::tolower(tChar,tLocale);
    }
    return (tOutput.str());
}
// function tolower

/******************************************************************************//**
 * \brief Print 1D standard vector to terminal - host function
 * \param [in] aInput 1D standard vector
 * \param [in] aName  container name (default = "")
**********************************************************************************/
inline void print_standard_vector_1D
(const std::vector<Plato::Scalar> & aInput, std::string aName = "Data")
{
    printf("BEGIN PRINT: %s\n", aName);
    Plato::OrdinalType tSize = aInput.size();
    for(decltype(tSize) tIndex = 0; tIndex < tSize; tIndex++)
    {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        printf("X(%lld)=%f\n", tIndex, aInput(tIndex));
#else
        printf("X(%d)=%f\n", tIndex, aInput[tIndex]);
#endif
    }
    printf("END PRINT: %s\n", aName);
}
// print_standard_vector_1D

/******************************************************************************//**
 * \brief Print input 1D container to terminal - device function
 * \param [in] aInput 1D container
 * \param [in] aName  container name (default = "")
**********************************************************************************/
template<typename ArrayT>
DEVICE_TYPE inline void print_array_1D_device
(const ArrayT & aInput, const char* aName)
{
    printf("BEGIN PRINT: %s\n", aName);
    Plato::OrdinalType tSize = aInput.size();
    for(decltype(tSize) tIndex = 0; tIndex < tSize; tIndex++)
    {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        printf("X(%lld)=%f\n", tIndex, aInput(tIndex));
#else
        printf("X(%d)=%f\n", tIndex, aInput(tIndex));
#endif
    }
    printf("END PRINT: %s\n", aName);
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
    Plato::OrdinalType tSize = aInput.extent(1);
    printf("BEGIN PRINT: %s\n", aName);
    for(decltype(tSize) tIndex = 0; tIndex < tSize; tIndex++)
    {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        printf("X(%lld,%lld)=%f\n", aLeadOrdinal, tIndex, aInput(aLeadOrdinal,tIndex));
#else
        printf("X(%d,%d)=%f\n", aLeadOrdinal, tIndex, aInput(aLeadOrdinal,tIndex));
#endif
    }
    printf("END PRINT: %s\n", aName);
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
    Plato::OrdinalType tDimOneLength = aInput.extent(1);
    Plato::OrdinalType tDimTwoLength = aInput.extent(2);
    printf("BEGIN PRINT: %s\n", aName);
    for (decltype(tDimOneLength) tIndexI = 0; tIndexI < tDimOneLength; tIndexI++)
    {
        for (decltype(tDimTwoLength) tIndexJ = 0; tIndexJ < tDimTwoLength; tIndexJ++)
        {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
            printf("X(%lld,%lld,%lld)=%f\n", aLeadOrdinal, tIndexI, tIndexJ, aInput(aLeadOrdinal, tIndexI, tIndexJ));
#else
            printf("X(%d,%d,%d)=%f\n", aLeadOrdinal, tIndexI, tIndexJ, aInput(aLeadOrdinal, tIndexI, tIndexJ));
#endif
        }
    }
    printf("END PRINT: %s\n", aName);
}
// print_array_3D_device

/******************************************************************************//**
 * \brief Print input 1D container of ordinals to terminal/console - host function
 * \param [in] aInput 1D container of ordinals
 * \param [in] aName  container name (default = "")
**********************************************************************************/
inline void print_array_ordinals_1D(const Plato::LocalOrdinalVector & aInput, std::string aName = "")
{
    printf("\nBEGIN PRINT: %s\n", aName);
    Plato::OrdinalType tSize = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tSize), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        printf("X[%lld] = %lld\n", aIndex, aInput(aIndex));
#else
        printf("X[%d] = %d\n", aIndex, aInput(aIndex));
#endif
    }, "print array ordinals 1D");
    printf("END PRINT: %s\n", aName);
}
// function print

/******************************************************************************//**
 * \brief Print input 1D container to terminal - host function
 * \param [in] aInput 1D container
 * \param [in] aName  container name (default = "")
**********************************************************************************/
template<typename ArrayT>
inline void print(const ArrayT & aInput, std::string aName = "")
{
    printf("\nBEGIN PRINT: %s\n", aName);
    Plato::OrdinalType tSize = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tSize), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
        printf("X[%lld] = %e\n", aIndex, aInput(aIndex));
#else
        printf("X[%d] = %e\n", aIndex, aInput(aIndex));
#endif
    }, "print 1D array");
    printf("END PRINT: %s\n", aName);
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
    printf("\nBEGIN PRINT: %s\n", aName);
    const Plato::OrdinalType tNumRows = aInput.extent(0);
    const Plato::OrdinalType tNumCols = aInput.extent(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & aRow)
    {
        for(Plato::OrdinalType tCol = 0; tCol < tNumCols; tCol++)
        {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
            printf("X(%lld,%lld) = %e\n", aRow, tCol, aInput(aRow, tCol));
#else
            printf("X(%d,%d) = %e\n", aRow, tCol, aInput(aRow, tCol));
#endif
        }
    }, "print 2D array");
    printf("END PRINT: %s\n", aName);
}
// function print_array_2D

template<class ArrayT>
inline void print_array_2D_Fad(Plato::OrdinalType aNumCells, 
                               Plato::OrdinalType aNumDofsPerCell, 
                               const ArrayT & aInput, 
                               std::string aName = "")
{
    printf("\nBEGIN PRINT: %s\n", aName);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCell)
    {
        for(Plato::OrdinalType tDof = 0; tDof < aNumDofsPerCell; tDof++)
        {
#ifdef PLATOANALYZE_LONG_LONG_ORDINALTYPE
            printf("X(%lld,%lld) = %e\n", aCell, tDof, aInput(aCell).dx(tDof));
#else
            printf("X(%d,%d) = %e\n", aCell, tDof, aInput(aCell).dx(tDof));
#endif
        }
    }, "print 2D array Fad");
    printf("END PRINT: %s\n", aName);
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
    printf("\nBEGIN PRINT: %s\n", aName);
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
                printf("X(%lld,%lld,%lld) = %e\n", aIndex, tRow, tCol, aInput(aIndex,tRow, tCol));
#else
                printf("X(%d,%d,%d) = %e\n", aIndex, tRow, tCol, aInput(aIndex,tRow, tCol));
#endif
            }
        }
    }, "print 3D array");
    printf("END PRINT: %s\n", aName);
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

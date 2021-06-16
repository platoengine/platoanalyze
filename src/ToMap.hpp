#ifndef PLATO_TO_MAP_HPP
#define PLATO_TO_MAP_HPP

#include <string>

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Null operation for all types (only specializations below are non-trivial)
 * \param [in/out] aDataMap output data storage
 * \param [in] aInput container
 * \param [in] aEntryName output data name
 **********************************************************************************/
template<typename InputType>
inline void
toMap(
          Plato::DataMap & aDataMap,
    const InputType      & aInput, 
    const std::string    & aEntryName
)
{
    // don't add to map
}
// function toMap

/******************************************************************************//**
 * \brief Null operation for all types (only specializations below are non-trivial)
 * \param [in/out] aDataMap output data storage
 * \param [in] aInput container
 * \param [in] aEntryName output data name
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 **********************************************************************************/
template<typename InputType>
inline void
toMap(
          Plato::DataMap       & aDataMap,
    const InputType            & aInput,
    const std::string          & aEntryName,
    const Plato::SpatialDomain & aSpatialDomain
)
{
    // don't add to map
}
// function toMap

/******************************************************************************//**
 * \brief Store 1D container in data map. The data map is used for output purposes
 * \param [in/out] aDataMap output data storage
 * \param [in] aInput 1D container
 * \param [in] aEntryName output data name
 **********************************************************************************/
template<>
inline void
toMap(
          Plato::DataMap      & aDataMap,
    const Plato::ScalarVector & aInput,
    const std::string         & aEntryName
)
{
    aDataMap.scalarVectors[aEntryName] = aInput;
}
// function toMap

/******************************************************************************//**
 * \brief Store 1D container in data map. The data map is used for output purposes
 * \param [in/out] aDataMap output data storage
 * \param [in] aInput 1D container
 * \param [in] aEntryName output data name
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 **********************************************************************************/
template<>
inline void
toMap(
          Plato::DataMap       & aDataMap,
    const Plato::ScalarVector  & aInput,
    const std::string          & aEntryName,
    const Plato::SpatialDomain & aSpatialDomain
)
{
    if( aDataMap.scalarVectors.count(aEntryName) == 0 )
    {
        Plato::ScalarVector tNewEntry(aEntryName, aSpatialDomain.Mesh.nelems());
        aDataMap.scalarVectors[aEntryName] = tNewEntry;
    }

    auto tData = aDataMap.scalarVectors.at(aEntryName);

    if( tData.extent(0) != aSpatialDomain.Mesh.nelems() )
    {
        THROWERR( "DataMap error: attempted to insert domain data into an incompatible view");
    }

    auto tNumCells = aSpatialDomain.numCells();
    auto tOrdinals = aSpatialDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tGlobalOrdinal = tOrdinals[aCellOrdinal];
        tData(tGlobalOrdinal) = aInput(aCellOrdinal);
    }, "Add domain entries");
}
// function toMap

/******************************************************************************//**
 * \brief Store 1D container in data map. The data map is used for output purposes
 * \param [in/out] aDataMap output data storage
 * \param [in] aInput 1D container
 * \param [in] aEntryName output data name
 * \param [in] aSpatialDomain Plato Analyze spatial domain
 **********************************************************************************/
inline void
toMap(
          std::map<std::string, Plato::ScalarVector> & aTo,
    const Plato::LocalOrdinalVector                  & aFrom,
          std::string                                  aName
)
{
    auto tNumEntries = aFrom.extent(0);
    Plato::ScalarVector tTo(aName, tNumEntries);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumEntries), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tTo(aOrdinal) = aFrom(aOrdinal);
    }, "cast");

    aTo[aName] = tTo;
}
// function toMap

/******************************************************************************//**
 * \brief Store 2D container in data map. The data map is used for output purposes
 * \param [in/out] aDataMap output data storage
 * \param [in] aInput 2D container
 * \param [in] aEntryName output data name
 **********************************************************************************/
template<>
inline void
toMap(
          Plato::DataMap           & aDataMap,
    const Plato::ScalarMultiVector & aInput,
    const std::string              & aEntryName
)
{
    aDataMap.scalarMultiVectors[aEntryName] = aInput;
}
// function toMap

/******************************************************************************//**
 * \brief Store 2D container in data map. The data map is used for output purposes
 * \param [in/out] aDataMap output data storage
 * \param [in] aInput 2D container
 * \param [in] aEntryName output data name
 **********************************************************************************/
template<>
inline void
toMap(
          Plato::DataMap           & aDataMap,
    const Plato::ScalarMultiVector & aInput,
    const std::string              & aEntryName,
    const Plato::SpatialDomain     & aSpatialDomain
)
{
    auto tDim = aInput.extent(1);
    if( aDataMap.scalarMultiVectors.count(aEntryName) == 0 )
    {
        Plato::ScalarMultiVector tNewEntry(aEntryName, aSpatialDomain.Mesh.nelems(), tDim);
        aDataMap.scalarMultiVectors[aEntryName] = tNewEntry;
    }

    auto tData = aDataMap.scalarMultiVectors.at(aEntryName);

    if( tData.extent(0) != aSpatialDomain.Mesh.nelems() )
    {
        THROWERR( "DataMap error: attempted to insert domain data into an incompatible view");
    }

    if( tData.extent(1) != tDim )
    {
        THROWERR( "DataMap error: attempted to insert domain data into an incompatible view");
    }

    auto tNumCells = aSpatialDomain.numCells();
    auto tOrdinals = aSpatialDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tGlobalOrdinal = tOrdinals[aCellOrdinal];
        for(decltype(tDim) iDim=0; iDim<tDim; iDim++)
        {
            tData(tGlobalOrdinal, iDim) = aInput(aCellOrdinal, iDim);
        }
    }, "Add domain entries");
}
// function toMap

/******************************************************************************//**
 * \brief Retrieve 2D container from data map.
 * \param [in/out] aDataMap output data storage
 * \param [in] aOutput 2D container
 * \param [in] aEntryName output data name
 **********************************************************************************/
inline void
fromMap(
          Plato::DataMap           & aDataMap,
    const Plato::ScalarMultiVector & aOutput,
    const std::string              & aEntryName,
    const Plato::SpatialDomain     & aSpatialDomain
)
{
    auto tDim = aOutput.extent(1);

    auto tNumCells = aSpatialDomain.numCells();
    if( aOutput.extent(0) != tNumCells )
    {
        THROWERR( "DataMap error: attempted to extract domain data into an incompatible view");
    }

    auto tData = aDataMap.scalarMultiVectors.at(aEntryName);

    auto tOrdinals = aSpatialDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tGlobalOrdinal = tOrdinals[aCellOrdinal];
        for(decltype(tDim) iDim=0; iDim<tDim; iDim++)
        {
            aOutput(aCellOrdinal, iDim) = tData(tGlobalOrdinal, iDim);
        }
    }, "Add domain entries");
}
// function fromMap

/******************************************************************************//**
 * \brief Retrieve 1D container from data map.
 * \param [in/out] aDataMap output data storage
 * \param [in] aOutput 1D container
 * \param [in] aEntryName output data name
 **********************************************************************************/
inline void
fromMap(
          Plato::DataMap       & aDataMap,
    const Plato::ScalarVector  & aOutput,
    const std::string          & aEntryName,
    const Plato::SpatialDomain & aSpatialDomain
)
{
    auto tNumCells = aSpatialDomain.numCells();
    if( aOutput.extent(0) != tNumCells )
    {
        THROWERR( "DataMap error: attempted to extract domain data into an incompatible view");
    }

    auto tData = aDataMap.scalarVectors.at(aEntryName);

    auto tOrdinals = aSpatialDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tGlobalOrdinal = tOrdinals[aCellOrdinal];
        aOutput(aCellOrdinal) = tData(tGlobalOrdinal);
    }, "Add domain entries");
}
// function fromMap


/******************************************************************************//**
 * \brief Store 3D container in data map. The data map is used for output purposes
 * \param [in/out] aDataMap output data storage
 * \param [in] aInput 3D container
 * \param [in] aEntryName output data name
 **********************************************************************************/
template<>
inline void
toMap(
          Plato::DataMap       & aDataMap,
    const Plato::ScalarArray3D & aInput,
    const std::string          & aEntryName
)
{
    aDataMap.scalarArray3Ds[aEntryName] = aInput;
}
// function toMap

/******************************************************************************//**
 * \brief Store 3D container in data map. The data map is used for output purposes
 * \param [in/out] aDataMap output data storage
 * \param [in] aInput 3D container
 * \param [in] aEntryName output data name
 **********************************************************************************/
template<>
inline void
toMap(
          Plato::DataMap       & aDataMap,
    const Plato::ScalarArray3D & aInput,
    const std::string          & aEntryName,
    const Plato::SpatialDomain & aSpatialDomain
)
{
    auto tDim1 = aInput.extent(1);
    auto tDim2 = aInput.extent(2);
    if( aDataMap.scalarArray3Ds.count(aEntryName) == 0 )
    {
        Plato::ScalarArray3D tNewEntry(aEntryName, aSpatialDomain.Mesh.nelems(), tDim1, tDim2);
        aDataMap.scalarArray3Ds[aEntryName] = tNewEntry;
    }

    auto tData = aDataMap.scalarArray3Ds.at(aEntryName);

    if( tData.extent(0) != aSpatialDomain.Mesh.nelems() )
    {
        THROWERR( "DataMap error: attempted to insert domain data into an incompatible view");
    }

    if( tData.extent(1) != tDim1 )
    {
        THROWERR( "DataMap error: attempted to insert domain data into an incompatible view");
    }

    if( tData.extent(2) != tDim2 )
    {
        THROWERR( "DataMap error: attempted to insert domain data into an incompatible view");
    }

    auto tNumCells = aSpatialDomain.numCells();
    auto tOrdinals = aSpatialDomain.cellOrdinals();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tGlobalOrdinal = tOrdinals[aCellOrdinal];
        for(decltype(tDim1) iDim1=0; iDim1<tDim1; iDim1++)
        {
            for(decltype(tDim2) iDim2=0; iDim2<tDim2; iDim2++)
            {
                tData(tGlobalOrdinal, iDim1, iDim2) = aInput(aCellOrdinal, iDim1, iDim2);
            }
        }
    }, "Add domain entries");
}
// function toMap

}// end namespace Plato

#endif

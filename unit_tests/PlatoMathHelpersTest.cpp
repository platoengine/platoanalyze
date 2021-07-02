/*
 * PlatoMathHelpersTest.cpp
 *
 *  Created on: July 11, 2018
 */

#include <vector>

//#define COMPUTE_GOLD_
#ifdef COMPUTE_GOLD_
  #include <iostream>
  #include <fstream>
#endif

#include <assert.h>

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "BLAS1.hpp"
#include "Solutions.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoMathFunctors.hpp"
#include "Mechanics.hpp"
#include "StabilizedMechanics.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/VectorFunction.hpp"
#include "VectorFunctionVMS.hpp"
#include "ApplyProjection.hpp"
#include "AnalyzeMacros.hpp"
#include "HyperbolicTangentProjection.hpp"
#include "alg/CrsMatrix.hpp"

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

#include <Omega_h_mesh.hpp>

namespace PlatoUnitTests
{


using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

namespace PlatoDevel {


/******************************************************************************/
/*! 
  \brief Set Kokkos::View data from std::vector
*/
/******************************************************************************/
template <typename DataType>
void setViewFromVector( Plato::ScalarVectorT<DataType> aView, std::vector<DataType> aVector)
{
  Kokkos::View<DataType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(aVector.data(),aVector.size());
  Kokkos::deep_copy(aView, tHostView);
}

/******************************************************************************/
/*! 
  \brief Set matrix data from provided views
*/
/******************************************************************************/
void setMatrixData(
  Teuchos::RCP<Plato::CrsMatrixType> aMatrix,
  std::vector<Plato::OrdinalType>    aRowMap,
  std::vector<Plato::OrdinalType>    aColMap,
  std::vector<Plato::Scalar>         aValues )
{
  Plato::ScalarVectorT<Plato::OrdinalType> tRowMap("row map", aRowMap.size());
  setViewFromVector(tRowMap, aRowMap);
  aMatrix->setRowMap(tRowMap);

  Plato::ScalarVectorT<Plato::OrdinalType> tColMap("col map", aColMap.size());
  setViewFromVector(tColMap, aColMap);
  aMatrix->setColumnIndices(tColMap);

  Plato::ScalarVectorT<Plato::Scalar> tValues("values", aValues.size());
  setViewFromVector(tValues, aValues);
  aMatrix->setEntries(tValues);
}

void sortColumnEntries(
      const Plato::ScalarVectorT<Plato::OrdinalType> & aMatrixRowMap,
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

void
fromFull( Teuchos::RCP<Plato::CrsMatrixType>            aOutMatrix,
          const std::vector<std::vector<Plato::Scalar>> aInMatrix )
{
    using Plato::OrdinalType;
    using Plato::Scalar;

    if( aOutMatrix->numRows() != aInMatrix.size()    ) { THROWERR("matrices have incompatible shapes"); }
    if( aOutMatrix->numCols() != aInMatrix[0].size() ) { THROWERR("matrices have incompatible shapes"); }

    auto tNumRowsPerBlock = aOutMatrix->numRowsPerBlock();
    auto tNumColsPerBlock = aOutMatrix->numColsPerBlock();
    auto tNumBlockRows = aOutMatrix->numRows() / tNumRowsPerBlock;
    auto tNumBlockCols = aOutMatrix->numCols() / tNumColsPerBlock;

    std::vector<OrdinalType> tBlockRowMap(tNumBlockRows+1);;

    tBlockRowMap[0] = 0;
    std::vector<OrdinalType> tColumnIndices;
    std::vector<Scalar> tBlockEntries;
    for( OrdinalType iBlockRowIndex=0; iBlockRowIndex<tNumBlockRows; iBlockRowIndex++)
    {
        for( OrdinalType iBlockColIndex=0; iBlockColIndex<tNumBlockCols; iBlockColIndex++)
        {
             bool blockIsNonZero = false;
             std::vector<Scalar> tLocalEntries;
             for( OrdinalType iLocalBlockRowIndex=0; iLocalBlockRowIndex<tNumRowsPerBlock; iLocalBlockRowIndex++)
             {
                 for( OrdinalType iLocalBlockColIndex=0; iLocalBlockColIndex<tNumColsPerBlock; iLocalBlockColIndex++)
                 {
                      auto tMatrixRow = iBlockRowIndex * tNumRowsPerBlock + iLocalBlockRowIndex;
                      auto tMatrixCol = iBlockColIndex * tNumColsPerBlock + iLocalBlockColIndex;
                      tLocalEntries.push_back( aInMatrix[tMatrixRow][tMatrixCol] );
                      if( aInMatrix[tMatrixRow][tMatrixCol] != 0.0 ) blockIsNonZero = true;
                 }
             }
             if( blockIsNonZero )
             {
                 tColumnIndices.push_back( iBlockColIndex );
                 tBlockEntries.insert(tBlockEntries.end(), tLocalEntries.begin(), tLocalEntries.end());
             }
        }
        tBlockRowMap[iBlockRowIndex+1] = tColumnIndices.size();
    }

    setMatrixData(aOutMatrix, tBlockRowMap, tColumnIndices, tBlockEntries);
}

void RowSum(
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrix,
            Plato::ScalarVector & aOutRowSum)
{
    auto tNumBlockRows = aInMatrix->rowMap().size()-1;
    auto tNumRows = aInMatrix->numRows();
    Plato::RowSum tRowSumFunctor(aInMatrix);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumBlockRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tBlockRowOrdinal) {
      tRowSumFunctor(tBlockRowOrdinal, aOutRowSum);
    });
}

void InverseMultiply(
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrix,
            Plato::ScalarVector & aInDiagonal)
{
    auto tNumBlockRows = aInMatrix->rowMap().size()-1;
    Plato::DiagonalInverseMultiply tDiagInverseMultiplyFunctor(aInMatrix);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumBlockRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tBlockRowOrdinal) {
      tDiagInverseMultiplyFunctor(tBlockRowOrdinal, aInDiagonal);
    });
}

void SlowDumbRowSummedInverseMultiply(
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
            Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo)
{
    auto F1 = ::PlatoUtestHelpers::toFull(aInMatrixOne);
    auto F2 = ::PlatoUtestHelpers::toFull(aInMatrixTwo);

    auto tNumM1Rows = aInMatrixOne->numRows();
    auto tNumM1Cols = aInMatrixOne->numCols();
    if (tNumM1Rows != tNumM1Cols) { THROWERR("matrix one must be square"); }

    auto tNumM2Rows = aInMatrixTwo->numRows();
    auto tNumM2Cols = aInMatrixTwo->numCols();
    if (tNumM1Cols != tNumM2Rows) { THROWERR("matrices have incompatible shapes"); }

    for (auto iRow=0; iRow<tNumM1Rows; iRow++)
    {
        Plato::Scalar tRowSum = 0.0;
        for (auto iCol=0; iCol<tNumM1Cols; iCol++)
        {
            tRowSum += F1[iRow][iCol];
        }
        for (auto iCol=0; iCol<tNumM2Cols; iCol++)
        {
            F2[iRow][iCol] /= tRowSum;
        }
    }
    fromFull(aInMatrixTwo, F2);
}


void MatrixMinusEqualsMatrix(
            Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo)
{
    auto tEntriesOne = aInMatrixOne->entries();
    auto tEntriesTwo = aInMatrixTwo->entries();
    auto tNumEntries = tEntriesOne.extent(0);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumEntries), LAMBDA_EXPRESSION(const Plato::OrdinalType & tEntryOrdinal) {
      tEntriesOne(tEntryOrdinal) -= tEntriesTwo(tEntryOrdinal);
    });
}

void MatrixMinusEqualsMatrix(
            Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo, Plato::OrdinalType aOffset)
{
    auto tRowMap = aInMatrixOne->rowMap();
    auto tNumBlockRows = tRowMap.size()-1;

    auto tFromNumRowsPerBlock = aInMatrixOne->numRowsPerBlock();
    auto tFromNumColsPerBlock = aInMatrixOne->numColsPerBlock();
    auto tToNumRowsPerBlock   = aInMatrixTwo->numRowsPerBlock();
    auto tToNumColsPerBlock   = aInMatrixTwo->numColsPerBlock();

    assert(tToNumColsPerBlock == tFromNumColsPerBlock);

    auto tEntriesOne = aInMatrixOne->entries();
    auto tEntriesTwo = aInMatrixTwo->entries();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumBlockRows), LAMBDA_EXPRESSION(const Plato::OrdinalType & tBlockRowOrdinal) {
      
        auto tFrom = tRowMap(tBlockRowOrdinal);
        auto tTo   = tRowMap(tBlockRowOrdinal+1);
        for( auto tColMapIndex=tFrom; tColMapIndex<tTo; ++tColMapIndex )
        {
            auto tFromEntryOffset = tColMapIndex * tFromNumRowsPerBlock * tFromNumColsPerBlock;
            auto tToEntryOffset   = tColMapIndex * tToNumRowsPerBlock   * tToNumColsPerBlock + aOffset * tToNumColsPerBlock;;
            for( Plato::OrdinalType tBlockColOrdinal=0; tBlockColOrdinal<tToNumColsPerBlock; ++tBlockColOrdinal )
            {
                auto tToEntryIndex   = tToEntryOffset   + tBlockColOrdinal;
                auto tFromEntryIndex = tFromEntryOffset + tBlockColOrdinal;
                tEntriesOne[tToEntryIndex] -= tEntriesTwo[tFromEntryIndex];
            }
        }
    });
}

void SlowDumbMatrixMinusMatrix( 
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo,
            int aOffset=-1)
{
    auto tNumM1Rows = aInMatrixOne->numRows();
    auto tNumM1Cols = aInMatrixOne->numCols();
    auto tNumM2Rows = aInMatrixTwo->numRows();
    auto tNumM2Cols = aInMatrixTwo->numCols();
    auto tNumM1RowsPerBlock = aInMatrixOne->numRowsPerBlock();
    auto tNumM2RowsPerBlock = aInMatrixTwo->numRowsPerBlock();
   
    if( aOffset == -1 )
    {
        if( tNumM1Rows != tNumM2Rows || tNumM1Cols != tNumM2Cols ) { THROWERR("input matrices have incompatible shapes"); }
    }
    else
    {
        if(  tNumM2RowsPerBlock != 1 || tNumM1Cols != tNumM2Cols ) { THROWERR("input matrices have incompatible shapes"); }
    }

    using Plato::Scalar;
    std::vector<std::vector<Scalar>> tFullMatrix(tNumM1Rows,std::vector<Scalar>(tNumM1Cols, 0.0));

    auto F1 = ::PlatoUtestHelpers::toFull(aInMatrixOne);
    auto F2 = ::PlatoUtestHelpers::toFull(aInMatrixTwo);

    if( aOffset == -1 )
    {
        for (auto iRow=0; iRow<tNumM1Rows; iRow++)
        {
            for (auto iCol=0; iCol<tNumM1Cols; iCol++)
            {
                tFullMatrix[iRow][iCol] = F1[iRow][iCol] - F2[iRow][iCol];
            }
        }
    }
    else
    {
        for (auto iRow=0; iRow<tNumM1Rows; iRow++)
        {
            for (auto iCol=0; iCol<tNumM1Cols; iCol++)
            {
                tFullMatrix[iRow][iCol] = F1[iRow][iCol];
            }
        }
        for (auto iRow=0; iRow<tNumM2Rows; iRow++)
        {
            for (auto iCol=0; iCol<tNumM2Cols; iCol++)
            {
                tFullMatrix[tNumM1RowsPerBlock*iRow+aOffset][iCol] -= F2[iRow][iCol];
            }
        }
    }
 
    fromFull(aInMatrixOne, tFullMatrix);
}


void SlowDumbMatrixMatrixMultiply( 
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixOne,
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixTwo,
            Teuchos::RCP<Plato::CrsMatrixType> & aOutMatrix)
{
    auto tNumOutMatrixRows = aInMatrixOne->numRows();
    auto tNumOutMatrixCols = aInMatrixTwo->numCols();

    if( aInMatrixOne->numCols() != aInMatrixTwo->numRows() ) { THROWERR("input matrices have incompatible shapes"); }

    auto tNumInner = aInMatrixOne->numCols();

    using Plato::Scalar;
    std::vector<std::vector<Scalar>> tFullMatrix(tNumOutMatrixRows,std::vector<Scalar>(tNumOutMatrixCols, 0.0));

    auto F1 = ::PlatoUtestHelpers::toFull(aInMatrixOne);
    auto F2 = ::PlatoUtestHelpers::toFull(aInMatrixTwo);

    for (auto iRow=0; iRow<tNumOutMatrixRows; iRow++)
    {
        for (auto iCol=0; iCol<tNumOutMatrixCols; iCol++)
        {
            tFullMatrix[iRow][iCol] = 0.0;
            for (auto iK=0; iK<tNumInner; iK++)
            {
                tFullMatrix[iRow][iCol] += F1[iRow][iK]*F2[iK][iCol];
            }
        }
    }
 
    fromFull(aOutMatrix, tFullMatrix);
}


} // end namespace PlatoDevel

Teuchos::RCP<Teuchos::ParameterList> gElastostaticsParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>         \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>        \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>      \n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Elliptic'>                                           \n"
    "    <ParameterList name='Penalty Function'>                                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );


template <typename DataType>
bool is_same(
      const Plato::ScalarVectorT<DataType> & aView,
      const std::vector<DataType>          & aVec)
 {
    auto tView_host = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView_host, aView);
    for (unsigned int i = 0; i < aVec.size(); ++i)
    {
        if(tView_host(i) != aVec[i])
        {
            return false;
        }
    }
    return true;
 }

template <typename DataType>
bool is_same(
      const Plato::ScalarVectorT<DataType> & aViewA,
      const Plato::ScalarVectorT<DataType> & aViewB)
 {
    if( aViewA.extent(0) != aViewB.extent(0) ) return false;

    auto tViewA_host = Kokkos::create_mirror(aViewA);
    Kokkos::deep_copy(tViewA_host, aViewA);
    auto tViewB_host = Kokkos::create_mirror(aViewB);
    Kokkos::deep_copy(tViewB_host, aViewB);
    for (unsigned int i = 0; i < aViewA.extent(0); ++i)
    {
        if(tViewA_host(i) != tViewB_host(i)) return false;
    }
    return true;
 }

bool is_sequential(
      const Plato::ScalarVectorT<Plato::OrdinalType> & aRowMap,
      const Plato::ScalarVectorT<Plato::OrdinalType> & aColMap)
 {
    auto tRowMap = PlatoUtestHelpers::get(aRowMap);
    auto tColMap = PlatoUtestHelpers::get(aColMap);

    auto tNumRows = tRowMap.extent(0)-1;

    for (unsigned int i = 0; i < tNumRows; ++i)
    {
        auto tFrom = tRowMap(i);
        auto tTo = tRowMap(i+1);
        for (auto iColMapEntry=tFrom; iColMapEntry<tTo-1; iColMapEntry++)
        {
            if ( tColMap(iColMapEntry) >= tColMap(iColMapEntry+1) )
            {
                return false;
            }
        }
    }
    return true;
 }
bool is_equivalent(
      const Plato::ScalarVectorT<Plato::OrdinalType> & aRowMap,
      const Plato::ScalarVectorT<Plato::OrdinalType> & aColMapA,
      const Plato::ScalarVectorT<Plato::Scalar>      & aValuesA,
      const Plato::ScalarVectorT<Plato::OrdinalType> & aColMapB,
      const Plato::ScalarVectorT<Plato::Scalar>      & aValuesB,
      Plato::Scalar tolerance = 1.0e-14)
 {
    if( aColMapA.extent(0) != aColMapB.extent(0) ) return false;
    if( aValuesA.extent(0) != aValuesB.extent(0) ) return false;

    auto tRowMap  = PlatoUtestHelpers::get(aRowMap);
    auto tColMapA = PlatoUtestHelpers::get(aColMapA);
    auto tValuesA = PlatoUtestHelpers::get(aValuesA);
    auto tColMapB = PlatoUtestHelpers::get(aColMapB);
    auto tValuesB = PlatoUtestHelpers::get(aValuesB);

    Plato::OrdinalType tANumEntriesPerBlock = aValuesA.extent(0) / aColMapA.extent(0);
    Plato::OrdinalType tBNumEntriesPerBlock = aValuesB.extent(0) / aColMapB.extent(0);
    if( tANumEntriesPerBlock != tBNumEntriesPerBlock ) return false;

    auto tNumRows = tRowMap.extent(0)-1;
    for (unsigned int i = 0; i < tNumRows; ++i)
    {
        auto tFrom = tRowMap(i);
        auto tTo = tRowMap(i+1);
        for (auto iColMapEntryA=tFrom; iColMapEntryA<tTo; iColMapEntryA++)
        {
            auto tColumnIndexA = tColMapA(iColMapEntryA);
            for (auto iColMapEntryB=tFrom; iColMapEntryB<tTo; iColMapEntryB++)
            {
                if (tColumnIndexA == tColMapB(iColMapEntryB) )
                {
                    for (auto iBlockEntry=0; iBlockEntry<tANumEntriesPerBlock; iBlockEntry++)
                    {
                        auto tBlockEntryIndexA = iColMapEntryA*tANumEntriesPerBlock+iBlockEntry;
                        auto tBlockEntryIndexB = iColMapEntryB*tBNumEntriesPerBlock+iBlockEntry;
                        Plato::Scalar tSum = fabs(tValuesA(tBlockEntryIndexA)) + fabs(tValuesB(tBlockEntryIndexB));
                        Plato::Scalar tDif = fabs(tValuesA(tBlockEntryIndexA) - tValuesB(tBlockEntryIndexB));
                        Plato::Scalar tRelVal = (tSum != 0.0) ? 2.0*tDif/tSum : 0.0;
                        if (tRelVal > tolerance)
                        {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
 }

bool is_zero(
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrix)
 {
    auto tEntries = aInMatrix->entries();
    auto tEntries_host = Kokkos::create_mirror(tEntries);
    Kokkos::deep_copy(tEntries_host, tEntries);
    for (unsigned int i = 0; i < tEntries_host.extent(0); ++i)
    {
        if(tEntries_host(i) != 0.0) return false;
    }
    return true;
 }

bool is_same(
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixA,
      const Teuchos::RCP<Plato::CrsMatrixType> & aInMatrixB)
 {
    if( !is_same(aInMatrixA->rowMap(), aInMatrixB->rowMap()) )
    {
        return false;
    }
    if( !is_same(aInMatrixA->columnIndices(), aInMatrixB->columnIndices()) )
    {
        return false;
    }
    if( !is_same(aInMatrixA->entries(), aInMatrixB->entries()) )
    {
        return false;
    }
    return true;
 }

Teuchos::RCP<Plato::CrsMatrixType> createSquareMatrix()
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);
  auto nverts = tMesh->nverts();

  // create vector data
  //
  Plato::ScalarVector u("state", spaceDim*nverts);
  Plato::ScalarVector z("control", nverts);
  Plato::blas1::fill(1.0, z);

  // create residual function
  //
  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *gElastostaticsParams);

  Plato::Elliptic::VectorFunction<::Plato::Mechanics<spaceDim>>
    tVectorFunction(tSpatialModel, tDataMap, *gElastostaticsParams, gElastostaticsParams->get<std::string>("PDE Constraint"));

  // compute and test gradient_u
  //
  return tVectorFunction.gradient_u(u,z);
}

template <typename PhysicsT>
Teuchos::RCP<Plato::VectorFunctionVMS<PhysicsT>>
createStabilizedResidual(const Plato::SpatialModel & aSpatialModel)
{
  Plato::DataMap tDataMap;
  return Teuchos::rcp( new Plato::VectorFunctionVMS<PhysicsT>
         (aSpatialModel, tDataMap, *gElastostaticsParams, gElastostaticsParams->get<std::string>("PDE Constraint")));
}

template <typename PhysicsT>
Teuchos::RCP<Plato::VectorFunctionVMS<typename PhysicsT::ProjectorT>>
createStabilizedProjector(const Plato::SpatialModel & aSpatialModel)
{
  Plato::DataMap tDataMap;
  return Teuchos::rcp( new Plato::VectorFunctionVMS<typename PhysicsT::ProjectorT>
         (aSpatialModel, tDataMap, *gElastostaticsParams, std::string("State Gradient Projection")));
}

/******************************************************************************/
/*! 
  \brief Transform a block matrix to a non-block matrix and back then verify
 that the starting and final matrices are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_FromToBlockMatrix)
{
  auto tMatrixA = createSquareMatrix();

  auto tNumRows = tMatrixA->numRows();
  auto tNumCols = tMatrixA->numCols();
  auto tNumRowsPerBlock = tMatrixA->numRowsPerBlock();
  auto tNumColsPerBlock = tMatrixA->numColsPerBlock();
  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );

  Plato::ScalarVectorT<Plato::Scalar> tMatrixEntries;
  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixRowMap, tMatrixColMap;
  Plato::getDataAsNonBlock  (tMatrixA, tMatrixRowMap, tMatrixColMap, tMatrixEntries);
  Plato::setDataFromNonBlock(tMatrixB, tMatrixRowMap, tMatrixColMap, tMatrixEntries);

  TEST_ASSERT(is_same(tMatrixA->rowMap(), tMatrixB->rowMap()));

  TEST_ASSERT(is_equivalent(tMatrixA->rowMap(),
                            tMatrixA->columnIndices(), tMatrixA->entries(),
                            tMatrixB->columnIndices(), tMatrixB->entries()));
}

/******************************************************************************/
/*! 
  \brief Transform a rectangular block matrix to a non-block matrix and back then verify
 that the starting and final matrices are the same.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_FromToBlockMatrix_Rect)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 9, 4, 3) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapA = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  PlatoDevel::setMatrixData(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tNumRows = tMatrixA->numRows();
  auto tNumCols = tMatrixA->numCols();
  auto tNumRowsPerBlock = tMatrixA->numRowsPerBlock();
  auto tNumColsPerBlock = tMatrixA->numColsPerBlock();
  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );

  Plato::ScalarVectorT<Plato::Scalar> tMatrixEntries;
  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixRowMap, tMatrixColMap;
  Plato::getDataAsNonBlock  (tMatrixA, tMatrixRowMap, tMatrixColMap, tMatrixEntries);
  Plato::setDataFromNonBlock(tMatrixB, tMatrixRowMap, tMatrixColMap, tMatrixEntries);

  TEST_ASSERT(is_same(tMatrixA->rowMap(), tMatrixB->rowMap()));

  TEST_ASSERT(is_equivalent(tMatrixA->rowMap(),
                            tMatrixA->columnIndices(), tMatrixA->entries(),
                            tMatrixB->columnIndices(), tMatrixB->entries()));
}


/******************************************************************************/
/*! 
  \brief Make sure is_same(A, A) == true, and is_same(A, B) == false when A != B
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_is_same)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 6, 4, 2) );
  std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
  PlatoDevel::setMatrixData(tMatrixA, tRowMap, tColMap, tValuesA);

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(6, 12, 2, 4) );
  std::vector<Plato::Scalar>      tValuesB = 
    { 1, 5, 2, 6, 3, 7, 4, 8, 1, 5, 2, 6, 3, 7, 4, 8,
      1, 5, 2, 6, 3, 7, 4, 8, 1, 5, 2, 6, 3, 7, 4, 8 };
  PlatoDevel::setMatrixData(tMatrixB, tRowMap, tColMap, tValuesB);

  TEST_ASSERT(is_same(tMatrixA, tMatrixA));
  TEST_ASSERT(!is_same(tMatrixA, tMatrixB));
}

/******************************************************************************/
/*! 
  \brief Create rectangular matrix, A and B=Tranpose(A), then compute A.B and 
         compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMatrixMultiply_Rect)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 6, 4, 2) );
  std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8,
      1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8 };
  PlatoDevel::setMatrixData(tMatrixA, tRowMap, tColMap, tValuesA);

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(6, 12, 2, 4) );
  std::vector<Plato::Scalar>      tValuesB = 
    { 1, 5, 2, 6, 3, 7, 4, 8, 1, 5, 2, 6, 3, 7, 4, 8,
      1, 5, 2, 6, 3, 7, 4, 8, 1, 5, 2, 6, 3, 7, 4, 8 };
  PlatoDevel::setMatrixData(tMatrixB, tRowMap, tColMap, tValuesB);

  auto tMatrixAB         = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  auto tSlowDumbMatrixAB = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );

  Plato::MatrixMatrixMultiply              ( tMatrixA, tMatrixB, tMatrixAB);
  PlatoDevel::SlowDumbMatrixMatrixMultiply ( tMatrixA, tMatrixB, tSlowDumbMatrixAB);

  TEST_ASSERT(is_same(tMatrixAB, tSlowDumbMatrixAB));
}

/******************************************************************************/
/*! 
  \brief Create a square matrix, A, then compute A.A and compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMatrixMultiply_1)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapA = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  PlatoDevel::setMatrixData(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tMatrixAA         = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  auto tSlowDumbMatrixAA = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );

  Plato::MatrixMatrixMultiply             ( tMatrixA, tMatrixA, tMatrixAA);
  PlatoDevel::SlowDumbMatrixMatrixMultiply( tMatrixA, tMatrixA, tSlowDumbMatrixAA);

  std::vector<Plato::Scalar> tGoldMatrixEntries = {
    90.0, 100.0, 110.0, 120.0, 202.0, 228.0, 254.0, 280.0, 314.0, 356.0, 398.0, 440.0, 426.0, 484.0,  542.0,  600.0,
   180.0, 200.0, 220.0, 240.0, 404.0, 456.0, 508.0, 560.0, 628.0, 712.0, 796.0, 880.0, 852.0, 968.0, 1084.0, 1200.0,
    90.0, 100.0, 110.0, 120.0, 202.0, 228.0, 254.0, 280.0, 314.0, 356.0, 398.0, 440.0, 426.0, 484.0,  542.0,  600.0,
    90.0, 100.0, 110.0, 120.0, 202.0, 228.0, 254.0, 280.0, 314.0, 356.0, 398.0, 440.0, 426.0, 484.0,  542.0,  600.0,
    90.0, 100.0, 110.0, 120.0, 202.0, 228.0, 254.0, 280.0, 314.0, 356.0, 398.0, 440.0, 426.0, 484.0,  542.0,  600.0
  };
  TEST_ASSERT(is_same(tMatrixAA->entries(), tGoldMatrixEntries));

  std::vector<Plato::OrdinalType> tGoldMatrixRowMap = { 0, 2, 4, 5 };
  TEST_ASSERT(is_same(tMatrixAA->rowMap(), tGoldMatrixRowMap));

  std::vector<Plato::OrdinalType> tGoldMatrixColMap = { 0, 2, 0, 2, 2 };
  TEST_ASSERT(is_same(tMatrixAA->columnIndices(), tGoldMatrixColMap));

  TEST_ASSERT(is_same(tMatrixAA, tSlowDumbMatrixAA));
}

/******************************************************************************/
/*! 
  \brief Create a square matrix, A, then sort the column entries
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_SortColumnEntries)
{
  auto tMatrix = createSquareMatrix();

  Plato::ScalarVectorT<Plato::Scalar> tMatrixEntries;
  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixRowMap, tMatrixColMap;
  Plato::getDataAsNonBlock (tMatrix, tMatrixRowMap, tMatrixColMap, tMatrixEntries);

  Plato::ScalarVectorT<Plato::Scalar> tMatrixEntriesSorted("values", tMatrixEntries.extent(0));
  Kokkos::deep_copy(tMatrixEntriesSorted, tMatrixEntries);

  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixColMapSorted("col map", tMatrixColMap.extent(0));
  Kokkos::deep_copy(tMatrixColMapSorted, tMatrixColMap);

  PlatoDevel::sortColumnEntries(tMatrixRowMap, tMatrixColMapSorted, tMatrixEntriesSorted);

  TEST_ASSERT(is_sequential(tMatrixRowMap, tMatrixColMapSorted) );
  TEST_ASSERT(is_equivalent(tMatrixRowMap,
                            tMatrixColMap, tMatrixEntries,
                            tMatrixColMapSorted, tMatrixEntriesSorted) );
}

/******************************************************************************/
/*! 
  \brief Create a square matrix, A, then convert to and from full (non-sparse)
         matrix and verify that A doesn't change.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_ToFromFull)
{
  auto tMatrix = createSquareMatrix();
  auto tFullMatrix = ::PlatoUtestHelpers::toFull(tMatrix);
  auto tSparseMatrix = Teuchos::rcp( new Plato::CrsMatrixType( tMatrix->numRows(), tMatrix->numCols(), 
                                     tMatrix->numRowsPerBlock(), tMatrix->numRowsPerBlock()));
  PlatoDevel::fromFull(tSparseMatrix, tFullMatrix);
  
  TEST_ASSERT(is_equivalent(tMatrix->rowMap(),
                            tMatrix->columnIndices(), tMatrix->entries(),
                            tSparseMatrix->columnIndices(), tSparseMatrix->entries()));
}

/******************************************************************************/
/*! 
  \brief Create a square matrix, A, then compute A.A and compare.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMatrixMultiply_2)
{
  auto tMatrixA = createSquareMatrix();
  auto tMatrixB = createSquareMatrix();

  auto tNumRows = tMatrixA->numRows();
  auto tNumCols = tMatrixB->numCols();
  auto tNumRowsPerBlock = tMatrixA->numRowsPerBlock();
  auto tNumColsPerBlock = tMatrixB->numColsPerBlock();
  auto tMatrixAB         = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );
  auto tSlowDumbMatrixAB = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );

  Plato::MatrixMatrixMultiply              ( tMatrixA, tMatrixB, tMatrixAB);
  PlatoDevel::SlowDumbMatrixMatrixMultiply ( tMatrixA, tMatrixB, tSlowDumbMatrixAB);

  TEST_ASSERT(is_same(tMatrixAB->rowMap(), tSlowDumbMatrixAB->rowMap()));
  TEST_ASSERT(is_equivalent(tMatrixAB->rowMap(),
                            tMatrixAB->columnIndices(), tMatrixAB->entries(),
                            tSlowDumbMatrixAB->columnIndices(), tSlowDumbMatrixAB->entries()));
}

/******************************************************************************/
/*! 
  \brief Create a square matrix, A, then verify that A - A = 0.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMinusEqualsMatrix)
{
  auto tMatrixA = createSquareMatrix();
  PlatoDevel::MatrixMinusEqualsMatrix( tMatrixA, tMatrixA );
  TEST_ASSERT(is_zero(tMatrixA));
}


/******************************************************************************/
/*! 
  \brief Create a sub-block matrix, B, and convert to full non-block.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_GetDataAsNonBlock)
{
  Plato::OrdinalType tTargetBlockColSize = 4;
  Plato::OrdinalType tTargetBlockColOffset = 3;
  auto tMatrix = Teuchos::rcp( new Plato::CrsMatrixType(  3, 12, 1, 4) );
  std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValues = 
    { 13, 14, 15, 16,
      13, 14, 15, 16,
      13, 14, 15, 16,
      13, 14, 15, 16 };
  PlatoDevel::setMatrixData(tMatrix, tRowMap, tColMap, tValues);

  Plato::ScalarVectorT<Plato::Scalar> tMatrixEntries;
  Plato::ScalarVectorT<Plato::OrdinalType> tMatrixRowMap, tMatrixColMap;
  Plato::getDataAsNonBlock (tMatrix, tMatrixRowMap, tMatrixColMap, tMatrixEntries,
                                 tTargetBlockColSize, tTargetBlockColOffset);

  std::vector<Plato::OrdinalType> tMatrixRowMap_Gold = {
    0, 8, 16, 24, 32, 36, 40, 44, 48, 52, 56, 60, 64
  };
  TEST_ASSERT(is_same(tMatrixRowMap, tMatrixRowMap_Gold));

  std::vector<Plato::OrdinalType> tMatrixColMap_Gold = {
    0, 1, 2,  3,  8, 9, 10, 11,
    0, 1, 2,  3,  8, 9, 10, 11,
    0, 1, 2,  3,  8, 9, 10, 11,
    0, 1, 2,  3,  8, 9, 10, 11,
    0, 1, 2,  3,  0, 1, 2,  3,
    0, 1, 2,  3,  0, 1, 2,  3,
    8, 9, 10, 11, 8, 9, 10, 11,
    8, 9, 10, 11, 8, 9, 10, 11
  };
  TEST_ASSERT(is_same(tMatrixColMap, tMatrixColMap_Gold));

  std::vector<Plato::Scalar> tMatrixEntries_Gold = {
     0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,
     0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,
     0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,
    13.0, 14.0, 15.0, 16.0, 13.0, 14.0, 15.0, 16.0,
     0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,
     0.0,  0.0,  0.0,  0.0, 13.0, 14.0, 15.0, 16.0,
     0.0,  0.0,  0.0,  0.0, 0.0,  0.0,  0.0,  0.0,
     0.0,  0.0,  0.0,  0.0, 13.0, 14.0, 15.0, 16.0
  };
  TEST_ASSERT(is_same(tMatrixEntries, tMatrixEntries_Gold));
}

/******************************************************************************/
/*! 
  \brief Create a block matrix, compute the row sum, and compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_RowSum)
{
  auto tMatrix = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValues = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  PlatoDevel::setMatrixData(tMatrix, tRowMap, tColMap, tValues);

  Plato::ScalarVector tRowSum("row sum", 12);

  PlatoDevel::RowSum( tMatrix, tRowSum );
  std::vector<Plato::Scalar> tRowSum_Gold = {
    20.0, 52.0, 84.0, 116.0, 10.0, 26.0, 42.0, 58.0, 10.0, 26.0, 42.0, 58.0
  };
  TEST_ASSERT(is_same(tRowSum, tRowSum_Gold));
}


/******************************************************************************/
/*! 
  \brief Create a block matrix, A, and a vector of diagonal weights, d={1.0}, and 
         compute the inverse weighted A: A[I][J]/d[I], and compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_InverseMultiply_1)
{
  auto tMatrix = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValues = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  PlatoDevel::setMatrixData(tMatrix, tRowMap, tColMap, tValues);

  Plato::ScalarVector tDiagonals("diagonals", 12);
  Plato::blas1::fill( 1.0, tDiagonals );
  
  PlatoDevel::InverseMultiply( tMatrix, tDiagonals );
  std::vector<Plato::Scalar> tGoldMatrixEntries =
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  auto tMatrixEntries = tMatrix->entries();
  TEST_ASSERT(is_same(tMatrixEntries, tGoldMatrixEntries));
}

/******************************************************************************/
/*! 
  \brief Create a block matrix, A, and a vector of diagonal weights, d={2.0}, and 
         compute the inverse weighted A: A[I][J]/d[I], and compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_InverseMultiply_2)
{
  auto tMatrix = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValues = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  PlatoDevel::setMatrixData(tMatrix, tRowMap, tColMap, tValues);

  Plato::ScalarVector tDiagonals("diagonals", 12);
  Plato::blas1::fill( 2.0, tDiagonals );
  
  PlatoDevel::InverseMultiply( tMatrix, tDiagonals );
  std::vector<Plato::Scalar> tGoldMatrixEntries =
    { 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
      0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
      0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8,
      0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8 };
  auto tMatrixEntries = tMatrix->entries();
  TEST_ASSERT(is_same(tMatrixEntries, tGoldMatrixEntries));
}

/******************************************************************************/
/*! 
  \brief Create block matrices, A and B, and apply the row summed inverse of A 
         to B: B[I][J]/d[I], where d[I] = Sum(A[I][J] in J). Compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_SlowDumbRowSummedInverseMultiply)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    PlatoDevel::setMatrixData(tMatrixA, tRowMap, tColMap, tValues);
  }

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    PlatoDevel::setMatrixData(tMatrixB, tRowMap, tColMap, tValues);
  }

  PlatoDevel::SlowDumbRowSummedInverseMultiply( tMatrixA, tMatrixB );
  std::vector<Plato::Scalar> tGoldMatrixEntries = {
   1.0/20.0,  2.0/20.0,  3.0/20.0,  4.0/20.0,  5.0/ 52.0,  6.0/ 52.0,  7.0/ 52.0,  8.0/ 52.0,
   9.0/84.0, 10.0/84.0, 11.0/84.0, 12.0/84.0, 13.0/116.0, 14.0/116.0, 15.0/116.0, 16.0/116.0,
   1.0/20.0,  2.0/20.0,  3.0/20.0,  4.0/20.0,  5.0/ 52.0,  6.0/ 52.0,  7.0/ 52.0,  8.0/ 52.0,
   9.0/84.0, 10.0/84.0, 11.0/84.0, 12.0/84.0, 13.0/116.0, 14.0/116.0, 15.0/116.0, 16.0/116.0,
   1.0/10.0,  2.0/10.0,  3.0/10.0,  4.0/10.0,  5.0/ 26.0,  6.0/ 26.0,  7.0/ 26.0,  8.0/ 26.0,
   9.0/42.0, 10.0/42.0, 11.0/42.0, 12.0/42.0, 13.0/ 58.0, 14.0/ 58.0, 15.0/ 58.0, 16.0/ 58.0,
   1.0/10.0,  2.0/10.0,  3.0/10.0,  4.0/10.0,  5.0/ 26.0,  6.0/ 26.0,  7.0/ 26.0,  8.0/ 26.0,
   9.0/42.0, 10.0/42.0, 11.0/42.0, 12.0/42.0, 13.0/ 58.0, 14.0/ 58.0, 15.0/ 58.0, 16.0/ 58.0
  };
  auto tMatrixEntries = tMatrixB->entries();
  TEST_ASSERT(is_same(tMatrixEntries, tGoldMatrixEntries));
}
/******************************************************************************/
/*! 
  \brief Create block matrices, A and B, and apply the row summed inverse of A 
         to B: B[I][J]/d[I], where d[I] = Sum(A[I][J] in J). Compare against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_RowSummedInverseMultiply)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    PlatoDevel::setMatrixData(tMatrixA, tRowMap, tColMap, tValues);
  }

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(12, 12, 4, 4) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    PlatoDevel::setMatrixData(tMatrixB, tRowMap, tColMap, tValues);
  }

  Plato::RowSummedInverseMultiply( tMatrixA, tMatrixB );
  std::vector<Plato::Scalar> tGoldMatrixEntries = {
   1.0/20.0,  2.0/20.0,  3.0/20.0,  4.0/20.0,  5.0/ 52.0,  6.0/ 52.0,  7.0/ 52.0,  8.0/ 52.0,
   9.0/84.0, 10.0/84.0, 11.0/84.0, 12.0/84.0, 13.0/116.0, 14.0/116.0, 15.0/116.0, 16.0/116.0,
   1.0/20.0,  2.0/20.0,  3.0/20.0,  4.0/20.0,  5.0/ 52.0,  6.0/ 52.0,  7.0/ 52.0,  8.0/ 52.0,
   9.0/84.0, 10.0/84.0, 11.0/84.0, 12.0/84.0, 13.0/116.0, 14.0/116.0, 15.0/116.0, 16.0/116.0,
   1.0/10.0,  2.0/10.0,  3.0/10.0,  4.0/10.0,  5.0/ 26.0,  6.0/ 26.0,  7.0/ 26.0,  8.0/ 26.0,
   9.0/42.0, 10.0/42.0, 11.0/42.0, 12.0/42.0, 13.0/ 58.0, 14.0/ 58.0, 15.0/ 58.0, 16.0/ 58.0,
   1.0/10.0,  2.0/10.0,  3.0/10.0,  4.0/10.0,  5.0/ 26.0,  6.0/ 26.0,  7.0/ 26.0,  8.0/ 26.0,
   9.0/42.0, 10.0/42.0, 11.0/42.0, 12.0/42.0, 13.0/ 58.0, 14.0/ 58.0, 15.0/ 58.0, 16.0/ 58.0
  };
  auto tMatrixEntries = tMatrixB->entries();
  TEST_ASSERT(is_same(tMatrixEntries, tGoldMatrixEntries));
}

/******************************************************************************/
/*! 
  \brief Create a condensed matrix: R = A - B RowSum(C)^-1 D

  Dimensions:
    A: Nf x Nf
    B: Nn x Nm
    C: Nm x Nm
    D: Nm x Nf
    R: Nf x Nf <= (Nf x Nf) - O(Nf,3)((Nn x Nm) . (Nm x Nm) . (Nm x Nf))

    where O(n,p)(M) expand a sub-block matrix, M, with row index, p, into
    a full-block matrix with number block rows, n.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_CondenseMatrix_1)
{
  const int Nf = 4;
  const int Nn = 1;
  const int Nm = 3;

  const int tNumNodes = 3;

  auto tA = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nf, tNumNodes*Nf, Nf, Nf) );
  auto tA_SlowDumb = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nf, tNumNodes*Nf, Nf, Nf) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    PlatoDevel::setMatrixData(tA, tRowMap, tColMap, tValues);
    PlatoDevel::setMatrixData(tA_SlowDumb, tRowMap, tColMap, tValues);
  }

  auto tB = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nn, tNumNodes*Nm, Nn, Nm) );
  auto tB_SlowDumb = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nn, tNumNodes*Nm, Nn, Nm) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3 };
    PlatoDevel::setMatrixData(tB, tRowMap, tColMap, tValues);
    PlatoDevel::setMatrixData(tB_SlowDumb, tRowMap, tColMap, tValues);
  }

  auto tC = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nm, tNumNodes*Nm, Nm, Nm) );
  auto tC_SlowDumb = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nm, tNumNodes*Nm, Nm, Nm) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9,
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        1, 2, 3, 4, 5, 6, 7, 8, 9,
        1, 2, 3, 4, 5, 6, 7, 8, 9 };
    PlatoDevel::setMatrixData(tC, tRowMap, tColMap, tValues);
    PlatoDevel::setMatrixData(tC_SlowDumb, tRowMap, tColMap, tValues);
  }

  auto tD = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nm, tNumNodes*Nf, Nm, Nf) );
  auto tD_SlowDumb = Teuchos::rcp( new Plato::CrsMatrixType(tNumNodes*Nm, tNumNodes*Nf, Nm, Nf) );
  {
    std::vector<Plato::OrdinalType> tRowMap = { 0, 2, 3, 4 };
    std::vector<Plato::OrdinalType> tColMap = { 0, 2, 0, 2 };
    std::vector<Plato::Scalar>      tValues = 
      { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    PlatoDevel::setMatrixData(tD, tRowMap, tColMap, tValues);
    PlatoDevel::setMatrixData(tD_SlowDumb, tRowMap, tColMap, tValues);
  }

  // Nn x Nf
  auto tMatrixProduct = Teuchos::rcp( new Plato::CrsMatrixType( tNumNodes*Nn, tNumNodes*Nf, Nn, Nf ) );
  auto tMatrixProduct_SlowDumb = Teuchos::rcp( new Plato::CrsMatrixType( tNumNodes*Nn, tNumNodes*Nf, Nn, Nf ) );

  // Nm x Nm . Nm x Nf => Nm x Nf
  Plato::RowSummedInverseMultiply              ( tC, tD );
  PlatoDevel::SlowDumbRowSummedInverseMultiply ( tC_SlowDumb, tD_SlowDumb );

  // Nn x Nm . Nm x Nf => Nn x Nf
  Plato::MatrixMatrixMultiply              ( tB, tD, tMatrixProduct );
  PlatoDevel::SlowDumbMatrixMatrixMultiply ( tB_SlowDumb, tD_SlowDumb, tMatrixProduct_SlowDumb );

  const int tOffset = 3;
  // Nf x Nf - O(Nf,3)(Nn x Nf)
  Plato::MatrixMinusMatrix              ( tA, tMatrixProduct, tOffset );
  PlatoDevel::SlowDumbMatrixMinusMatrix ( tA_SlowDumb, tMatrixProduct_SlowDumb, tOffset );

  TEST_ASSERT(is_equivalent(tA->rowMap(),
                            tA->columnIndices(), tA->entries(),
                            tA_SlowDumb->columnIndices(), tA_SlowDumb->entries()));
}
/******************************************************************************/
/*! 
  \brief Create a stabilized residual, g, and a projector, P, then compute
 derivatives dg/du^T, dg/dn^T, dP/du^T, and dP/dn and the condensed matrix:

 A = dg/du^T - dP/du^T . RowSum(dP/dn)^{-1} . dg/dn^T

*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_CondenseMatrix_2)
{
  constexpr int cSpaceDim  = 3;
  constexpr int cMeshWidth = 2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(cSpaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *gElastostaticsParams);

  using PhysicsT = Plato::StabilizedMechanics<cSpaceDim>;
  auto tResidual = createStabilizedResidual<PhysicsT>(tSpatialModel);
  auto tProjector = createStabilizedProjector<PhysicsT>(tSpatialModel);

  auto tNverts = tMesh->nverts();
  Plato::ScalarVector U("state",          tResidual->size());
  Plato::ScalarVector N("project p grad", tProjector->size());
  Plato::ScalarVector z("control",        tNverts);
  Plato::ScalarVector p("nodal pressure", tNverts);
  Plato::blas1::fill(1.0, z);

  //                                        u, n, z
  auto t_dg_du_T = tResidual->gradient_u_T (U, N, z);
  auto t_dg_dn_T = tResidual->gradient_n_T (U, N, z);
  auto t_dP_dn_T = tProjector->gradient_n_T(N, p, z);
  auto t_dP_du   = tProjector->gradient_u  (N, p, z);

  auto t_dg_du_T_sd = tResidual->gradient_u_T (U, N, z);
  auto t_dg_dn_T_sd = tResidual->gradient_n_T (U, N, z);
  auto t_dP_dn_T_sd = tProjector->gradient_n_T(N, p, z);
  auto t_dP_du_sd   = tProjector->gradient_u  (N, p, z);
  
  auto tNumRows = t_dP_dn_T->numRows();
  auto tNumCols = t_dg_dn_T->numCols();
  auto tNumRowsPerBlock = t_dP_dn_T->numRowsPerBlock();
  auto tNumColsPerBlock = t_dg_dn_T->numColsPerBlock();

  // Nn x Nf
  auto tMatrixProduct    = Teuchos::rcp( new Plato::CrsMatrixType(tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );
  auto tMatrixProduct_sd = Teuchos::rcp( new Plato::CrsMatrixType(tNumRows, tNumCols, tNumRowsPerBlock, tNumColsPerBlock) );

  // Nm x Nm . Nm x Nf => Nm x Nf
  Plato::RowSummedInverseMultiply              ( t_dP_du,    t_dg_dn_T    );
  PlatoDevel::SlowDumbRowSummedInverseMultiply ( t_dP_du_sd, t_dg_dn_T_sd );

  // Nn x Nm . Nm x Nf => Nn x Nf
  Plato::MatrixMatrixMultiply              ( t_dP_dn_T,    t_dg_dn_T,    tMatrixProduct    );
  PlatoDevel::SlowDumbMatrixMatrixMultiply ( t_dP_dn_T_sd, t_dg_dn_T_sd, tMatrixProduct_sd );

  auto tOffset = PhysicsT::ProjectorT::SimplexT::mProjectionDof;
  // Nf x Nf - O( Nn x Nf ) => Nf x Nf
  Plato::MatrixMinusMatrix              ( t_dg_du_T,    tMatrixProduct,    tOffset );
  PlatoDevel::SlowDumbMatrixMinusMatrix ( t_dg_du_T_sd, tMatrixProduct_sd, tOffset );

  TEST_ASSERT(is_equivalent(t_dg_du_T->rowMap(),
                            t_dg_du_T->columnIndices(), t_dg_du_T->entries(),
                            t_dg_du_T_sd->columnIndices(), t_dg_du_T_sd->entries()));
}

/******************************************************************************/
/*! 
  \brief Create a full block matrix, A, and a sub-block matrix, B, and
         compute C=A-B, then verify C against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_SlowDumbMatrixMinusMatrix_1)
{
  Plato::OrdinalType tOffset = 3;

  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType( 12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapA = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  PlatoDevel::setMatrixData(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(  3, 12, 1, 4) );
  std::vector<Plato::OrdinalType> tRowMapB = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapB = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesB = 
    { 13, 14, 15, 16,
      13, 14, 15, 16,
      13, 14, 15, 16,
      13, 14, 15, 16 };
  PlatoDevel::setMatrixData(tMatrixB, tRowMapB, tColMapB, tValuesB);

  PlatoDevel::SlowDumbMatrixMinusMatrix( tMatrixA, tMatrixB, tOffset );

  auto tMatrixC_Gold = Teuchos::rcp( new Plato::CrsMatrixType( 12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMapC = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapC = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesC = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0 };
  PlatoDevel::setMatrixData(tMatrixC_Gold, tRowMapC, tColMapC, tValuesC);

  TEST_ASSERT(is_same(tMatrixA, tMatrixC_Gold));
}
/******************************************************************************/
/*! 
  \brief Create a full non-square block matrix, A, with size nrows X ncols and
         a vector, a, with length nrows and a vector, b, with length ncols.
         compute c = a*A + b, then verify c against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_VectorTimesMatrixPlusVector)
{
  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType( 3, 12, 1, 4) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 3, 6, 9 };
  std::vector<Plato::OrdinalType> tColMapA = { 0, 1, 2, 0, 1, 2, 0, 1, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
  PlatoDevel::setMatrixData(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tMatrixA_full = ::PlatoUtestHelpers::toFull(tMatrixA);

  Plato::ScalarVector tVector_a("a", 3);
  std::vector<Plato::Scalar> tVector_a_full({12,11,10});
  PlatoDevel::setViewFromVector<Plato::Scalar>(tVector_a, tVector_a_full);

  Plato::ScalarVector tVector_b("b", 12);
  std::vector<Plato::Scalar> tVector_b_full({12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1});
  PlatoDevel::setViewFromVector<Plato::Scalar>(tVector_b, tVector_b_full);

  std::vector<Plato::Scalar> tVector_c_gold(12);
  for(int j=0; j<12; j++)
  {
    tVector_c_gold[j] = tVector_b_full[j];
    for(int i=0; i<3; i++)
    {
      tVector_c_gold[j] += tVector_a_full[i] * tMatrixA_full[i][j];
    }
  }

  Plato::VectorTimesMatrixPlusVector( tVector_a, tMatrixA, tVector_b );

  TEST_ASSERT(is_same(tVector_b, tVector_c_gold));
}
/******************************************************************************/
/*! 
  \brief Create a full block matrix, A, and a sub-block matrix, B, and
         compute C=A-B, then verify C against gold.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixMinusMatrix_1)
{
  Plato::OrdinalType tOffset = 3;

  auto tMatrixA = Teuchos::rcp( new Plato::CrsMatrixType( 12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMapA = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapA = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesA = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  PlatoDevel::setMatrixData(tMatrixA, tRowMapA, tColMapA, tValuesA);

  auto tMatrixB = Teuchos::rcp( new Plato::CrsMatrixType(  3, 12, 1, 4) );
  std::vector<Plato::OrdinalType> tRowMapB = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapB = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesB = 
    { 13, 14, 15, 16,
      13, 14, 15, 16,
      13, 14, 15, 16,
      13, 14, 15, 16 };
  PlatoDevel::setMatrixData(tMatrixB, tRowMapB, tColMapB, tValuesB);

  Plato::MatrixMinusMatrix( tMatrixA, tMatrixB, tOffset );

  auto tMatrixC_Gold = Teuchos::rcp( new Plato::CrsMatrixType( 12, 12, 4, 4) );
  std::vector<Plato::OrdinalType> tRowMapC = { 0, 2, 3, 4 };
  std::vector<Plato::OrdinalType> tColMapC = { 0, 2, 0, 2 };
  std::vector<Plato::Scalar>      tValuesC = 
    { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0,
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,  0,  0,  0,  0 };
  PlatoDevel::setMatrixData(tMatrixC_Gold, tRowMapC, tColMapC, tValuesC);

  TEST_ASSERT(is_same(tMatrixA, tMatrixC_Gold));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_InvertLocalMatrices)
{
    const int N = 3; // Number of matrices to invert
    Plato::ScalarArray3D tMatrix("Matrix A", N, 2, 2);
    auto tHostMatrix = Kokkos::create_mirror(tMatrix);
    for (unsigned int i = 0; i < N; ++i)
    {
      const Plato::Scalar tScaleFactor = 1.0 / (1.0 + i);
      tHostMatrix(i,0,0) = -2.0 * tScaleFactor;
      tHostMatrix(i,1,0) =  1.0 * tScaleFactor;
      tHostMatrix(i,0,1) =  1.5 * tScaleFactor;
      tHostMatrix(i,1,1) = -0.5 * tScaleFactor;
    }
    Kokkos::deep_copy(tMatrix, tHostMatrix);

    Plato::ScalarArray3D tAInverse("A Inverse", N, 2, 2);
    auto tHostAInverse = Kokkos::create_mirror(tAInverse);
    for (unsigned int i = 0; i < N; ++i)
    {
      tHostAInverse(i,0,0) = 1.0;
      tHostAInverse(i,1,0) = 0.0;
      tHostAInverse(i,0,1) = 0.0;
      tHostAInverse(i,1,1) = 1.0;
    }
    Kokkos::deep_copy(tAInverse, tHostAInverse);

    using namespace KokkosBatched;

    /// [template]AlgoType: Unblocked, Blocked, CompatMKL
    /// [in/out]A: 2d view
    /// [in]tiny: a magnitude scalar value compatible to the value type of A
    /// int SerialLU<Algo::LU::Unblocked>::invoke(const AViewType &A, const ScalarType tiny = 0)

    /// [template]SideType: Side::Left or Side::Right
    /// [template]UploType: Uplo::Upper or Uplo::Lower
    /// [template]TransType: Trans::NoTranspose or Trans::Transpose
    /// [template]DiagType: Diag::Unit or Diag::NonUnit
    /// [template]AlgoType: Unblocked, Blocked, CompatMKL
    /// [in]alpha: a scalar value
    /// [in]A: 2d view
    /// [in/out]B: 2d view
    /// int SerialTrsm<SideType,UploType,TransType,DiagType,AlgoType>
    ///    ::invoke(const ScalarType alpha, const AViewType &A, const BViewType &B);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,N), LAMBDA_EXPRESSION(const Plato::OrdinalType & n) {
      auto A    = Kokkos::subview(tMatrix  , n, Kokkos::ALL(), Kokkos::ALL());
      auto Ainv = Kokkos::subview(tAInverse, n, Kokkos::ALL(), Kokkos::ALL());

      SerialLU<Algo::LU::Blocked>::invoke(A);
      SerialTrsm<Side::Left,Uplo::Lower,Trans::NoTranspose,Diag::Unit   ,Algo::Trsm::Blocked>::invoke(1.0, A, Ainv);
      SerialTrsm<Side::Left,Uplo::Upper,Trans::NoTranspose,Diag::NonUnit,Algo::Trsm::Blocked>::invoke(1.0, A, Ainv);
    });

    const Plato::Scalar tTolerance = 1e-6;
    std::vector<std::vector<Plato::Scalar> > tGoldMatrixInverse = { {1.0, 3.0}, {2.0, 4.0} };

    Kokkos::deep_copy(tHostAInverse, tAInverse);
    for (unsigned int n = 0; n < N; ++n)
      for (unsigned int i = 0; i < 2; ++i)
        for (unsigned int j = 0; j < 2; ++j)
          {
            //printf("Matrix %d Inverse (%d,%d) = %f\n", n, i, j, tHostAInverse(n, i, j));
            const Plato::Scalar tScaleFactor = (1.0 + n);
            TEST_FLOATING_EQUALITY(tHostAInverse(n, i, j), tScaleFactor * tGoldMatrixInverse[i][j], tTolerance);
          }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, HyperbolicTangentProjection)
{
    const Plato::OrdinalType tNumNodesPerCell = 2;
    typedef Sacado::Fad::SFad<Plato::Scalar, tNumNodesPerCell> FadType;

    // SET EVALUATION TYPES FOR UNIT TEST
    const Plato::OrdinalType tNumCells = 1;
    Plato::ScalarVectorT<Plato::Scalar> tOutputVal("OutputVal", tNumCells);
    Plato::ScalarVectorT<Plato::Scalar> tOutputGrad("OutputGrad", tNumNodesPerCell);
    Plato::ScalarMultiVectorT<FadType> tControl("Control", tNumCells, tNumNodesPerCell);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tControl(aCellOrdinal, 0) = FadType(tNumNodesPerCell, 0, 1.0);
        tControl(aCellOrdinal, 1) = FadType(tNumNodesPerCell, 1, 1.0);
    }, "Set Controls");

    // SET EVALUATION TYPES FOR UNIT TEST
    Plato::HyperbolicTangentProjection tProjection;
    Plato::ApplyProjection<Plato::HyperbolicTangentProjection> tApplyProjection(tProjection);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        FadType tValue = tApplyProjection(aCellOrdinal, tControl);
        tOutputVal(aCellOrdinal) = tValue.val();
        tOutputGrad(0) = tValue.dx(0);
        tOutputGrad(1) = tValue.dx(1);
    }, "UnitTest: HyperbolicTangentProjection_GradZ");

    // TEST OUTPUT
    auto tHostVal = Kokkos::create_mirror(tOutputVal);
    Kokkos::deep_copy(tHostVal, tOutputVal);
    auto tHostGrad = Kokkos::create_mirror(tOutputGrad);
    Kokkos::deep_copy(tHostGrad, tOutputGrad);

    const Plato::Scalar tTolerance = 1e-6;
    std::vector<Plato::Scalar> tGoldVal = { 1.0 };
    std::vector<Plato::Scalar> tGoldGrad = { 4.539992985607449e-4, 4.539992985607449e-4 };
    TEST_FLOATING_EQUALITY(tHostVal(0), tGoldVal[0], tTolerance);
    TEST_FLOATING_EQUALITY(tHostGrad(0), tGoldGrad[0], tTolerance);
    TEST_FLOATING_EQUALITY(tHostGrad(1), tGoldGrad[1], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_ConditionalExpression)
{
    const Plato::OrdinalType tRange = 1;
    Plato::ScalarVector tOuput("Output", 2 /* number of outputs */);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tRange), LAMBDA_EXPRESSION(Plato::OrdinalType tOrdinal)
    {
        Plato::Scalar tConditionalValOne = 5;
        Plato::Scalar tConditionalValTwo = 4;
        Plato::Scalar tConsequentValOne = 2;
        Plato::Scalar tConsequentValTwo = 3;
        tOuput(tOrdinal) = Plato::conditional_expression(tConditionalValOne, tConditionalValTwo, tConsequentValOne, tConsequentValTwo);

        tConditionalValOne = 3;
        tOuput(tOrdinal + 1) = Plato::conditional_expression(tConditionalValOne, tConditionalValTwo, tConsequentValOne, tConsequentValTwo);
    }, "Test inline conditional_expression function");

    auto tHostOuput = Kokkos::create_mirror(tOuput);
    Kokkos::deep_copy(tHostOuput, tOuput);
    const Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tHostOuput(0), 3.0, tTolerance);
    TEST_FLOATING_EQUALITY(tHostOuput(1), 2.0, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_dot)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec A", tNumElems);
  Plato::blas1::fill(1.0, tVecA);
  Plato::ScalarVector tVecB("Vec B", tNumElems);
  Plato::blas1::fill(2.0, tVecB);

  const Plato::Scalar tOutput = Plato::blas1::dot(tVecA, tVecB);

  constexpr Plato::Scalar tTolerance = 1e-4;
  TEST_FLOATING_EQUALITY(20., tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_norm)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec A", tNumElems);
  Plato::blas1::fill(1.0, tVecA);

  const Plato::Scalar tOutput = Plato::blas1::norm(tVecA);
  constexpr Plato::Scalar tTolerance = 1e-6;
  TEST_FLOATING_EQUALITY(3.16227766016838, tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_sum)
{
  constexpr Plato::OrdinalType tNumElems = 10;
  Plato::ScalarVector tVecA("Vec", tNumElems);
  Plato::blas1::fill(1.0, tVecA);

  Plato::Scalar tOutput = 0.0;
  Plato::blas1::local_sum(tVecA, tOutput);

  constexpr Plato::Scalar tTolerance = 1e-4;
  TEST_FLOATING_EQUALITY(10., tOutput, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_fill)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numVerts = mesh->nverts();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::blas1::fill(2.0, tSomeVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), 2.0, 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), 2.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_copy)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numVerts = mesh->nverts();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::blas1::fill(2.0, tSomeVector);

  Plato::ScalarVector tSomeOtherVector("some other vector", numVerts);
  Plato::blas1::copy(tSomeVector, tSomeOtherVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  auto tSomeOtherVectorHost = Kokkos::create_mirror_view(tSomeOtherVector);
  Kokkos::deep_copy(tSomeOtherVectorHost, tSomeOtherVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), tSomeOtherVectorHost(0), 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), tSomeOtherVectorHost(numVerts-1), 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_scale)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numVerts = mesh->nverts();
  
  Plato::ScalarVector tSomeVector("some vector", numVerts);
  Plato::blas1::fill(1.0, tSomeVector);
  Plato::blas1::scale(2.0, tSomeVector);

  auto tSomeVectorHost = Kokkos::create_mirror_view(tSomeVector);
  Kokkos::deep_copy(tSomeVectorHost, tSomeVector);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(0), 2.0, 1e-17);
  TEST_FLOATING_EQUALITY(tSomeVectorHost(numVerts-1), 2.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_update)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numVerts = mesh->nverts();
  
  Plato::ScalarVector tVector_A("vector a", numVerts);
  Plato::ScalarVector tVector_B("vector b", numVerts);
  Plato::blas1::fill(1.0, tVector_A);
  Plato::blas1::fill(2.0, tVector_B);
  Plato::blas1::update(2.0, tVector_A, 3.0, tVector_B);

  auto tVector_B_Host = Kokkos::create_mirror_view(tVector_B);
  Kokkos::deep_copy(tVector_B_Host, tVector_B);
  TEST_FLOATING_EQUALITY(tVector_B_Host(0), 8.0, 1e-17);
  TEST_FLOATING_EQUALITY(tVector_B_Host(numVerts-1), 8.0, 1e-17);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMathHelpers_MatrixTimesVectorPlusVector)
{
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( tMesh->nverts(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);

  // create mesh based displacement from host data
  //
  auto stateSize = spaceDim*tMesh->nverts();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, stateSize);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view(u);
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for( int i = 0; i<stateSize; i++) u_host(i) = (disp += dval);
  Kokkos::deep_copy(u, u_host);

  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Internal Elastic Energy'>                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Elliptic'>                                             \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create criterion
  //
  Plato::DataMap tDataMap;
  std::string tMyFunction("Internal Elastic Energy");
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParams);

  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<spaceDim>>
    eeScalarFunction(tSpatialModel, tDataMap, *tParams, tMyFunction);

  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto dfdx = eeScalarFunction.gradient_x(tSolution, z);

  // create PDE constraint
  //
  Plato::Elliptic::VectorFunction<::Plato::Mechanics<spaceDim>>
    esVectorFunction(tSpatialModel, tDataMap, *tParams, tParams->get<std::string>("PDE Constraint"));

  auto dgdx = esVectorFunction.gradient_x(u,z);

#ifdef COMPUTE_GOLD_
  {
    auto dfdxHost = Kokkos::create_mirror_view(dfdx); 
    Kokkos::deep_copy(dfdxHost, dfdx);
    std::ofstream ofile;
    ofile.open("dfdx_before.dat");
    for(int i=0; i<dfdxHost.size(); i++) 
      ofile << std::setprecision(18) << dfdxHost(i) << std::endl;
    ofile.close();
  }
#endif

  Plato::MatrixTimesVectorPlusVector(dgdx, (Plato::ScalarVector)u, dfdx);

  auto dfdx_host = Kokkos::create_mirror_view(dfdx);
  Kokkos::deep_copy(dfdx_host, dfdx);

#ifdef COMPUTE_GOLD_
  {
    auto dfdxHost = Kokkos::create_mirror_view(dfdx); 
    Kokkos::deep_copy(dfdxHost, dfdx);
    std::ofstream ofile;
    ofile.open("dfdx_after.dat");
    for(int i=0; i<dfdxHost.size(); i++) 
      ofile << std::setprecision(18) << dfdxHost(i) << std::endl;
    ofile.close();
  }

  {
    std::ofstream ofile;
    ofile.open("u.dat");
    for(int i=0; i<u_host.size(); i++) 
      ofile << u_host(i) << std::endl;
    ofile.close();
  }

  {
    auto rowMapHost = Kokkos::create_mirror_view(dgdx->rowMap()); 
    Kokkos::deep_copy(rowMapHost, dgdx->rowMap());
    std::ofstream ofile;
    ofile.open("rowMap.dat");
    for(int i=0; i<rowMapHost.size(); i++) 
      ofile << rowMapHost(i) << std::endl;
    ofile.close();
  }

  {
    auto columnIndicesHost = Kokkos::create_mirror_view(dgdx->columnIndices());
    Kokkos::deep_copy(columnIndicesHost, dgdx->columnIndices());
    std::ofstream ofile;
    ofile.open("columnIndices.dat");
    for(int i=0; i<columnIndicesHost.size(); i++) 
      ofile << columnIndicesHost(i) << std::endl;
    ofile.close();
  }

  {
    auto entriesHost = Kokkos::create_mirror_view(dgdx->entries());
    Kokkos::deep_copy(entriesHost, dgdx->entries());
    std::ofstream ofile;
    ofile.open("entries.dat");
    for(int i=0; i<entriesHost.size(); i++) 
      ofile << std::setprecision(18) << entriesHost(i) << std::endl;
    ofile.close();
  }
#endif

  std::vector<Plato::Scalar> dfdx_gold = {
    73.3153846153846,-47.0163461538462,-21.8596153846154,
    68.8629807692308,-53.5716346153847,6.80192307692308,
    4.52163461538462,-14.1706730769231,23.1490384615385,
    35.2860576923076,23.0581730769231,-8.91346153846153,
    -5.02788461538462,2.85144230769231,19.2591346153846,
    -1.35,7.94423076923076,5.86730769230769,
    -3.96346153846154,17.7923076923077,-4.4826923076923,
    -0.917307692307696,6.19615384615385,-2.78653846153846,
    -10.0514423076923,1.82596153846153,-13.1668269230769,
    33.2134615384616,-4.18413461538466,-14.3870192307693,
    18.7052884615385,-23.0408653846154,-1.0125,
    2.475,3.72115384615385,-1.21153846153846,
    -0.051923076923075,6.43846153846155,-7.6326923076923,
    -13.6081730769229,16.7798076923077,14.8975961538462,
    -32.733173076923,15.5033653846154,29.3019230769231,
    -30.7298076923077,4.98028846153846,12.6649038461538,
    -25.0052884615385,6.91442307692306,-5.27451923076924,
    4.08028846153847,5.11875000000002,-11.4836538461539,
    -4.49134615384617,1.94711538461539,2.36250000000001,
    5.36105769230771,-2.72596153846155,4.73798076923079,
    8.79230769230771,11.9206730769231,0.912980769230768,
    -45.1990384615385,-21.8596153846153,4.8548076923077,
    -25.5850961538462,-14.7721153846154,16.3168269230769,
    -0.173076923076924,0.346153846153847,0.65769230769231,
    0.220673076923078,0.700961538461542,2.55721153846155,
    -65.8038461538463,42.364903846154,-42.529326923077,
    9.8567307692308,4.93701923076925,-9.60144230769233};

  for(int iNode=0; iNode<int(dfdx_gold.size()); iNode++){
      TEST_FLOATING_EQUALITY(dfdx_host[iNode], dfdx_gold[iNode], 1e-13);
  }
}

} // namespace PlatoUnitTests

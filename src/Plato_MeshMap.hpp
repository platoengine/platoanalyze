/*
//@HEADER
// *************************************************************************
//   Plato Engine v.1.0: Copyright 2018, National Technology & Engineering
//                    Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Sandia Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact the Plato team (plato3D-help@sandia.gov)
//
// *************************************************************************
//@HEADER
*/

/*!
 * Plato_MeshMap.hpp
 *
 * Created on: Nov 13, 2019
 *
 */

#ifndef PLATO_MESHMAP_HPP_
#define PLATO_MESHMAP_HPP_

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include "alg/PlatoLambda.hpp"

#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_file.hpp>

#include "Plato_MeshMapUtils.hpp"

namespace Plato {
namespace Geometry {

template <typename Scalar_T>
struct MathMapBase
{
    using ScalarT = Scalar_T;
};

/***************************************************************************//**
* @brief Functor for no prescribed symmetry
*******************************************************************************/
template <typename ScalarT>
struct Full : public MathMapBase<ScalarT>
{
    using VectorArrayT = typename Plato::ScalarMultiVectorT<ScalarT>;
    using OrdinalT     = typename VectorArrayT::size_type;

    ScalarT mOrigin[cSpaceDim];
    ScalarT mNormal[cSpaceDim];

    Full(const Plato::InputData & aInput){}

    DEVICE_TYPE inline void
    operator()( OrdinalT aOrdinal, VectorArrayT aInValue, VectorArrayT aOutValue ) const
    {
        aOutValue(Dim::X, aOrdinal) = aInValue(Dim::X, aOrdinal);
        aOutValue(Dim::Y, aOrdinal) = aInValue(Dim::Y, aOrdinal);
        aOutValue(Dim::Z, aOrdinal) = aInValue(Dim::Z, aOrdinal);
    }
};

/***************************************************************************//**
* @brief Functor for mirror plane symmetry

  The mirror plane is defined by an origin and a normal vector.  The given
  normal vector is unitized during initialization.  Points that have a negative
  projection onto the plane are reflected, that is, the positive side
  is the parent side and the negative side is the child side.
*******************************************************************************/
template <typename ScalarT>
struct SymmetryPlane : public MathMapBase<ScalarT>
{
    using VectorArrayT = typename Plato::ScalarMultiVectorT<ScalarT>;
    using OrdinalT     = typename VectorArrayT::size_type;

    ScalarT mOrigin[cSpaceDim];
    ScalarT mNormal[cSpaceDim];

    SymmetryPlane(const Plato::InputData & aInput)
    {
        auto tOriginInput = aInput.get<Plato::InputData>("Origin");
        mOrigin[Dim::X] = Plato::Get::Double(tOriginInput, "X");
        mOrigin[Dim::Y] = Plato::Get::Double(tOriginInput, "Y");
        mOrigin[Dim::Z] = Plato::Get::Double(tOriginInput, "Z");

        auto tNormalInput = aInput.get<Plato::InputData>("Normal");
        mNormal[Dim::X] = Plato::Get::Double(tNormalInput, "X");
        mNormal[Dim::Y] = Plato::Get::Double(tNormalInput, "Y");
        mNormal[Dim::Z] = Plato::Get::Double(tNormalInput, "Z");

        auto tLength = mNormal[Dim::X] * mNormal[Dim::X]
                     + mNormal[Dim::Y] * mNormal[Dim::Y]
                     + mNormal[Dim::Z] * mNormal[Dim::Z];

        if( tLength == 0.0 )
        {
            throw Plato::ParsingException("SymmetryPlane: Normal vector has zero length.");
        }
        tLength = sqrt(tLength);
        mNormal[Dim::X] /= tLength;
        mNormal[Dim::Y] /= tLength;
        mNormal[Dim::Z] /= tLength;
    }
    DEVICE_TYPE inline void
    operator()( OrdinalT aOrdinal, VectorArrayT aInValue, VectorArrayT aOutValue ) const
    {
        ScalarT tProjVal = 0.0;
        tProjVal += (aInValue(Dim::X, aOrdinal) - mOrigin[Dim::X]) * mNormal[Dim::X];
        tProjVal += (aInValue(Dim::Y, aOrdinal) - mOrigin[Dim::Y]) * mNormal[Dim::Y];
        tProjVal += (aInValue(Dim::Z, aOrdinal) - mOrigin[Dim::Z]) * mNormal[Dim::Z];

        aOutValue(Dim::X, aOrdinal) = aInValue(Dim::X, aOrdinal);
        aOutValue(Dim::Y, aOrdinal) = aInValue(Dim::Y, aOrdinal);
        aOutValue(Dim::Z, aOrdinal) = aInValue(Dim::Z, aOrdinal);

        if( tProjVal < 0.0 )
        {
            aOutValue(Dim::X, aOrdinal) -= 2.0*tProjVal*mNormal[Dim::X];
            aOutValue(Dim::Y, aOrdinal) -= 2.0*tProjVal*mNormal[Dim::Y];
            aOutValue(Dim::Z, aOrdinal) -= 2.0*tProjVal*mNormal[Dim::Z];
        }
    }
};

/***************************************************************************//**
* @brief Functor for translation

  Given a point and translation vector, the corresponding translated
  point is found.
*******************************************************************************/
template <typename ScalarT>
struct Translation : public MathMapBase<ScalarT>
{
    using VectorArrayT = typename Plato::ScalarMultiVectorT<ScalarT>;
    using OrdinalT     = typename VectorArrayT::size_type;

    ScalarT mTranslation[cSpaceDim];

    Translation(const Plato::InputData & aInput)
    {
        auto tTranslationInput = aInput.get<Plato::InputData>("Vector");
        mTranslation[Dim::X] = Plato::Get::Double(tTranslationInput, "X");
        mTranslation[Dim::Y] = Plato::Get::Double(tTranslationInput, "Y");
        mTranslation[Dim::Z] = Plato::Get::Double(tTranslationInput, "Z");

        auto tLength = mTranslation[Dim::X] * mTranslation[Dim::X]
                     + mTranslation[Dim::Y] * mTranslation[Dim::Y]
                     + mTranslation[Dim::Z] * mTranslation[Dim::Z];

        if( tLength == 0.0 )
        {
            throw Plato::ParsingException("Translation: Vector has zero length.");
        }
    }

    DEVICE_TYPE inline void
    operator()( OrdinalT aOrdinal, VectorArrayT aInValue, VectorArrayT aOutValue ) const
    {
        aOutValue(Dim::X, aOrdinal) = aInValue(Dim::X, aOrdinal) + mTranslation[Dim::X];
        aOutValue(Dim::Y, aOrdinal) = aInValue(Dim::Y, aOrdinal) + mTranslation[Dim::Y];
        aOutValue(Dim::Z, aOrdinal) = aInValue(Dim::Z, aOrdinal) + mTranslation[Dim::Z];
    }
};

/***************************************************************************//**
* @brief MeshMap

   This base class contains most of the functionality needed for a MeshMap. The
   MathMap (i.e., Full, SymmetryPlane, etc) is added in the template derived
   class.
*******************************************************************************/
template <typename ScalarT>
class MeshMap
{
  public:
    using ScalarArrayT  = typename Plato::ScalarVectorT<ScalarT>;
    using VectorArrayT  = typename Plato::ScalarMultiVectorT<ScalarT>;
    using OrdinalT      = typename ScalarArrayT::size_type;
    using OrdinalArrayT = typename Plato::ScalarVectorT<OrdinalT>;
    using IntegerArrayT = typename Plato::ScalarVectorT<int>;

    MeshMap(Omega_h::Mesh& aMesh, Plato::InputData& aInput) :
      mFilter(nullptr),
      mFilterT(nullptr),
      mFilterFirst(Plato::Get::Bool(aInput, "FilterFirst", /*default=*/ true)) {}

    struct SparseMatrix {
        OrdinalArrayT mRowMap;
        OrdinalArrayT mColMap;
        ScalarArrayT  mEntries;

        OrdinalT mNumRows;
        OrdinalT mNumCols;
    };

#ifndef MAKE_PUBLIC
  protected:
#else
  public:
#endif
    SparseMatrix mMatrix;
    SparseMatrix mMatrixT;
    std::shared_ptr<SparseMatrix> mFilter;
    std::shared_ptr<SparseMatrix> mFilterT;
    bool mFilterFirst;

  public:
    /***************************************************************************//**
    * @brief Set map matrix values from parent element
    *******************************************************************************/
    void setMatrixValues(Omega_h::Mesh& aMesh, IntegerArrayT aParentElements, VectorArrayT aLocation, SparseMatrix& aMatrix)
    {
        auto tNVerts = aMesh.nverts();
        aMatrix.mNumRows = tNVerts;
        aMatrix.mNumCols = tNVerts;

        // determine rowmap
        auto tNumRows = aMatrix.mNumRows;
        OrdinalArrayT tRowMap("row map", tNumRows+1);
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalT iRowOrdinal)
        {
            if( aParentElements(iRowOrdinal) == -2 ) // no parent element found
            {
                tRowMap(iRowOrdinal) = 0;
            }
            else
            if( aParentElements(iRowOrdinal) == -1 ) // not mapped
            {
                tRowMap(iRowOrdinal) = 1;
            }
            else
            {
                tRowMap(iRowOrdinal) = cNVertsPerElem; // mapped
            }
        }, "nonzeros");

        OrdinalT tNumEntries(0);
        Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalT>(0,tNumRows+1),
        KOKKOS_LAMBDA (const OrdinalT& iOrdinal, OrdinalT& aUpdate, const bool& tIsFinal)
        {
            const OrdinalT tVal = tRowMap(iOrdinal);
            if( tIsFinal )
            {
              tRowMap(iOrdinal) = aUpdate;
            }
            aUpdate += tVal;
        }, tNumEntries);
        aMatrix.mRowMap = tRowMap;

        // determine column map and entries
        OrdinalArrayT tColMap("row map", tNumEntries);
        ScalarArrayT tEntries("entries", tNumEntries);
        GetBasis<ScalarT> tGetBasis(aMesh);
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalT iRowOrdinal)
        {
            auto iEntryOrdinal = tRowMap(iRowOrdinal);
            auto iElemOrdinal = aParentElements(iRowOrdinal);
            if( iElemOrdinal == -1 ) // not mapped
            {
                tColMap(iEntryOrdinal) = iRowOrdinal;
                tEntries(iEntryOrdinal) = 1.0;
            }
            else
            {
                tGetBasis(aLocation, iRowOrdinal, iElemOrdinal, iEntryOrdinal, tColMap, tEntries);
            }

        }, "colmap and entries");
        mMatrix.mColMap = tColMap;
        mMatrix.mEntries = tEntries;
    }

    /***************************************************************************//**
    * @brief Compute transpose if input matrix exists
    *******************************************************************************/
    inline std::shared_ptr<SparseMatrix>
    createTranspose(std::shared_ptr<SparseMatrix> aMatrix)
    {
        auto tRetMatrix = std::make_shared<SparseMatrix>();
        if( aMatrix != nullptr )
        {
            *tRetMatrix = createTranspose(*aMatrix);
        }
        else
        {
            tRetMatrix = nullptr;
        }
        return tRetMatrix;
    }

    /***************************************************************************//**
    * @brief Compute transpose (existence of input matrix is assumed)
    *******************************************************************************/
    inline SparseMatrix
    createTranspose(SparseMatrix& aMatrix)
    {
        SparseMatrix tRetMatrix;

        OrdinalArrayT tRowMapT("row map", aMatrix.mRowMap.size());
        tRetMatrix.mRowMap = tRowMapT;

        OrdinalArrayT tColMapT("col map", aMatrix.mColMap.size());
        tRetMatrix.mColMap = tColMapT;

        ScalarArrayT tEntriesT("entries", aMatrix.mEntries.size());
        tRetMatrix.mEntries = tEntriesT;

        tRetMatrix.mNumRows = aMatrix.mNumCols;
        tRetMatrix.mNumCols = aMatrix.mNumRows;

        auto tRowMap = aMatrix.mRowMap;
        auto tColMap = aMatrix.mColMap;
        auto tEntries = aMatrix.mEntries;

        // determine rowmap
        auto tNumRows = aMatrix.mNumRows;
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalT iRowOrdinal)
        {
            auto tRowStart = tRowMap(iRowOrdinal);
            auto tRowEnd = tRowMap(iRowOrdinal + 1);
            for (auto tEntryIndex = tRowStart; tEntryIndex < tRowEnd; tEntryIndex++)
            {
                auto iColumnIndex = tColMap(tEntryIndex);
                Kokkos::atomic_increment(&tRowMapT(iColumnIndex));
            }
        }, "nonzeros");

        OrdinalT tNumEntries(0);
        Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalT>(0,tNumRows+1),
        KOKKOS_LAMBDA (const OrdinalT& iOrdinal, OrdinalT& aUpdate, const bool& tIsFinal)
        {
            const OrdinalT tVal = tRowMapT(iOrdinal);
            if( tIsFinal )
            {
              tRowMapT(iOrdinal) = aUpdate;
            }
            aUpdate += tVal;
        }, tNumEntries);

        // determine column map and entries
        OrdinalArrayT tOffsetT("offsets", tNumRows);
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalT iRowOrdinal)
        {
            auto tRowStart = tRowMap(iRowOrdinal);
            auto tRowEnd = tRowMap(iRowOrdinal + 1);
            for (auto iEntryIndex = tRowStart; iEntryIndex < tRowEnd; iEntryIndex++)
            {
                auto iRowIndexT = tColMap(iEntryIndex);
                auto tMyOffset = Kokkos::atomic_fetch_add(&tOffsetT(iRowIndexT), 1);
                auto iEntryIndexT = tRowMapT(iRowIndexT)+tMyOffset;
                tColMapT(iEntryIndexT) = iRowOrdinal;
                tEntriesT(iEntryIndexT) = tEntries(iEntryIndex);
            }
        }, "colmap and entries");

        return tRetMatrix;
    }


    /***************************************************************************//**
    * @brief Create linear filter
    *******************************************************************************/
    inline void
    createLinearFilter(ScalarT aRadius, SparseMatrix& aMatrix, VectorArrayT aLocations)
    {
        // conduct search
        //
        auto d_x = Kokkos::subview(aLocations, (size_t)Dim::X, Kokkos::ALL());
        auto d_y = Kokkos::subview(aLocations, (size_t)Dim::Y, Kokkos::ALL());
        auto d_z = Kokkos::subview(aLocations, (size_t)Dim::Z, Kokkos::ALL());
        decltype(d_x) d_r("radii", d_x.size());
        Kokkos::deep_copy(d_r, aRadius);

        ArborX::BVH<DeviceType>
          bvh{Points{d_x.data(), d_y.data(), d_z.data(), (int)d_x.size()}};

        Kokkos::View<int*, DeviceType> tIndices("indices", 0), tOffset("offset", 0);
        bvh.query(Spheres{d_x.data(), d_y.data(), d_z.data(), d_r.data(), (int)d_x.size()}, tIndices, tOffset);

        // create matrix entries
        //
        setLinearFilterMatrixValues(aRadius, aMatrix, aLocations, tIndices, tOffset);
    }

    /***************************************************************************//**
    * @brief Set linear filter matrix values
    *******************************************************************************/
    inline void
    setLinearFilterMatrixValues(
      ScalarT aRadius,
      SparseMatrix& aMatrix,
      VectorArrayT aLocations,
      Kokkos::View<int*, DeviceType> aIndices,
      Kokkos::View<int*, DeviceType> aOffset)
    {
        aMatrix.mNumRows = aLocations.extent(1);
        aMatrix.mNumCols = aLocations.extent(1);

        // determine rowmap
        auto tNumRows = aMatrix.mNumRows;
        OrdinalArrayT tRowMap("row map", tNumRows+1);
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalT iRowOrdinal)
        {
            tRowMap(iRowOrdinal) = aOffset(iRowOrdinal+1) - aOffset(iRowOrdinal);
        }, "nonzeros");

        OrdinalT tNumEntries(0);
        Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalT>(0,tNumRows+1),
        KOKKOS_LAMBDA (const OrdinalT& iOrdinal, OrdinalT& aUpdate, const bool& tIsFinal)
        {
            const OrdinalT tVal = tRowMap(iOrdinal);
            if( tIsFinal )
            {
              tRowMap(iOrdinal) = aUpdate;
            }
            aUpdate += tVal;
        }, tNumEntries);
        aMatrix.mRowMap = tRowMap;

        // determine column map and entries
        auto tRadius = aRadius;
        OrdinalArrayT tColMap("row map", tNumEntries);
        ScalarArrayT tEntries("entries", tNumEntries);
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalT iRowOrdinal)
        {
            auto iMatrixEntryOrdinal = tRowMap(iRowOrdinal);
            auto x = aLocations(Dim::X, iRowOrdinal);
            auto y = aLocations(Dim::Y, iRowOrdinal);
            auto z = aLocations(Dim::Z, iRowOrdinal);
            decltype(x) tTotalWeight(0.0);
            for( int iOffset=aOffset(iRowOrdinal); iOffset<aOffset(iRowOrdinal+1); iOffset++ )
            {
                auto iVertOrdinal = aIndices(iOffset);
                tColMap(iMatrixEntryOrdinal) = iVertOrdinal;
                auto dx = x - aLocations(Dim::X, iVertOrdinal);
                auto dy = y - aLocations(Dim::Y, iVertOrdinal);
                auto dz = z - aLocations(Dim::Z, iVertOrdinal);
                auto d2 = dx*dx + dy*dy + dz*dz;
                auto tDistance = (d2 > 0.0) ? sqrt(d2) : 0.0;
                auto tEntry = tRadius - tDistance;
                tTotalWeight += tEntry;
                tEntries(iMatrixEntryOrdinal++) = tRadius - tDistance;
            }
            iMatrixEntryOrdinal = tRowMap(iRowOrdinal);
            for( int iOffset=aOffset(iRowOrdinal); iOffset<aOffset(iRowOrdinal+1); iOffset++ )
            {
                tEntries(iMatrixEntryOrdinal++) /= tTotalWeight;
            }
        }, "colmap and entries");
        aMatrix.mColMap = tColMap;
        aMatrix.mEntries = tEntries;
    }

    /***************************************************************************//**
    * @brief create filter of type specified in input
    *******************************************************************************/
    inline std::shared_ptr<SparseMatrix>
    createFilter(Plato::InputData aFilterSpec, VectorArrayT aLocations)
    {
        auto tRetMatrix = std::make_shared<SparseMatrix>();
        auto tFilterType = Plato::Get::String(aFilterSpec, "Type", /*to_upper=*/ true);
        if( tFilterType == "LINEAR" )
        {
            auto tFilterRadius = Plato::Get::Double(aFilterSpec, "Radius");
            if( tFilterRadius <= 0.0 )
            {
                throw Plato::ParsingException("Filter 'Radius' must be greater than zero.");
            }
            createLinearFilter(tFilterRadius, *tRetMatrix, aLocations);
        }
        else
        {
            tRetMatrix = nullptr;
        }
        return tRetMatrix;
    }


  public:

    /***************************************************************************//**
    * @brief Apply mapping
    *******************************************************************************/
    void apply(const ScalarArrayT & aInput, ScalarArrayT aOutput)
    {
        if( mFilter != nullptr )
        {
            if( mFilterFirst )
            {
                matvec(*mFilter, aInput, aOutput);
                matvec(mMatrix, aOutput);
            }
            else
            {
                matvec(mMatrix, aInput, aOutput);
                matvec(*mFilter, aOutput);
            }
        }
        else
        {
            matvec(mMatrix, aInput, aOutput);
        }
    }

    /***************************************************************************//**
    * @brief Matrix times vector in place (overwrites input vector)
    *******************************************************************************/
    void matvec(const SparseMatrix & aMatrix, const ScalarArrayT & aInput)
    {
        ScalarArrayT tOutput("output vector", aInput.extent(0));
        Kokkos::deep_copy(tOutput, aInput);
        matvec(aMatrix, aInput, tOutput);
        Kokkos::deep_copy(aInput, tOutput);
    }

    /***************************************************************************//**
    * @brief Matrix times vector
    *******************************************************************************/
    void matvec(const SparseMatrix & aMatrix, const ScalarArrayT & aInput, ScalarArrayT aOutput)
    {
        auto tRowMap = aMatrix.mRowMap;
        auto tColMap = aMatrix.mColMap;
        auto tEntries = aMatrix.mEntries;
        auto tNumRows = tRowMap.size() - 1;

        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalT aRowOrdinal)
        {
            auto tRowStart = tRowMap(aRowOrdinal);
            auto tRowEnd = tRowMap(aRowOrdinal + 1);
            ScalarT tSum = 0.0;
            for (auto tEntryIndex = tRowStart; tEntryIndex < tRowEnd; tEntryIndex++)
            {
                auto tColumnIndex = tColMap(tEntryIndex);
                tSum += tEntries(tEntryIndex) * aInput(tColumnIndex);
            }
            aOutput(aRowOrdinal) = tSum;
        },"Matrix * Vector");
    }

    /***************************************************************************//**
    * @brief Apply transpose of mapping
    *******************************************************************************/
    void applyT(const ScalarArrayT & aInput, ScalarArrayT aOutput)
    {
        if( mFilterT != nullptr )
        {
            if( !mFilterFirst )
            {
                matvec(*mFilterT, aInput, aOutput);
                matvec(mMatrixT, aOutput);
            }
            else
            {
                matvec(mMatrixT, aInput, aOutput);
                matvec(*mFilterT, aOutput);
            }
        }
        else
        {
            matvec(mMatrixT, aInput, aOutput);
        }
    }
};

/***************************************************************************//**
* @brief Derived class template that adds MathMap functionality.
*******************************************************************************/
template <typename MathMapType>
class MeshMapDerived : public Plato::Geometry::MeshMap<typename MathMapType::ScalarT>
{
    MathMapType mMathMap;

    using ScalarT       = typename MathMapType::ScalarT;
    using ScalarArrayT  = typename Plato::ScalarVectorT<ScalarT>;
    using IntegerArrayT = typename Plato::ScalarVectorT<int>;
    using VectorArrayT  = typename Plato::ScalarMultiVectorT<ScalarT>;
    using SparseMatrix  = typename MeshMap<ScalarT>::SparseMatrix;
    using OrdinalT      = typename ScalarArrayT::size_type;
    using OrdinalArrayT = typename Plato::ScalarVectorT<OrdinalT>;

    using MapBase = MeshMap<ScalarT>;
    using MapBase::mMatrix;
    using MapBase::mMatrixT;
    using MapBase::mFilter;
    using MapBase::mFilterT;

  public:

    MeshMapDerived(Omega_h::Mesh& aMesh, Plato::InputData& aInput) :
      MeshMap<typename MathMapType::ScalarT>(aMesh, aInput),
      mMathMap(aInput.get<Plato::InputData>("LinearMap"))
    {
        // compute mapped values
        //
        auto tNVerts = aMesh.nverts();
        VectorArrayT tVertexLocations       ("mesh node locations",        cSpaceDim, tNVerts);
        VectorArrayT tMappedVertexLocations ("mapped mesh node locations", cSpaceDim, tNVerts);
        mapVertexLocations(aMesh, tVertexLocations, tMappedVertexLocations);


        // find elements that contain mapped locations
        //
        IntegerArrayT tParentElements("mapped mask", tNVerts);
        findParentElements<ScalarT>(aMesh, tVertexLocations, tMappedVertexLocations, tParentElements);


        // populate crs matrix
        //
        MapBase::setMatrixValues(aMesh, tParentElements, tMappedVertexLocations, mMatrix);
        mMatrixT = MapBase::createTranspose(mMatrix);

        // build filter if requested
        //
        auto tFilterSpec = aInput.get_add<Plato::InputData>("Filter");
        mFilter  = MapBase::createFilter(tFilterSpec, tVertexLocations);
        mFilterT = MapBase::createTranspose(mFilter);
    }

    void mapVertexLocations(Omega_h::Mesh& aMesh, VectorArrayT aLocations, VectorArrayT aMappedLocations)
    {
        auto tCoords = aMesh.coords();
        auto tNVerts = aMesh.nverts();
        auto tMathMap = mMathMap;
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNVerts), LAMBDA_EXPRESSION(OrdinalT iOrdinal)
        {
            for(size_t iDim=0; iDim<cSpaceDim; ++iDim)
            {
                aLocations(iDim, iOrdinal) = tCoords[iOrdinal*cSpaceDim+iDim];
            }
            tMathMap(iOrdinal, aLocations, aMappedLocations);
        }, "get verts and apply map");
    }

    ~MeshMapDerived()
    {
    }
}; // end class MeshMapDerived



template <typename ScalarT = double>
struct MeshMapFactory
{

    inline std::shared_ptr<Plato::Geometry::MeshMap<ScalarT>>
    create(Plato::InputData aInput)
    {
        // load mesh
        //
        auto tMeshFileName = Plato::Get::String(aInput, "Mesh");
        Omega_h::Library tLibOSH(0, nullptr);
        Omega_h::Mesh tMesh = Omega_h::read_mesh_file(tMeshFileName, tLibOSH.world());
        tMesh.set_parting(Omega_h_Parting::OMEGA_H_GHOSTED);

        return create(tMesh, aInput);
    }

    inline std::shared_ptr<Plato::Geometry::MeshMap<ScalarT>>
    create(Omega_h::Mesh& aMesh, Plato::InputData aInput)
    {
        auto tLinearMapInputs = aInput.getByName<Plato::InputData>("LinearMap");
        if( tLinearMapInputs.size() == 0 )
        {
            throw Plato::ParsingException("Attempted to create a MeshMap with no Map");
        }
        else
        if( tLinearMapInputs.size() > 1 )
        {
            throw Plato::ParsingException("MeshMap creation failed.  Multiple LinearMaps specified.");
        }

        auto tLinearMapInput = tLinearMapInputs[0];
        auto tLinearMapType = Plato::Get::String(tLinearMapInput, "Type");

        if(tLinearMapType == "SymmetryPlane")
        {
            return std::make_shared<Plato::Geometry::MeshMapDerived<SymmetryPlane<ScalarT>>>(aMesh, aInput);
        } else
        if(tLinearMapType == "Translation")
        {
            return std::make_shared<Plato::Geometry::MeshMapDerived<Translation<ScalarT>>>(aMesh, aInput);
        } else
        if(tLinearMapType == "")
        {
            return std::make_shared<Plato::Geometry::MeshMapDerived<Full<ScalarT>>>(aMesh, aInput);
        }
        else
        {
            throw Plato::ParsingException("MeshMap creation failed.  Unknown LinearMap Type requested.");
        }
    }
};
}  // end namespace Geometry
}  // end namespace Plato

#endif

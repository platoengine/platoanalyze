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


namespace Plato {
namespace Geometry {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = typename ExecSpace::memory_space;
using DeviceType = Kokkos::Device<ExecSpace, MemSpace>;

struct BoundingBoxes
{
  double *d_x0;
  double *d_y0;
  double *d_z0;
  double *d_x1;
  double *d_y1;
  double *d_z1;
  int N;
};

struct Spheres
{
  double *d_x;
  double *d_y;
  double *d_z;
  double *d_r;
  int N;
};

struct Points
{
  double *d_x;
  double *d_y;
  double *d_z;
  int N;
};

} // namespace Geometry
} // namespace Plato


namespace ArborX
{
namespace Traits
{
template <>
struct Access<Plato::Geometry::BoundingBoxes, PrimitivesTag>
{
  inline static std::size_t size(Plato::Geometry::BoundingBoxes const &boxes) { return boxes.N; }
  KOKKOS_INLINE_FUNCTION static Box get(Plato::Geometry::BoundingBoxes const &boxes, std::size_t i)
  {
    return {{boxes.d_x0[i], boxes.d_y0[i], boxes.d_z0[i]},
            {boxes.d_x1[i], boxes.d_y1[i], boxes.d_z1[i]}};
  }
  using memory_space = Plato::Geometry::MemSpace;
};

template <>
struct Access<Plato::Geometry::Points, PrimitivesTag>
{
  inline static std::size_t size(Plato::Geometry::Points const &points) { return points.N; }
  KOKKOS_INLINE_FUNCTION static Point get(Plato::Geometry::Points const &points, std::size_t i)
  {
    return {{points.d_x[i], points.d_y[i], points.d_z[i]}};
  }
  using memory_space = Plato::Geometry::MemSpace;
};

template <>
struct Access<Plato::Geometry::Spheres, PredicatesTag>
{
  inline static std::size_t size(Plato::Geometry::Spheres const &d) { return d.N; }
  KOKKOS_INLINE_FUNCTION static auto get(Plato::Geometry::Spheres const &d, std::size_t i)
  {
    return intersects(Sphere{{{d.d_x[i], d.d_y[i], d.d_z[i]}}, d.d_r[i]});
  }
  using memory_space = Plato::Geometry::MemSpace;
};

template <>
struct Access<Plato::Geometry::Points, PredicatesTag>
{
  inline static std::size_t size(Plato::Geometry::Points const &d) { return d.N; }
  KOKKOS_INLINE_FUNCTION static auto get(Plato::Geometry::Points const &d, std::size_t i)
  {
    return intersects(Point{d.d_x[i], d.d_y[i], d.d_z[i]});
  }
  using memory_space = Plato::Geometry::MemSpace;
};

} // namespace Traits
} // namespace ArborX


namespace Plato {
namespace Geometry {

enum Dim { X=0, Y, Z };
constexpr static size_t cSpaceDim = 3;
constexpr static size_t cNVertsPerElem = cSpaceDim+1;
constexpr static size_t cNFacesPerElem = cSpaceDim+1;


using OrdinalType = size_t;

template <typename ScalarType>
using VectorArrayT = Kokkos::View<ScalarType**, Kokkos::LayoutRight, MemSpace>;

template <typename ScalarType>
using ArrayT = Kokkos::View<ScalarType*, MemSpace>;

using OrdinalArray = ArrayT<OrdinalType>;

template <typename VectorArrayT>
struct MathMapBase
{
    using VectorArrayType = VectorArrayT;
};

template <typename VectorArrayType>
struct SymmetryPlane : public MathMapBase<VectorArrayType>
{

    using ScalarType = typename VectorArrayType::value_type;

    ScalarType mOrigin[cSpaceDim];
    ScalarType mNormal[cSpaceDim];

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
    }
    DEVICE_TYPE inline void
    operator()( OrdinalType aOrdinal, VectorArrayType aInValue, VectorArrayType aOutValue ) const
    {
        ScalarType tProjVal = 0.0;
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

template <typename VectorArrayType>
struct EnclosingElement
{
    using ScalarType = typename VectorArrayType::value_type;

    const Omega_h::LOs mCells2Nodes;
    const Omega_h::Reals mCoords;

    EnclosingElement(Omega_h::Mesh& aMesh) :
      mCells2Nodes(aMesh.ask_elem_verts()),
      mCoords(aMesh.coords()) {}

    DEVICE_TYPE inline bool
    operator()( OrdinalType aElemOrdinal, OrdinalType aNodeOrdinal, VectorArrayType aLocations ) const
    {
        bool tRetVal = true;

        for(OrdinalType iFace=0; iFace<cNFacesPerElem; ++iFace)
        {
            // get vertex indices
            OrdinalType i0 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+iFace];
            OrdinalType i1 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+(iFace+1)%cNFacesPerElem];
            OrdinalType i2 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+(iFace+2)%cNFacesPerElem];
            OrdinalType i3 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+(iFace+3)%cNFacesPerElem];

            // get vertex point values
            ScalarType X0=mCoords[i0*cSpaceDim+Dim::X], Y0=mCoords[i0*cSpaceDim+Dim::Y], Z0=mCoords[i0*cSpaceDim+Dim::Z];
            ScalarType X1=mCoords[i1*cSpaceDim+Dim::X], Y1=mCoords[i1*cSpaceDim+Dim::Y], Z1=mCoords[i1*cSpaceDim+Dim::Z];
            ScalarType X2=mCoords[i2*cSpaceDim+Dim::X], Y2=mCoords[i2*cSpaceDim+Dim::Y], Z2=mCoords[i2*cSpaceDim+Dim::Z];
            ScalarType X3=mCoords[i3*cSpaceDim+Dim::X], Y3=mCoords[i3*cSpaceDim+Dim::Y], Z3=mCoords[i3*cSpaceDim+Dim::Z];

            // get input point values
            ScalarType PX=aLocations(Dim::X,aNodeOrdinal),
                       PY=aLocations(Dim::Y,aNodeOrdinal),
                       PZ=aLocations(Dim::Z,aNodeOrdinal);

            ScalarType V1X=X1-X0, V1Y=Y1-Y0, V1Z=Z1-Z0; // V1 - V0
            ScalarType V2X=X2-X0, V2Y=Y2-Y0, V2Z=Z2-Z0; // V2 - V0
            ScalarType V3X=X3-X0, V3Y=Y3-Y0, V3Z=Z3-Z0; // V3 - V0
            ScalarType VPX=PX-X0, VPY=PY-Y0, VPZ=PZ-Z0; // VP - V0
            ScalarType VNX=V1Y*V2Z-V1Z*V2Y,
                       VNY=V1Z*V2X-V1X*V2Z,
                       VNZ=V1X*V2Y-V1Y*V2X;  // VN = V1 X V2
            ScalarType P3=V3X*VNX+V3Y*VNY+V3Z*VNZ; // V3 . VN: projection out of plane vertex onto plane normal
            ScalarType PP=VPX*VNX+VPY*VNY+VPZ*VNZ; // VP . VN: projection of input point vector onto plane normal
            ScalarType sgnP3 = (P3 >= 0.0) ? 1.0 : -1.0;
            ScalarType tol = 1e-4;
            tRetVal = (tRetVal && (P3*(PP+sgnP3*tol) > 0.0));  // if the projections are the same sign ...
        }
        return tRetVal;
    }
};

template <typename VectorArrayT>
struct GetBasis
{
    using ScalarType = typename VectorArrayT::value_type;
    using ScalarArrayT = ArrayT<ScalarType>;

    const Omega_h::LOs mCells2Nodes;
    const Omega_h::Reals mCoords;

    GetBasis(Omega_h::Mesh& aMesh) :
      mCells2Nodes(aMesh.ask_elem_verts()),
      mCoords(aMesh.coords()) {}

    DEVICE_TYPE inline void
    operator()(
      VectorArrayT aLocations,
      OrdinalType  aNodeOrdinal,
      int          aElemOrdinal,
      OrdinalType  aEntryOrdinal,
      OrdinalArray aColumnMap,
      ScalarArrayT aEntries) const
    {
        // get vertex indices
        OrdinalType i0 = mCells2Nodes[aElemOrdinal*cNVertsPerElem  ];
        OrdinalType i1 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+1];
        OrdinalType i2 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+2];
        OrdinalType i3 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+3];

        // get vertex point values
        ScalarType X0=mCoords[i0*cSpaceDim+Dim::X], Y0=mCoords[i0*cSpaceDim+Dim::Y], Z0=mCoords[i0*cSpaceDim+Dim::Z];
        ScalarType X1=mCoords[i1*cSpaceDim+Dim::X], Y1=mCoords[i1*cSpaceDim+Dim::Y], Z1=mCoords[i1*cSpaceDim+Dim::Z];
        ScalarType X2=mCoords[i2*cSpaceDim+Dim::X], Y2=mCoords[i2*cSpaceDim+Dim::Y], Z2=mCoords[i2*cSpaceDim+Dim::Z];
        ScalarType X3=mCoords[i3*cSpaceDim+Dim::X], Y3=mCoords[i3*cSpaceDim+Dim::Y], Z3=mCoords[i3*cSpaceDim+Dim::Z];

        ScalarType a11=X0-X3, a12=X1-X3, a13=X2-X3;
        ScalarType a21=Y0-Y3, a22=Y1-Y3, a23=Y2-Y3;
        ScalarType a31=Z0-Z3, a32=Z1-Z3, a33=Z2-Z3;

        ScalarType detA = a11*a22*a33+a12*a23*a31+a13*a21*a32-a11*a23*a32-a12*a21*a33-a13*a22*a31;

        ScalarType b11=(a22*a33-a23*a32)/detA, b12=(a13*a32-a12*a33)/detA, b13=(a12*a23-a13*a22)/detA;
        ScalarType b21=(a23*a31-a21*a33)/detA, b22=(a11*a33-a13*a31)/detA, b23=(a13*a21-a11*a23)/detA;
        ScalarType b31=(a21*a32-a22*a31)/detA, b32=(a12*a31-a11*a32)/detA, b33=(a11*a22-a12*a21)/detA;

        // get input point values
        ScalarType Xh=aLocations(Dim::X,aNodeOrdinal),
                   Yh=aLocations(Dim::Y,aNodeOrdinal),
                   Zh=aLocations(Dim::Z,aNodeOrdinal);

        ScalarType FX=Xh-X3, FY=Yh-Y3, FZ=Zh-Z3;

        ScalarType N0=b11*FX+b12*FY+b13*FZ;
        ScalarType N1=b21*FX+b22*FY+b23*FZ;
        ScalarType N2=b31*FX+b32*FY+b33*FZ;
        ScalarType N3=1.0-N0-N1-N2;

        aColumnMap(aEntryOrdinal  ) = i0;
        aColumnMap(aEntryOrdinal+1) = i1;
        aColumnMap(aEntryOrdinal+2) = i2;
        aColumnMap(aEntryOrdinal+3) = i3;

        aEntries(aEntryOrdinal  ) = N0;
        aEntries(aEntryOrdinal+1) = N1;
        aEntries(aEntryOrdinal+2) = N2;
        aEntries(aEntryOrdinal+3) = N3;
    }
};

template <typename ScalarType>
class AbstractMeshMap
{
  public:
    using ScalarArray = ArrayT<ScalarType>;

    virtual void apply(const ScalarArray & aInput, ScalarArray aOutput) = 0;
    virtual void applyT(const ScalarArray & aInput, ScalarArray aOutput) = 0;
};

template <typename MathMapType>
class MeshMap : public Plato::Geometry::AbstractMeshMap<typename MathMapType::VectorArrayType::value_type>
{
    using VectorArrayType = typename MathMapType::VectorArrayType;
    using ScalarType      = typename VectorArrayType::value_type;

  public:
    using ScalarArray = typename AbstractMeshMap<ScalarType>::ScalarArray;

    struct SparseMatrix {
        OrdinalArray mRowMap;
        OrdinalArray mColMap;
        ScalarArray mEntries;

        OrdinalType mNumRows;
        OrdinalType mNumCols;
    };


  private:
    MathMapType mMathMap;
    SparseMatrix mMatrix;
    std::shared_ptr<SparseMatrix> mFilter;
    bool mFilterFirst;

  public:
    MeshMap(Omega_h::Mesh& aMesh, Plato::InputData& aInput) :
      mMathMap(aInput.get<Plato::InputData>("LinearMap")),
      mFilter(nullptr),
      mFilterFirst(Plato::Get::Bool(aInput, "FilterFirst", /*default=*/ true))
    {
        // compute mapped values
        //
        auto tNVerts = aMesh.nverts();
        VectorArrayType tVertexLocations       ("mesh node locations",        cSpaceDim, tNVerts);
        VectorArrayType tMappedVertexLocations ("mapped mesh node locations", cSpaceDim, tNVerts);
        mapVertexLocations(aMesh, tVertexLocations, tMappedVertexLocations);


        // find elements that contain mapped locations
        //
        ArrayT<int> tParentElements("mapped mask", tNVerts);
        findParentElements(aMesh, tVertexLocations, tMappedVertexLocations, tParentElements);


        // conduct search and populate crs matrix
        //
        setMatrixValues(aMesh, tParentElements, tMappedVertexLocations, mMatrix);

        // build filter if requested
        //
        auto tFilterSpec = aInput.get_add<Plato::InputData>("Filter");
        mFilter = createFilter(tFilterSpec, tVertexLocations);
    }

    inline void
    createLinearFilter(ScalarType aRadius, SparseMatrix& aMatrix, VectorArrayType aLocations)
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

    inline void
    setLinearFilterMatrixValues(
      ScalarType aRadius,
      SparseMatrix& aMatrix,
      VectorArrayType aLocations,
      Kokkos::View<int*, DeviceType> aIndices,
      Kokkos::View<int*, DeviceType> aOffset)
    {
        aMatrix.mNumRows = aLocations.extent(1);
        aMatrix.mNumCols = aLocations.extent(1);

        // determine rowmap
        auto tNumRows = aMatrix.mNumRows;
        OrdinalArray tRowMap("row map", tNumRows+1);
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalType>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalType iRowOrdinal)
        {
            tRowMap(iRowOrdinal) = aOffset(iRowOrdinal+1) - aOffset(iRowOrdinal);
        }, "nonzeros");

        OrdinalType tNumEntries(0);
        Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalType>(0,tNumRows+1),
        KOKKOS_LAMBDA (const OrdinalType& iOrdinal, OrdinalType& aUpdate, const bool& tIsFinal)
        {
            const OrdinalType tVal = tRowMap(iOrdinal);
            if( tIsFinal )
            {
              tRowMap(iOrdinal) = aUpdate;
            }
            aUpdate += tVal;
        }, tNumEntries);
        aMatrix.mRowMap = tRowMap;

        // determine column map and entries
        auto tRadius = aRadius;
        OrdinalArray tColMap("row map", tNumEntries);
        ScalarArray tEntries("entries", tNumEntries);
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalType>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalType iRowOrdinal)
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

    inline std::shared_ptr<SparseMatrix>
    createFilter(Plato::InputData aFilterSpec, VectorArrayType aLocations)
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

    void
    findParentElements(
      Omega_h::Mesh& aMesh,
      VectorArrayType aLocations,
      VectorArrayType aMappedLocations,
      ArrayT<int> aParentElements)
    {
        auto tNElems = aMesh.nelems();
        VectorArrayType tMin("min", cSpaceDim, tNElems);
        VectorArrayType tMax("max", cSpaceDim, tNElems);

        ScalarType tol = 1e-4;

        // fill d_* data
        auto tCoords = aMesh.coords();
        Omega_h::LOs tCells2Nodes = aMesh.ask_elem_verts();
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalType>(0, tNElems), LAMBDA_EXPRESSION(OrdinalType iCellOrdinal)
        {
            OrdinalType tNVertsPerElem = cSpaceDim+1;

            // set min and max of element bounding box to first node
            for(size_t iDim=0; iDim<cSpaceDim; ++iDim)
            {
                OrdinalType tVertIndex = tCells2Nodes[iCellOrdinal*tNVertsPerElem];
                tMin(iDim, iCellOrdinal) = tCoords[tVertIndex*cSpaceDim+iDim];
                tMax(iDim, iCellOrdinal) = tCoords[tVertIndex*cSpaceDim+iDim];
            }
            // loop on remaining nodes to find min
            for(OrdinalType iVert=1; iVert<tNVertsPerElem; ++iVert)
            {
                OrdinalType tVertIndex = tCells2Nodes[iCellOrdinal*tNVertsPerElem + iVert];
                for(size_t iDim=0; iDim<cSpaceDim; ++iDim)
                {
                    if( tMin(iDim, iCellOrdinal) > tCoords[tVertIndex*cSpaceDim+iDim] )
                    {
                        tMin(iDim, iCellOrdinal) = tCoords[tVertIndex*cSpaceDim+iDim];
                    }
                    else
                    if( tMax(iDim, iCellOrdinal) < tCoords[tVertIndex*cSpaceDim+iDim] )
                    {
                        tMax(iDim, iCellOrdinal) = tCoords[tVertIndex*cSpaceDim+iDim];
                    }
                }
            }
            for(size_t iDim=0; iDim<cSpaceDim; ++iDim)
            {
                tMax(iDim, iCellOrdinal) += tol;
                tMin(iDim, iCellOrdinal) -= tol;
            }
        }, "element bounding boxes");


        auto d_x0 = Kokkos::subview(tMin, (size_t)Dim::X, Kokkos::ALL());
        auto d_y0 = Kokkos::subview(tMin, (size_t)Dim::Y, Kokkos::ALL());
        auto d_z0 = Kokkos::subview(tMin, (size_t)Dim::Z, Kokkos::ALL());
        auto d_x1 = Kokkos::subview(tMax, (size_t)Dim::X, Kokkos::ALL());
        auto d_y1 = Kokkos::subview(tMax, (size_t)Dim::Y, Kokkos::ALL());
        auto d_z1 = Kokkos::subview(tMax, (size_t)Dim::Z, Kokkos::ALL());

        // construct search tree
        ArborX::BVH<DeviceType>
          bvh{BoundingBoxes{d_x0.data(), d_y0.data(), d_z0.data(),
                            d_x1.data(), d_y1.data(), d_z1.data(), tNElems}};

        // conduct search for bounding box elements
        auto d_x = Kokkos::subview(aMappedLocations, (size_t)Dim::X, Kokkos::ALL());
        auto d_y = Kokkos::subview(aMappedLocations, (size_t)Dim::Y, Kokkos::ALL());
        auto d_z = Kokkos::subview(aMappedLocations, (size_t)Dim::Z, Kokkos::ALL());

        auto tNVerts = aMesh.nverts();
        Kokkos::View<int*, DeviceType> tIndices("indices", 0), tOffset("offset", 0);
        bvh.query(Points{d_x.data(), d_y.data(), d_z.data(), tNVerts}, tIndices, tOffset);

        // loop over indices and find containing element
        EnclosingElement<VectorArrayType> tEnclosingElement(aMesh);
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalType>(0, tNVerts), LAMBDA_EXPRESSION(OrdinalType iNodeOrdinal)
        {
            aParentElements(iNodeOrdinal) = -1;
            if( aLocations(Dim::X, iNodeOrdinal) != aMappedLocations(Dim::X, iNodeOrdinal) ||
                aLocations(Dim::Y, iNodeOrdinal) != aMappedLocations(Dim::Y, iNodeOrdinal) ||
                aLocations(Dim::Z, iNodeOrdinal) != aMappedLocations(Dim::Z, iNodeOrdinal) )
            {
                aParentElements(iNodeOrdinal) = -2;
                for( int iElem=tOffset(iNodeOrdinal); iElem<tOffset(iNodeOrdinal+1); iElem++ )
                {
                    auto tElem = tIndices(iElem);
                    if( tEnclosingElement(tElem, iNodeOrdinal, aMappedLocations) )
                    {
                        aParentElements(iNodeOrdinal) = tElem;
                        break;
                    }
                }
            }
        }, "find parent element");
    }

    void setMatrixValues(Omega_h::Mesh& aMesh, ArrayT<int> aParentElements, VectorArrayType aLocation, SparseMatrix& aMatrix)
    {
        auto tNVerts = aMesh.nverts();
        aMatrix.mNumRows = tNVerts;
        aMatrix.mNumCols = tNVerts;

        // determine rowmap
        auto tNumRows = aMatrix.mNumRows;
        OrdinalArray tRowMap("row map", tNumRows+1);
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalType>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalType iRowOrdinal)
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

        OrdinalType tNumEntries(0);
        Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalType>(0,tNumRows+1),
        KOKKOS_LAMBDA (const OrdinalType& iOrdinal, OrdinalType& aUpdate, const bool& tIsFinal)
        {
            const OrdinalType tVal = tRowMap(iOrdinal);
            if( tIsFinal )
            {
              tRowMap(iOrdinal) = aUpdate;
            }
            aUpdate += tVal;
        }, tNumEntries);
        aMatrix.mRowMap = tRowMap;

        // determine column map and entries
        OrdinalArray tColMap("row map", tNumEntries);
        ScalarArray tEntries("entries", tNumEntries);
        GetBasis<VectorArrayType> tGetBasis(aMesh);
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalType>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalType iRowOrdinal)
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

    SparseMatrix createMatrix(OrdinalArray aMask)
    {

        // determine nrows and ncolumns
        //
        OrdinalType tNumRows = aMask.extent(0);
        OrdinalType tNumCols = 0;
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalType aOrdinal, OrdinalType & aNumNonzeros)
        {
            if( aMask(aOrdinal) > 0 )
            {
                aNumNonzeros++;
            }
        }, tNumCols);

        SparseMatrix tMatrix;
        tMatrix.mNumRows = tNumRows;
        tMatrix.mNumCols = tNumCols;

        return tMatrix;
    }

    void createMask(Omega_h::Mesh& aMesh, ArrayT<int> aParentElements, OrdinalArray aMask)
    {
        auto tNVerts = aMesh.nverts();
        typename OrdinalArray::value_type tFlag(1);
        Omega_h::LOs tCells2Nodes = aMesh.ask_elem_verts();
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalType>(0, tNVerts), LAMBDA_EXPRESSION(OrdinalType iVertOrdinal)
        {
            OrdinalType tNVerts = cSpaceDim+1;

            auto tParentElement = aParentElements(iVertOrdinal);

            if( tParentElement >= 0 )
            {
                for(OrdinalType iElemVert=0; iElemVert<tNVerts; ++iElemVert)
                {
                    OrdinalType tVertIndex = tCells2Nodes[tParentElement*tNVerts + iElemVert];
                    Kokkos::atomic_assign(&aMask(tVertIndex), tFlag);
                }
            }
        }, "get mask");
    }

    void mapVertexLocations(Omega_h::Mesh& aMesh, VectorArrayType aLocations, VectorArrayType aMappedLocations)
    {
        auto tCoords = aMesh.coords();
        auto tNVerts = aMesh.nverts();
        auto tMathMap = mMathMap;
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalType>(0, tNVerts), LAMBDA_EXPRESSION(OrdinalType iOrdinal)
        {
            for(size_t iDim=0; iDim<cSpaceDim; ++iDim)
            {
                aLocations(iDim, iOrdinal) = tCoords[iOrdinal*cSpaceDim+iDim];
            }
            tMathMap(iOrdinal, aLocations, aMappedLocations);
        }, "get verts and apply map");
    }

    ~MeshMap()
    {
    }

    /***************************************************************************//**
    * @brief Get number of columns in map matrix
    *******************************************************************************/
    OrdinalType numColumns();

    /***************************************************************************//**
    * @brief Get number of rows in map matrix
    *******************************************************************************/
    OrdinalType numRows();

    /***************************************************************************//**
    *
    * @brief Apply mapping
    *
    *******************************************************************************/
    void apply(const ScalarArray & aInput, ScalarArray aOutput)
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
    *
    * @brief Matrix times vector
    *
    *******************************************************************************/
    void matvec(const SparseMatrix & aMatrix, const ScalarArray & aInput)
    {
        ScalarArray tOutput("output vector", aInput.extent(0));
        Kokkos::deep_copy(tOutput, aInput);
        matvec(aMatrix, aInput, tOutput);
        Kokkos::deep_copy(aInput, tOutput);
    }
    void matvec(const SparseMatrix & aMatrix, const ScalarArray & aInput, ScalarArray aOutput)
    {
        auto tRowMap = aMatrix.mRowMap;
        auto tColMap = aMatrix.mColMap;
        auto tEntries = aMatrix.mEntries;
        auto tNumRows = tRowMap.size() - 1;

        using ScalarT = typename VectorArrayType::value_type;

        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalType>(0, tNumRows), LAMBDA_EXPRESSION(OrdinalType aRowOrdinal)
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
    *
    * @brief Apply transpose of mapping
    *
    *******************************************************************************/
    void applyT(const ScalarArray & aInput, ScalarArray aOutput)
    {
        return;
    }

}; // end class MeshMap



template <typename ScalarType = double>
struct MeshMapFactory
{
    using VectorArrayType = VectorArrayT<ScalarType>;

    inline std::shared_ptr<Plato::Geometry::AbstractMeshMap<ScalarType>>
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

    inline std::shared_ptr<Plato::Geometry::AbstractMeshMap<ScalarType>>
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
            return std::make_shared<Plato::Geometry::MeshMap<SymmetryPlane<VectorArrayType>>>(aMesh, aInput);
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

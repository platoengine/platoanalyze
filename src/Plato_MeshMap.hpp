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
* @brief Functor that computes position in local coordinates of a point given
         in global coordinates then returns the basis values at that local
         point.

  The local position is computed as follows.  Given:
  \f{eqnarray*}{
    \bar{x}^h(\xi) = N_I(\xi) \bar{x}_I \\
    N_I = \left\{\begin{array}{cccc}
              x_l & y_l & z_l & 1-x_l-y_l-z_l
           \end{array}\right\}^T
  \f}
  Find: \f$ x_l \f$, \f$ y_l \f$, and \f$ z_l \f$.

  Simplifying the above yields:
  \f[
    \left[\begin{array}{ccc}
      x_1-x_4 & x_2-x_4 & x_3-x_4 \\
      y_1-y_4 & y_2-y_4 & y_3-y_4 \\
      z_1-z_4 & z_2-z_4 & z_3-z_4 \\
    \end{array}\right]
    \left\{\begin{array}{c}
      x_l \\ y_l \\ z_l
    \end{array}\right\} =
    \left\{\begin{array}{c}
      x^h-x_4 \\ y^h-y_4 \\ z^h-z_4
    \end{array}\right\}
  \f]
  Below directly solves the linear system above for \f$x_l\f$, \f$ y_l \f$, and
  \f$ z_l \f$ then evaluates the basis.
*******************************************************************************/
template <typename ScalarT>
struct GetBasis
{
    using ScalarArrayT  = typename Plato::ScalarVectorT<ScalarT>;
    using VectorArrayT  = typename Plato::ScalarMultiVectorT<ScalarT>;
    using OrdinalT      = typename ScalarArrayT::size_type;
    using OrdinalArrayT = typename Plato::ScalarVectorT<OrdinalT>;

    const Omega_h::LOs mCells2Nodes;
    const Omega_h::Reals mCoords;

    GetBasis(Omega_h::Mesh& aMesh) :
      mCells2Nodes(aMesh.ask_elem_verts()),
      mCoords(aMesh.coords()) {}

    /******************************************************************************//**
     * @brief Find local coordinates from global coordinates, compute basis values
     * @param [in]  Zh, Yh, Zh position in global coordinates
     * @param [in]  i0, i1, i2, i3 global indices of nodes comprised by the element
     * @param [out] b0, b1, b2, b3 basis values
    **********************************************************************************/
    DEVICE_TYPE inline void
    basis(
      ScalarT  Xh, ScalarT  Yh, ScalarT  Zh,
      OrdinalT i0, OrdinalT i1, OrdinalT i2, OrdinalT i3,
      ScalarT& b0, ScalarT& b1, ScalarT& b2, ScalarT& b3) const
    {
        // get vertex point values
        ScalarT X0=mCoords[i0*cSpaceDim+Dim::X], Y0=mCoords[i0*cSpaceDim+Dim::Y], Z0=mCoords[i0*cSpaceDim+Dim::Z];
        ScalarT X1=mCoords[i1*cSpaceDim+Dim::X], Y1=mCoords[i1*cSpaceDim+Dim::Y], Z1=mCoords[i1*cSpaceDim+Dim::Z];
        ScalarT X2=mCoords[i2*cSpaceDim+Dim::X], Y2=mCoords[i2*cSpaceDim+Dim::Y], Z2=mCoords[i2*cSpaceDim+Dim::Z];
        ScalarT X3=mCoords[i3*cSpaceDim+Dim::X], Y3=mCoords[i3*cSpaceDim+Dim::Y], Z3=mCoords[i3*cSpaceDim+Dim::Z];

        ScalarT a11=X0-X3, a12=X1-X3, a13=X2-X3;
        ScalarT a21=Y0-Y3, a22=Y1-Y3, a23=Y2-Y3;
        ScalarT a31=Z0-Z3, a32=Z1-Z3, a33=Z2-Z3;

        ScalarT detA = a11*a22*a33+a12*a23*a31+a13*a21*a32-a11*a23*a32-a12*a21*a33-a13*a22*a31;

        ScalarT b11=(a22*a33-a23*a32)/detA, b12=(a13*a32-a12*a33)/detA, b13=(a12*a23-a13*a22)/detA;
        ScalarT b21=(a23*a31-a21*a33)/detA, b22=(a11*a33-a13*a31)/detA, b23=(a13*a21-a11*a23)/detA;
        ScalarT b31=(a21*a32-a22*a31)/detA, b32=(a12*a31-a11*a32)/detA, b33=(a11*a22-a12*a21)/detA;

        ScalarT FX=Xh-X3, FY=Yh-Y3, FZ=Zh-Z3;

        b0=b11*FX+b12*FY+b13*FZ;
        b1=b21*FX+b22*FY+b23*FZ;
        b2=b31*FX+b32*FY+b33*FZ;
        b3=1.0-b0-b1-b2;
    }


    /******************************************************************************//**
     * @brief Find local coordinates from global coordinates, compute basis values, and
              assembles them into the columnMap and entries of a sparse matrix.
     * @param [in]  aLocations view of points (D, N)
     * @param [in]  aNodeOrdinal index of point for which to determine local coords
     * @param [in]  aElemOrdinal index of element whose bases will be used for interpolation
     * @param [in]  aEntryOrdinal index into aColumnMap and aEntries
     * @param [out] aColumnMap of the sparse matrix
     * @param [out] aEntries of the sparse matrix
    **********************************************************************************/
    DEVICE_TYPE inline void
    operator()(
      VectorArrayT  aLocations,
      OrdinalT      aNodeOrdinal,
      int           aElemOrdinal,
      OrdinalT      aEntryOrdinal,
      OrdinalArrayT aColumnMap,
      ScalarArrayT  aEntries) const
    {
        // get input point values
        ScalarT Xh=aLocations(Dim::X,aNodeOrdinal),
                Yh=aLocations(Dim::Y,aNodeOrdinal),
                Zh=aLocations(Dim::Z,aNodeOrdinal);

        // get vertex indices
        OrdinalT i0 = mCells2Nodes[aElemOrdinal*cNVertsPerElem  ];
        OrdinalT i1 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+1];
        OrdinalT i2 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+2];
        OrdinalT i3 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+3];

        ScalarT b0, b1, b2, b3;

        basis(Xh, Yh, Zh,
              i0, i1, i2, i3,
              b0, b1, b2, b3);

        aColumnMap(aEntryOrdinal  ) = i0;
        aColumnMap(aEntryOrdinal+1) = i1;
        aColumnMap(aEntryOrdinal+2) = i2;
        aColumnMap(aEntryOrdinal+3) = i3;

        aEntries(aEntryOrdinal  ) = b0;
        aEntries(aEntryOrdinal+1) = b1;
        aEntries(aEntryOrdinal+2) = b2;
        aEntries(aEntryOrdinal+3) = b3;
    }

    /******************************************************************************//**
     * @brief Find local coordinates from global coordinates and compute basis values
     * @param [in]  aLocations view of points (D, N)
     * @param [in]  aNodeOrdinal index of point for which to determine local coords
     * @param [in]  aElemOrdinal index of element whose bases will be used for interpolation
     * @param [out] aBases basis values
    **********************************************************************************/
    DEVICE_TYPE inline void
    operator()(
      VectorArrayT  aLocations,
      OrdinalT      aNodeOrdinal,
      int           aElemOrdinal,
      ScalarT       aBases[cNVertsPerElem]) const
    {
        // get input point values
        ScalarT Xh=aLocations(Dim::X,aNodeOrdinal),
                Yh=aLocations(Dim::Y,aNodeOrdinal),
                Zh=aLocations(Dim::Z,aNodeOrdinal);

        // get vertex indices
        OrdinalT i0 = mCells2Nodes[aElemOrdinal*cNVertsPerElem  ];
        OrdinalT i1 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+1];
        OrdinalT i2 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+2];
        OrdinalT i3 = mCells2Nodes[aElemOrdinal*cNVertsPerElem+3];

        ScalarT b0, b1, b2, b3;

        basis(Xh, Yh, Zh,
              i0, i1, i2, i3,
              b0, b1, b2, b3);

        aBases[0] = b0;
        aBases[1] = b1;
        aBases[2] = b2;
        aBases[3] = b3;
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
    * @brief Find element that contains each mapped node
     * @param [in]  aLocations location of mesh nodes
     * @param [in]  aMappedLocations mapped location of mesh nodes
     * @param [out] aParentElements if node is mapped, index of parent element.

       If a node is mapped (i.e., aLocations(*,node_id)!=aMappedLocations(*,node_id))
       and the parent element is found, aParentElements(node_id) is set to the index
       of the parent element.
       If a node is mapped but the parent element isn't found, aParentElements(node_id)
       is set to -2.
       If a node is not mapped, aParentElements(node_id) is set to -1.
    *******************************************************************************/
    void
    findParentElements(
      Omega_h::Mesh& aMesh,
      VectorArrayT aLocations,
      VectorArrayT aMappedLocations,
      IntegerArrayT aParentElements)
    {
        auto tNElems = aMesh.nelems();
        VectorArrayT tMin("min", cSpaceDim, tNElems);
        VectorArrayT tMax("max", cSpaceDim, tNElems);

        constexpr ScalarT cRelativeTol = 1e-2;

        // fill d_* data
        auto tCoords = aMesh.coords();
        Omega_h::LOs tCells2Nodes = aMesh.ask_elem_verts();
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNElems), LAMBDA_EXPRESSION(OrdinalT iCellOrdinal)
        {
            OrdinalT tNVertsPerElem = cSpaceDim+1;

            // set min and max of element bounding box to first node
            for(size_t iDim=0; iDim<cSpaceDim; ++iDim)
            {
                OrdinalT tVertIndex = tCells2Nodes[iCellOrdinal*tNVertsPerElem];
                tMin(iDim, iCellOrdinal) = tCoords[tVertIndex*cSpaceDim+iDim];
                tMax(iDim, iCellOrdinal) = tCoords[tVertIndex*cSpaceDim+iDim];
            }
            // loop on remaining nodes to find min
            for(OrdinalT iVert=1; iVert<tNVertsPerElem; ++iVert)
            {
                OrdinalT tVertIndex = tCells2Nodes[iCellOrdinal*tNVertsPerElem + iVert];
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
                ScalarT tLen = tMax(iDim, iCellOrdinal) - tMin(iDim, iCellOrdinal);
                tMax(iDim, iCellOrdinal) += cRelativeTol * tLen;
                tMin(iDim, iCellOrdinal) -= cRelativeTol * tLen;
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
        GetBasis<ScalarT> tGetBasis(aMesh);
        Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNVerts), LAMBDA_EXPRESSION(OrdinalT iNodeOrdinal)
        {
            ScalarT tBasis[cNVertsPerElem];
            aParentElements(iNodeOrdinal) = -1;
            if( aLocations(Dim::X, iNodeOrdinal) != aMappedLocations(Dim::X, iNodeOrdinal) ||
                aLocations(Dim::Y, iNodeOrdinal) != aMappedLocations(Dim::Y, iNodeOrdinal) ||
                aLocations(Dim::Z, iNodeOrdinal) != aMappedLocations(Dim::Z, iNodeOrdinal) )
            {
                aParentElements(iNodeOrdinal) = -2;
                constexpr ScalarT cNotFound = -1e8; // big negative number ensures max min is found
                ScalarT tMaxMin = cNotFound;
                typename IntegerArrayT::value_type iParent = -2;
                for( int iElem=tOffset(iNodeOrdinal); iElem<tOffset(iNodeOrdinal+1); iElem++ )
                {
                    auto tElem = tIndices(iElem);
                    tGetBasis(aMappedLocations, iNodeOrdinal, tElem, tBasis);
                    ScalarT tMin = tBasis[0];
                    for(OrdinalT iB=1; iB<cNVertsPerElem; iB++)
                    {
                        if( tBasis[iB] < tMin ) tMin = tBasis[iB];
                    }
                    if( tMin > cNotFound )
                    {
                         tMaxMin = tMin;
                         iParent = tElem;
                    }
                }
                if( tMaxMin > cNotFound )
                {
                    aParentElements(iNodeOrdinal) = iParent;
                }
            }
        }, "find parent element");
    }

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
        MapBase::findParentElements(aMesh, tVertexLocations, tMappedVertexLocations, tParentElements);


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


/*!
 * Plato_MeshMapUtils.hpp
 *
 * Created on: Oct 1, 2020
 *
 */

#ifndef PLATO_MESHMAP_UTILS_HPP_
#define PLATO_MESHMAP_UTILS_HPP_

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

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
template <typename ScalarT>
void
findParentElements(
  Omega_h::Mesh& aMesh,
  Plato::ScalarMultiVectorT<ScalarT> aLocations,
  Plato::ScalarMultiVectorT<ScalarT> aMappedLocations,
  Plato::ScalarVectorT<int> aParentElements)
{
    using OrdinalT      = typename Plato::ScalarVectorT<ScalarT>::size_type;

    auto tNElems = aMesh.nelems();
    Plato::ScalarMultiVectorT<ScalarT> tMin("min", cSpaceDim, tNElems);
    Plato::ScalarMultiVectorT<ScalarT> tMax("max", cSpaceDim, tNElems);

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

    auto tNumLocations = aParentElements.size();
    Kokkos::View<int*, DeviceType> tIndices("indices", 0), tOffset("offset", 0);
    bvh.query(Points{d_x.data(), d_y.data(), d_z.data(), static_cast<int>(tNumLocations)}, tIndices, tOffset);

    // loop over indices and find containing element
    GetBasis<ScalarT> tGetBasis(aMesh);
    Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalT>(0, tNumLocations), LAMBDA_EXPRESSION(OrdinalT iNodeOrdinal)
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
            typename Plato::ScalarVectorT<int>::value_type iParent = -2;
            for( int iElem=tOffset(iNodeOrdinal); iElem<tOffset(iNodeOrdinal+1); iElem++ )
            {
                auto tElem = tIndices(iElem);
                tGetBasis(aMappedLocations, iNodeOrdinal, tElem, tBasis);
                ScalarT tEleMin = tBasis[0];
                for(OrdinalT iB=1; iB<cNVertsPerElem; iB++)
                {
                    if( tBasis[iB] < tEleMin ) tEleMin = tBasis[iB];
                }
                if( tEleMin > tMaxMin )
                {
                     tMaxMin = tEleMin;
                     iParent = tElem;
                }
            }
            if( tMaxMin >= 0.0 )
            {
                aParentElements(iNodeOrdinal) = iParent;
            }
        }
    }, "find parent element");
}

}  // end namespace Geometry
}  // end namespace Plato

#endif

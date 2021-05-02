#pragma once

#include "Simplex.hpp"
#include "OmegaHUtilities.hpp"
#include "geometric/WorksetBase.hpp"
#include "SurfaceIntegralUtilities.hpp"

#include "PlatoStaticsTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "geometric/GeometricSimplexFadTypes.hpp"
#include "geometric/AbstractScalarFunction.hpp"

#include "ExpInstMacros.hpp"

#include <fstream>
#include <ArborX.hpp>

namespace Plato
{
namespace GMF
{
    struct Points
    {
      double *d_x;
      double *d_y;
      double *d_z;
      int N;
    };
} // end namespace GMF
} // end namespace Plato

namespace Plato
{
namespace Geometric
{
enum Dim { X=0, Y, Z };

//using ExecSpace = Kokkos::DefaultExecutionSpace;
//using MemSpace = typename ExecSpace::memory_space;
//using DeviceType = Kokkos::Device<ExecSpace, MemSpace>;

} // end namespace Geometric
} // end namespace Plato


namespace ArborX
{
namespace Traits
{
template <>
struct Access<Plato::GMF::Points, PrimitivesTag>
{
  inline static std::size_t size(Plato::GMF::Points const &points) { return points.N; }
  KOKKOS_INLINE_FUNCTION static Point get(Plato::GMF::Points const &points, std::size_t i)
  {
    return {{points.d_x[i], points.d_y[i], points.d_z[i]}};
  }
  using memory_space = Plato::MemSpace;
};
template <>
struct Access<Plato::GMF::Points, PredicatesTag>
{
  inline static std::size_t size(Plato::GMF::Points const &d) { return d.N; }
  KOKKOS_INLINE_FUNCTION static auto get(Plato::GMF::Points const &d, std::size_t i)
  {
    return nearest(Point{d.d_x[i], d.d_y[i], d.d_z[i]}, 1);
  }
  using memory_space = Plato::MemSpace;
};
} // end namespace Traits
} // end namespace ArborX

namespace Plato
{

namespace Geometric
{


/******************************************************************************/
template<typename EvaluationType>
class GeometryMisfit : public Plato::Geometric::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    
    using Plato::Geometric::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Geometric::AbstractScalarFunction<EvaluationType>::mDataMap;

    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    std::string mPointCloudName;
    std::string mPointCloudRowMapName;
    std::string mPointCloudColMapName;
    std::string mSideSetName;

  public:
    /**************************************************************************/
    GeometryMisfit(
        const Plato::SpatialDomain   & aSpatialDomain, 
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aFunctionParams, 
              std::string            & aFunctionName
    ) :
        Plato::Geometric::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aFunctionName),
        mPointCloudName("")
    /**************************************************************************/
    {
        this->mHasBoundaryTerm = true;
        auto aCriterionParams = aFunctionParams.sublist("Criteria").sublist("Geometry Misfit");
        mSideSetName = aCriterionParams.get<std::string>("Sides");

        parsePointCloud(aCriterionParams);

        createPointGraph(aSpatialDomain.Mesh, aSpatialDomain.MeshSets);
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <ResultScalarType > & aResult
    ) const override
    /**************************************************************************/
    { /* No-op.  Misfit is purely a boundary quantity */ }

    /**************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                          & aModel,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT    <ConfigScalarType > & aConfig,
              Plato::ScalarVectorT     <ResultScalarType > & aResult
    ) const override
    /**************************************************************************/
    {

        // load the sideset specified in the input
        auto tFaceLids = Plato::get_face_ordinals(aModel.MeshSets, mSideSetName);
        auto tFace2Verts = aModel.Mesh.ask_verts_of(SpaceDim-1);
        auto tCell2Verts = aModel.Mesh.ask_elem_verts();

        auto tFace2eElems = aModel.Mesh.ask_up(SpaceDim - 1, SpaceDim);
        auto tFace2Elems_map   = tFace2eElems.a2ab;
        auto tFace2Elems_elems = tFace2eElems.ab2b;

        auto tOffsets = mDataMap.ordinalVectors[mPointCloudRowMapName];
        auto tIndices = mDataMap.ordinalVectors[mPointCloudColMapName];

        auto tPoints = mDataMap.scalarMultiVectors[mPointCloudName];

        // create functors
        Plato::CalculateSurfaceArea<SpaceDim> tCalculateSurfaceArea;
        Plato::CalculateSurfaceJacobians<SpaceDim> tCalculateSurfaceJacobians;
        Plato::CreateFaceLocalNode2ElemLocalNodeIndexMap<SpaceDim> tCreateFaceLocalNode2ElemLocalNodeIndexMap;

        auto tNumFaces = tFaceLids.size();
        Plato::ScalarVector tResultDenominator("denominator", tNumFaces);

        // for each point in the cloud, compute the square of the average normal distance to the nearest face
        auto tNumPoints = tOffsets.extent(0) - 1;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumPoints), LAMBDA_EXPRESSION(const Plato::OrdinalType & aPointOrdinal)
        {
            // get face index
            auto tLocalFaceOrdinal = tIndices(tOffsets(aPointOrdinal));
            auto tFaceOrdinal = tFaceLids[tLocalFaceOrdinal];

            // get elem index
            auto tCellOffset  = tFace2Elems_map[tFaceOrdinal];
            auto tCellOrdinal = tFace2Elems_elems[tCellOffset];

            // compute the map from local node id to global
            Plato::OrdinalType tLocalNodeOrd[SpaceDim];
            tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

            // get face centroid TODO
            ConfigScalarType C_x = aConfig(tCellOrdinal, tLocalNodeOrd[0], Dim::X);
            ConfigScalarType C_y = aConfig(tCellOrdinal, tLocalNodeOrd[0], Dim::Y);
            ConfigScalarType C_z = aConfig(tCellOrdinal, tLocalNodeOrd[0], Dim::Z);
            for (Plato::OrdinalType tFaceVertI=1; tFaceVertI<SpaceDim; tFaceVertI++)
            {
                C_x += aConfig(tCellOrdinal, tLocalNodeOrd[tFaceVertI], Dim::X);
                C_y += aConfig(tCellOrdinal, tLocalNodeOrd[tFaceVertI], Dim::Y);
                C_z += aConfig(tCellOrdinal, tLocalNodeOrd[tFaceVertI], Dim::Z);
            }
            constexpr Plato::OrdinalType cNumVertsPerFace = SpaceDim;
            C_x /= cNumVertsPerFace;
            C_y /= cNumVertsPerFace;
            C_z /= cNumVertsPerFace;

            // get vertex 0 coordinates
            ConfigScalarType a_x = aConfig(tCellOrdinal, tLocalNodeOrd[0], Dim::X);
            ConfigScalarType a_y = aConfig(tCellOrdinal, tLocalNodeOrd[0], Dim::Y);
            ConfigScalarType a_z = aConfig(tCellOrdinal, tLocalNodeOrd[0], Dim::Z);

            // get vertex 1 coordinates
            ConfigScalarType b_x = aConfig(tCellOrdinal, tLocalNodeOrd[1], Dim::X);
            ConfigScalarType b_y = aConfig(tCellOrdinal, tLocalNodeOrd[1], Dim::Y);
            ConfigScalarType b_z = aConfig(tCellOrdinal, tLocalNodeOrd[1], Dim::Z);

            // vector, A, from centroid to vertex 0
            ConfigScalarType A_x = a_x - C_x;
            ConfigScalarType A_y = a_y - C_y;
            ConfigScalarType A_z = a_z - C_z;

            // vector, B, from centroid to vertex 1
            ConfigScalarType B_x = b_x - C_x;
            ConfigScalarType B_y = b_y - C_y;
            ConfigScalarType B_z = b_z - C_z;

            // unit normal vector, n = A X B / |A X B|
            ConfigScalarType n_x = A_y*B_z - A_z*B_y;
            ConfigScalarType n_y = A_z*B_x - A_x*B_z;
            ConfigScalarType n_z = A_x*B_y - A_y*B_x;
            ConfigScalarType mag_n = n_x*n_x + n_y*n_y + n_z*n_z;
            mag_n = sqrt(mag_n);
            n_x /= mag_n;
            n_y /= mag_n;
            n_z /= mag_n;

            // vector, P, from centroid to point
            ConfigScalarType P_x = tPoints(Dim::X, aPointOrdinal) - C_x;
            ConfigScalarType P_y = tPoints(Dim::Y, aPointOrdinal) - C_y;
            ConfigScalarType P_z = tPoints(Dim::Z, aPointOrdinal) - C_z;

            // normal distance to point
            ResultScalarType tDistance = P_x*n_x + P_y*n_y + P_z*n_z;

            // add to result
            Kokkos::atomic_add(&aResult(tCellOrdinal), tDistance);
            Kokkos::atomic_add(&tResultDenominator(tLocalFaceOrdinal), 1.0);

        }, "compute misfit");

        // it appears that the ComputeIntegralWeight() function is missing a 1/2 factor.
        Plato::Scalar tMultiplier = 1.0/2.0;

        Plato::ScalarArray3DT<ConfigScalarType> tJacobian("jacobian", tNumFaces, SpaceDim-1, SpaceDim);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceOrdinal)
        {
            // get global face index from local index
            auto tFaceOrdinal = tFaceLids[aFaceOrdinal];

            // get elem index
            auto tCellOffset  = tFace2Elems_map[tFaceOrdinal];
            auto tCellOrdinal = tFace2Elems_elems[tCellOffset];

            // compute the map from local node id to global
            Plato::OrdinalType tLocalNodeOrd[SpaceDim];
            tCreateFaceLocalNode2ElemLocalNodeIndexMap(tCellOrdinal, tFaceOrdinal, tCell2Verts, tFace2Verts, tLocalNodeOrd);

            // compute integration weights
            ConfigScalarType tWeight(0.0);
            tCalculateSurfaceJacobians(tCellOrdinal, aFaceOrdinal, tLocalNodeOrd, aConfig, tJacobian);
            tCalculateSurfaceArea(aFaceOrdinal, tMultiplier, tJacobian, tWeight);

            if (tResultDenominator(aFaceOrdinal) != 0)
            {
                aResult(tCellOrdinal) /= tResultDenominator(aFaceOrdinal);
            }
            aResult(tCellOrdinal) = tWeight * aResult(tCellOrdinal) * aResult(tCellOrdinal);

        }, "divide and apply weight");
    }

  public:

    void
    faceCentroids(
              Omega_h::Mesh            & aMesh,
        const Omega_h::LOs             & aFaceLids,
              Plato::ScalarMultiVector   aCentroids
    )
    {
        auto tCoords = aMesh.coords();
        auto tFace2Verts = aMesh.ask_verts_of(SpaceDim-1);

        constexpr auto cNodesPerFace = SpaceDim;

        auto tNumFaces = aFaceLids.size();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumFaces), LAMBDA_EXPRESSION(const Plato::OrdinalType & aFaceOrdinal)
        {

            auto tFaceLid = aFaceLids[aFaceOrdinal];

            for (Plato::OrdinalType tNodeI=0; tNodeI<cNodesPerFace; tNodeI++)
            {
                auto tVertexOrdinal = tFace2Verts[tFaceLid*cNodesPerFace+tNodeI];
                for (Plato::OrdinalType tDimI=0; tDimI<SpaceDim; tDimI++)
                {
                    aCentroids(tDimI, aFaceOrdinal) += tCoords[tVertexOrdinal * SpaceDim + tDimI] / cNodesPerFace;
                }
            }

        }, "compute centroids");
    }

    void
    createPointGraph(
              Omega_h::Mesh     & aMesh,
        const Omega_h::MeshSets & aMeshSets
    )
    {
        std::stringstream tRowMapName;
        tRowMapName << mPointCloudName << "_rowmap";
        mPointCloudRowMapName = tRowMapName.str();

        std::stringstream tColMapName;
        tColMapName << mPointCloudName << "_colmap";
        mPointCloudColMapName = tColMapName.str();

        // only create the point cloud to face graph once
        if (mDataMap.ordinalVectors.count(mPointCloudRowMapName) == 0)
        {
            auto tFaceLids = Plato::get_face_ordinals(aMeshSets, mSideSetName);

            // create centroids
            Plato::ScalarMultiVector tCentroids("face centroids", SpaceDim, tFaceLids.size());
            faceCentroids(aMesh, tFaceLids, tCentroids);

            // construct search tree (this needs to be done in the constructor since the search result doesn't change)
            Plato::ScalarVector prim_x = Kokkos::subview(tCentroids, 0, Kokkos::ALL());
            Plato::ScalarVector prim_y = Kokkos::subview(tCentroids, 1, Kokkos::ALL());
            Plato::ScalarVector prim_z = Kokkos::subview(tCentroids, 2, Kokkos::ALL());
            Plato::OrdinalType tNumPrimitives = prim_x.extent(0);
            ArborX::BVH<Plato::DeviceType> bvh{Plato::GMF::Points{prim_x.data(), prim_y.data(), prim_z.data(), tNumPrimitives}};

            auto tPoints = mDataMap.scalarMultiVectors[mPointCloudName];
            Plato::ScalarVector pred_x = Kokkos::subview(tPoints, 0, Kokkos::ALL());
            Plato::ScalarVector pred_y = Kokkos::subview(tPoints, 1, Kokkos::ALL());
            Plato::ScalarVector pred_z = Kokkos::subview(tPoints, 2, Kokkos::ALL());
            Plato::OrdinalType tNumPredicates = pred_x.extent(0);

            Kokkos::View<int*, Plato::DeviceType> tIndices("indices", 0);
            Kokkos::View<int*, Plato::DeviceType> tOffsets("offsets", 0);

            bvh.query(Plato::GMF::Points{pred_x.data(), pred_y.data(), pred_z.data(), tNumPredicates}, tIndices, tOffsets);

            mDataMap.ordinalVectors[mPointCloudRowMapName] = tOffsets;
            mDataMap.ordinalVectors[mPointCloudColMapName] = tIndices;

            writeSearchResults(tPoints, tCentroids, tOffsets, tIndices);
        }
    }

    void
    writeSearchResults(
      const Plato::ScalarMultiVector                 & aPoints,
      const Plato::ScalarMultiVector                 & aCentroids,
      const Plato::ScalarVectorT<Plato::OrdinalType> & aOffsets,
      const Plato::ScalarVectorT<Plato::OrdinalType> & aIndices
    )
    {
        toVTK(/*fileBaseName=*/ "geometryMisfitPoints", aPoints);

        toVTK(/*fileBaseName=*/ "geometryMisfitCentroids", aCentroids);

        // compute vectors from aPoints to nearest aCentroid
        auto tNumPoints = aPoints.extent(1);
        auto tNumDims = aPoints.extent(0);
        Plato::ScalarMultiVector tVectors("vectors", tNumDims, tNumPoints);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumPoints), LAMBDA_EXPRESSION(const Plato::OrdinalType & aPointOrdinal)
        {
            // TODO index directly into aIndices?
            auto tCentroidOrd = aIndices(aOffsets(aPointOrdinal));
            for(Plato::OrdinalType iDim=0; iDim<SpaceDim; iDim++)
            {
                tVectors(iDim, aPointOrdinal) = aCentroids(iDim, tCentroidOrd) - aPoints(iDim, aPointOrdinal);
            }
        }, "vectors");

        // write vectors from points to nearest centroid
        toVTK(/*fileBaseName=*/ "geometryMisfitVectors", aPoints, tVectors);
    }

    void
    writeVTKHeader(
        std::ofstream & aOutFile
    )
    {
        aOutFile << "# vtk DataFile Version 2.0"  << std::endl;
        aOutFile << "Point data" << std::endl;
        aOutFile << "ASCII" << std::endl;
        aOutFile << "DATASET POLYDATA" << std::endl;
    }



    void
    toVTK(
            std::string                aBaseName,
      const Plato::ScalarMultiVector & aPoints
    )
    {
        std::ofstream tOutFile;
        tOutFile.open(aBaseName+".vtk");

        writeVTKHeader(tOutFile);

        auto tNumEntries = aPoints.extent(1);
        tOutFile << "POINTS " << tNumEntries << " float" << std::endl;

        writeToStream(tOutFile, aPoints);
    }

    void
    writeToStream(
            std::ostream             & aOutFile,
      const Plato::ScalarMultiVector & aEntries
    )
    {
        auto tEntries_Host = Kokkos::create_mirror_view(aEntries);
        Kokkos::deep_copy(tEntries_Host, aEntries);

        auto tNumDims = tEntries_Host.extent(0);
        auto tNumEntries = tEntries_Host.extent(1);
        for(Plato::OrdinalType iEntry=0; iEntry<tNumEntries; iEntry++)
        {
            for(Plato::OrdinalType iDim=0; iDim<tNumDims; iDim++)
            {
                aOutFile << tEntries_Host(iDim, iEntry) << " ";
            }
            aOutFile << std::endl;
        }
    }

    void
    toVTK(
            std::string                aBaseName,
      const Plato::ScalarMultiVector & aPoints,
      const Plato::ScalarMultiVector & aVectors
    )
    {
        std::ofstream tOutFile;
        tOutFile.open(aBaseName+".vtk");

        writeVTKHeader(tOutFile);

        auto tNumEntries = aPoints.extent(1);
        tOutFile << "POINTS " << tNumEntries << " float" << std::endl;
        writeToStream(tOutFile, aPoints);

        tOutFile << "POINT_DATA " << tNumEntries << std::endl;
        tOutFile << "VECTORS vectors float" << std::endl;
        writeToStream(tOutFile, aVectors);
    }

    void
    parsePointCloud(
        Teuchos::ParameterList & aFunctionParams
    )
    {
        mPointCloudName = aFunctionParams.get<std::string>("Point Cloud File Name");

        // only read the point cloud once
        if (mDataMap.scalarMultiVectors.count(mPointCloudName) == 0)
        {
            auto tHostPoints = this->readPointsFromFile(mPointCloudName);
            auto tDevicePoints = scalarMultiVectorFromData(tHostPoints);

            mDataMap.scalarMultiVectors[mPointCloudName] = tDevicePoints;
        }
    }

    inline std::vector<Plato::Scalar>
    readPoint(
        std::string tLineIn
    )
    {
        std::stringstream tStreamIn(tLineIn);
        std::vector<Plato::Scalar> tParsedValues;
        while (tStreamIn.good())
        {
            std::string tSubstr;
            getline(tStreamIn, tSubstr, ',');
            auto tNewValue = stod(tSubstr);
            tParsedValues.push_back(stod(tSubstr));
        }
        if (tParsedValues.size() != SpaceDim)
        {
            THROWERR("Error reading point cloud: line encountered with other than three values");
        }
        return tParsedValues;
    }

    std::vector<std::vector<Plato::Scalar>>
    readPointsFromFile(std::string aFileName)
    {
        std::vector<std::vector<Plato::Scalar>> tPoints;
        std::string tLineIn;
        std::ifstream tFileIn (mPointCloudName);
        if (tFileIn.is_open())
        {
            while (getline(tFileIn, tLineIn))
            {
                if (tLineIn.size() > 0 && tLineIn[0] != '#')
                {
                    auto tNewPoint = readPoint(tLineIn);
                    tPoints.push_back(tNewPoint);
                }
            }
            tFileIn.close();
        }
        else
        {
            THROWERR("Failed to open point cloud file.");
        }
        return tPoints;
    }

    Plato::ScalarMultiVector
    scalarMultiVectorFromData(
        std::vector<std::vector<Plato::Scalar>> aPoints
    )
    {
        Plato::ScalarMultiVector tDevicePoints("point cloud", SpaceDim, aPoints.size());
        auto tDevicePoints_Host = Kokkos::create_mirror_view(tDevicePoints);

        for (Plato::OrdinalType iPoint=0; iPoint<aPoints.size(); iPoint++)
        {
            for (Plato::OrdinalType iDim=0; iDim<SpaceDim; iDim++)
            {
                tDevicePoints_Host(iDim, iPoint) = aPoints[iPoint][iDim];
            }
        }
        Kokkos::deep_copy(tDevicePoints, tDevicePoints_Host);

        return tDevicePoints;
    }
};

} // namespace Geometric

} // namespace Plato

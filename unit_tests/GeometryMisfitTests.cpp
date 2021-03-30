/*
 * GeometryMisfitTest.cpp
 *
 *  Created on: March 11, 2021
 */


// TODO
// --  add evaluate_boundary to gradient_x() in GeometryScalarFunction.hpp
// --  add evaluate_boundary to gradient_z() in GeometryScalarFunction.hpp

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoTestHelpers.hpp"

#include <ArborX.hpp>

#include "Simplex.hpp"
#include "Geometrical.hpp"
#include "PlatoStaticsTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "OmegaHUtilities.hpp"
#include "NaturalBCUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "geometric/WorksetBase.hpp"
#include "geometric/GeometricSimplexFadTypes.hpp"
#include "geometric/AbstractScalarFunction.hpp"

#include <iostream>
#include <sstream>

namespace Plato
{
namespace Devel
{
    struct Points
    {
      double *d_x;
      double *d_y;
      double *d_z;
      int N;
    };
} // end namespace Devel
} // end namespace Plato

namespace Plato
{
namespace Geometric
{
enum Dim { X=0, Y, Z };

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = typename ExecSpace::memory_space;
using DeviceType = Kokkos::Device<ExecSpace, MemSpace>;

} // end namespace Geometric
} // end namespace Plato


namespace ArborX
{
namespace Traits
{
template <>
struct Access<Plato::Devel::Points, PrimitivesTag>
{
  inline static std::size_t size(Plato::Devel::Points const &points) { return points.N; }
  KOKKOS_INLINE_FUNCTION static Point get(Plato::Devel::Points const &points, std::size_t i)
  {
    return {{points.d_x[i], points.d_y[i], points.d_z[i]}};
  }
  using memory_space = Plato::Geometric::MemSpace;
};
template <>
struct Access<Plato::Devel::Points, PredicatesTag>
{
  inline static std::size_t size(Plato::Devel::Points const &d) { return d.N; }
  KOKKOS_INLINE_FUNCTION static auto get(Plato::Devel::Points const &d, std::size_t i)
  {
    return nearest(Point{d.d_x[i], d.d_y[i], d.d_z[i]}, 1);
  }
  using memory_space = Plato::Geometric::MemSpace;
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
        Plato::ComputeSurfaceJacobians<SpaceDim> tComputeSurfaceJacobians;
        Plato::ComputeSurfaceIntegralWeight<SpaceDim> tComputeSurfaceIntegralWeight;
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
            ConfigScalarType tDistance = P_x*n_x + P_y*n_y + P_z*n_z;

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
            tComputeSurfaceJacobians(tCellOrdinal, aFaceOrdinal, tLocalNodeOrd, aConfig, tJacobian);
            tComputeSurfaceIntegralWeight(aFaceOrdinal, tMultiplier, tJacobian, tWeight);

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
                    aCentroids(tDimI, aFaceOrdinal) += tCoords[tVertexOrdinal * SpaceDim + tDimI];
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
            ArborX::BVH<DeviceType> bvh{Plato::Devel::Points{prim_x.data(), prim_y.data(), prim_z.data(), tNumPrimitives}};

            auto tPoints = mDataMap.scalarMultiVectors[mPointCloudName];
            Plato::ScalarVector pred_x = Kokkos::subview(tPoints, 0, Kokkos::ALL());
            Plato::ScalarVector pred_y = Kokkos::subview(tPoints, 1, Kokkos::ALL());
            Plato::ScalarVector pred_z = Kokkos::subview(tPoints, 2, Kokkos::ALL());
            Plato::OrdinalType tNumPredicates = pred_x.extent(0);

            Kokkos::View<int*, DeviceType> tIndices("indices", 0);
            Kokkos::View<int*, DeviceType> tOffsets("offsets", 0);

            bvh.query(Plato::Devel::Points{pred_x.data(), pred_y.data(), pred_z.data(), tNumPredicates}, tIndices, tOffsets);

            mDataMap.ordinalVectors[mPointCloudRowMapName] = tOffsets;
            mDataMap.ordinalVectors[mPointCloudColMapName] = tIndices;
        }
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
        if (tParsedValues.size() > SpaceDim)
        {
            THROWERR("Error reading point cloud: line encountered with less than three values");
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
// class GeometryMisfit

} // namespace Geometric

} // namespace Plato

namespace GeometryMisfitTest
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Misfit)
{
  // create geometry misfit criterion
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                  \n"
    "  <ParameterList name='Spatial Model'>                                                \n"
    "    <ParameterList name='Domains'>                                                    \n"
    "      <ParameterList name='Design Volume'>                                            \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                  \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>          \n"
    "      </ParameterList>                                                                \n"
    "    </ParameterList>                                                                  \n"
    "  </ParameterList>                                                                    \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>                   \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                           \n"
    "  <ParameterList name='Material Models'>                                              \n"
    "    <ParameterList name='Unobtainium'>                                                \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                 \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>                  \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>                \n"
    "      </ParameterList>                                                                \n"
    "    </ParameterList>                                                                  \n"
    "  </ParameterList>                                                                    \n"
    "  <ParameterList name='Criteria'>                                                     \n"
    "    <ParameterList name='Geometry Misfit'>                                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function' />                 \n"
    "      <Parameter name='Linear' type='bool' value='true' />                            \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Geometry Misfit' /> \n"
    "      <Parameter name='Point Cloud File Name' type='string' value='points.xyz' />     \n"
    "      <Parameter name='Sides' type='string' value='x+' />                             \n"
    "    </ParameterList>                                                                  \n"
    "  </ParameterList>                                                                    \n"
    "</ParameterList>                                                                      \n"
  );

    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    Plato::DataMap tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    using GeometryT = typename Plato::Geometrical<tSpaceDim>;
    using ResidualT = typename Plato::Geometric::Evaluation<typename GeometryT::SimplexT>::Residual;

    std::string tFunctionName("Geometry Misfit");
    auto tGeometryMisfit = Plato::Geometric::GeometryMisfit<ResidualT>(tOnlyDomain, tDataMap, *tParamList, tFunctionName);

    auto aFileName = tParamList->sublist("Criteria").sublist("Geometry Misfit").get<std::string>("Point Cloud File Name");
    auto tHostPoints = tGeometryMisfit.readPointsFromFile(aFileName);

    // verify that points loaded correctly
    //
    std::vector<std::vector<double>> tPointData = {
      {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}
    };
    for (int iPoint=0; iPoint<tPointData.size(); iPoint++)
    {
        for (int iDim=0; iDim<tSpaceDim; iDim++)
        {
            TEST_FLOATING_EQUALITY(tPointData[iPoint][iDim], tHostPoints[iPoint][iDim], 1e-16);
        }
    }

    // check the DataMap
    //
    TEST_ASSERT(tDataMap.scalarMultiVectors.count(aFileName) == 1);
    auto tDevicePoints = tDataMap.scalarMultiVectors.at(aFileName);
    auto tDevicePoints_Host = Kokkos::create_mirror_view(tDevicePoints);
    Kokkos::deep_copy(tDevicePoints_Host, tDevicePoints);

    for (int iPoint=0; iPoint<tPointData.size(); iPoint++)
    {
        for (int iDim=0; iDim<tSpaceDim; iDim++)
        {
            TEST_FLOATING_EQUALITY(tPointData[iPoint][iDim], tDevicePoints_Host(iDim, iPoint), 1e-16);
        }
    }

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarMultiVectorT<ResidualT::ControlScalarType> tControlWS("design variables", tNumCells, GeometryT::mNumNodesPerCell);
    Kokkos::deep_copy(tControlWS, 1.0);

    // evaluate the function
    //
    using WorksetBaseT = typename Plato::Geometric::WorksetBase<GeometryT>;
    WorksetBaseT tWorksetBase(*tMesh);

    {
        Plato::ScalarArray3DT<ResidualT::ConfigScalarType> tConfigWS("configuration", tNumCells, GeometryT::mNumNodesPerCell, tSpaceDim);
        tWorksetBase.worksetConfig(tConfigWS, tOnlyDomain);

        Plato::ScalarVectorT<ResidualT::ResultScalarType> tResultWS("result", tNumCells);

        tGeometryMisfit.evaluate_boundary(tSpatialModel, tControlWS, tConfigWS, tResultWS);

        auto tResultWS_Host = Kokkos::create_mirror_view(tResultWS);
        Kokkos::deep_copy(tResultWS_Host, tResultWS);

        Plato::Scalar tError = 0.0;
        for (int iCell=0; iCell<tResultWS.extent(0); iCell++)
        {
            tError += tResultWS_Host(iCell);
        }
        TEST_FLOATING_EQUALITY(tError, 1.0, 1e-16);
    }


    {
        using GradientX = typename Plato::Geometric::Evaluation<typename GeometryT::SimplexT>::GradientX;
        auto tGeometryMisfit_GradientX = Plato::Geometric::GeometryMisfit<GradientX>(tOnlyDomain, tDataMap, *tParamList, tFunctionName);

        Plato::ScalarArray3DT<GradientX::ConfigScalarType> tConfigWS("configuration", tNumCells, GeometryT::mNumNodesPerCell, tSpaceDim);
        tWorksetBase.worksetConfig(tConfigWS, tOnlyDomain);

        Plato::ScalarVectorT<GradientX::ResultScalarType> tResultWS("result", tNumCells);

        tGeometryMisfit_GradientX.evaluate_boundary(tSpatialModel, tControlWS, tConfigWS, tResultWS);

        auto tResultWS_Host = Kokkos::create_mirror_view(tResultWS);
        Kokkos::deep_copy(tResultWS_Host, tResultWS);

        std::vector<std::vector<Plato::Scalar>> tGold_gradX{
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, -0.5, 1, -0.5, 0.5, 0, 0.5, 0, 0, 0, 0},
          {0, -0.5, 0, 0, 0, 0.5, 1, 0.5, -0.5, 0, 0, 0}
        };

        Plato::OrdinalType tNumNodesPerCell = tSpaceDim+1;
        Plato::OrdinalType tNumDofsPerCell = tSpaceDim*tNumNodesPerCell;
        for (int iCell=0; iCell<tResultWS.extent(0); iCell++)
        {
            for (int iDof=0; iDof<tNumDofsPerCell; iDof++)
            {
                if (tGold_gradX[iCell][iDof] == 0)
                {
                    TEST_ASSERT(fabs(tResultWS_Host(iCell).dx(iDof)) < 1e-15);
                }
                else
                {
                    TEST_FLOATING_EQUALITY(tGold_gradX[iCell][iDof], tResultWS_Host(iCell).dx(iDof), 1e-15);
                }
            }
        }
    }
}

} // namespace GeometryMisfitTest

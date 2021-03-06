/*
 * OmegaHUtilitiesTests.cpp
 *
 *  Created on: Mar 18, 2020
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoUtilities.hpp"
#include "OmegaHUtilities.hpp"
#include "ImplicitFunctors.hpp"
#include "PlatoTestHelpers.hpp"


namespace OmegaHUtilitiesTests
{


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, OmegaH_EdgeID_OnY1)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,10,2);
    auto tIDs = PlatoUtestHelpers::get_edge_ids_on_y1(tMesh.operator*());
    Plato::ScalarVector tResults("face ordinals", tIDs.size());
    Kokkos::parallel_for("print array", tIDs.size(), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tResults(aIndex) = tIDs[aIndex];
        //printf("X(%d)=%d\n", aIndex, tIDs[aIndex]);
    });

    auto tHostResults = Kokkos::create_mirror(tResults);
    Kokkos::deep_copy(tHostResults, tResults);
    TEST_EQUALITY(11, tHostResults(0));
    TEST_EQUALITY(14, tHostResults(1));
    TEST_EQUALITY(26, tHostResults(2));
    TEST_EQUALITY(33, tHostResults(3));
    TEST_EQUALITY(35, tHostResults(4));
    TEST_EQUALITY(37, tHostResults(5));
    TEST_EQUALITY(49, tHostResults(6));
    TEST_EQUALITY(61, tHostResults(7));
    TEST_EQUALITY(62, tHostResults(8));
    TEST_EQUALITY(64, tHostResults(9));
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, OmegaH_EdgeID_OnY0)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,10,2);
    auto tIDs = PlatoUtestHelpers::get_edge_ids_on_y0(tMesh.operator*());
    Plato::ScalarVector tResults("face ordinals", tIDs.size());
    Kokkos::parallel_for("print array", tIDs.size(), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tResults(aIndex) = tIDs[aIndex];
        //printf("X(%d)=%d\n", aIndex, tIDs[aIndex]);
    });

    auto tHostResults = Kokkos::create_mirror(tResults);
    Kokkos::deep_copy(tHostResults, tResults);
    TEST_EQUALITY(0, tHostResults(0));
    TEST_EQUALITY(5, tHostResults(1));
    TEST_EQUALITY(15, tHostResults(2));
    TEST_EQUALITY(20, tHostResults(3));
    TEST_EQUALITY(27, tHostResults(4));
    TEST_EQUALITY(41, tHostResults(5));
    TEST_EQUALITY(46, tHostResults(6));
    TEST_EQUALITY(53, tHostResults(7));
    TEST_EQUALITY(58, tHostResults(8));
    TEST_EQUALITY(68, tHostResults(9));
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, OmegaH_EdgeID_OnX0)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,10,2);
    auto tIDs = PlatoUtestHelpers::get_edge_ids_on_x0(tMesh.operator*());
    Plato::ScalarVector tResults("face ordinals", tIDs.size());
    Kokkos::parallel_for("print array", tIDs.size(), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tResults(aIndex) = tIDs[aIndex];
        //printf("X(%d)=%d\n", aIndex, tIDs[aIndex]);
    });

    auto tHostResults = Kokkos::create_mirror(tResults);
    Kokkos::deep_copy(tHostResults, tResults);
    TEST_EQUALITY(2, tHostResults(0));
    TEST_EQUALITY(12, tHostResults(1));
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, OmegaH_EdgeID_OnX1)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,10,2);
    auto tIDs = PlatoUtestHelpers::get_edge_ids_on_x1(tMesh.operator*());
    Plato::ScalarVector tResults("face ordinals", tIDs.size());
    Kokkos::parallel_for("print array", tIDs.size(), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tResults(aIndex) = tIDs[aIndex];
        //printf("X(%d)=%d\n", aIndex, tIDs[aIndex]);
    });

    auto tHostResults = Kokkos::create_mirror(tResults);
    Kokkos::deep_copy(tHostResults, tResults);
    TEST_EQUALITY(70, tHostResults(0));
    TEST_EQUALITY(71, tHostResults(1));
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, OmegaHGrapsh)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumFaces = 6;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    auto tNumCells = tMesh->nelems();
    auto tElem2FaceMap = tMesh->ask_down(tSpaceDim,tSpaceDim-1);
    Plato::ScalarVector tResults("face ordinals", tNumFaces + 1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, 1), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tResults(0) = Plato::get_face_ordinal<tSpaceDim>(0 /*cell*/, 0 /*face*/, tElem2FaceMap.ab2b);
        tResults(1) = Plato::get_face_ordinal<tSpaceDim>(0 /*cell*/, 4 /*face*/, tElem2FaceMap.ab2b);
        tResults(2) = Plato::get_face_ordinal<tSpaceDim>(0 /*cell*/, 1 /*face*/, tElem2FaceMap.ab2b);
        tResults(3) = Plato::get_face_ordinal<tSpaceDim>(1 /*cell*/, 3 /*face*/, tElem2FaceMap.ab2b);
        tResults(4) = Plato::get_face_ordinal<tSpaceDim>(1 /*cell*/, 2 /*face*/, tElem2FaceMap.ab2b);
        tResults(5) = Plato::get_face_ordinal<tSpaceDim>(1 /*cell*/, 1 /*face*/, tElem2FaceMap.ab2b);
        tResults(6) = Plato::get_face_ordinal<tSpaceDim>(1 /*cell*/, 4 /*face*/, tElem2FaceMap.ab2b);
    }, "test get_face_ordinal");

    auto tHostResults = Kokkos::create_mirror(tResults);
    Kokkos::deep_copy(tHostResults, tResults);
    TEST_EQUALITY(0,    tHostResults(0));
    TEST_EQUALITY(1,    tHostResults(1));
    TEST_EQUALITY(2,    tHostResults(2));
    TEST_EQUALITY(0,    tHostResults(3));
    TEST_EQUALITY(1,    tHostResults(4));
    TEST_EQUALITY(2,    tHostResults(5));
    TEST_EQUALITY(-100, tHostResults(6));
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LocalElementCoords_1D)
{
    constexpr Plato::OrdinalType tSpaceDim = 1;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    //printf("\n");
    auto tNumCells = tMesh->nelems();
    constexpr auto tNodesPerCell = tSpaceDim + 1;
    Plato::ScalarArray3D tCellCoords("normals", tNumCells, tNodesPerCell, tSpaceDim);

    Plato::NodeCoordinate<tSpaceDim> tCoords(tMesh.getRawPtr());
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        auto tCellPoints = Plato::local_element_coords<tSpaceDim>(aCellIndex, tCoords);
        for (Plato::OrdinalType jNode = 0; jNode < tNodesPerCell; jNode++)
        {
            for (Plato::OrdinalType tDim = 0; tDim < tSpaceDim; tDim++)
            {
                tCellCoords(aCellIndex, jNode, tDim) = tCellPoints[jNode][tDim];
                //printf("Coords[%d][%d][%d] = %f\n", aCellIndex, jNode, tDim, tCellPoints[jNode][tDim]);
            }
        }
    }, "test local_element_coords function - return cell coordinates");

    Plato::ScalarArray3D tGold("gold", tNumCells, tNodesPerCell, tSpaceDim);
    auto tHostGold = Kokkos::create_mirror(tGold);
    tHostGold(0,0,0) = 0.0; tHostGold(0,1,0) = 1.0;

    const Plato::Scalar tTolerance = 1e-4;
    auto tHostCellCoords = Kokkos::create_mirror(tCellCoords);
    Kokkos::deep_copy(tHostCellCoords, tCellCoords);
    for(Plato::OrdinalType tCell=0; tCell < tNumCells; tCell++)
    {
        for(Plato::OrdinalType tNode=0; tNode < tNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim=0; tDim < tSpaceDim; tDim++)
            {
                TEST_FLOATING_EQUALITY(tHostCellCoords(tCell, tNode, tDim), tHostGold(tCell, tNode, tDim), tTolerance);
            }
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LocalElementCoords_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    //printf("\n");
    auto tNumCells = tMesh->nelems();
    constexpr auto tNodesPerCell = tSpaceDim + 1;
    Plato::ScalarArray3D tCellCoords("normals", tNumCells, tNodesPerCell, tSpaceDim);

    Plato::NodeCoordinate<tSpaceDim> tCoords(tMesh.getRawPtr());
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        auto tCellPoints = Plato::local_element_coords<tSpaceDim>(aCellIndex, tCoords);
        for (Plato::OrdinalType jNode = 0; jNode < tNodesPerCell; jNode++)
        {
            for (Plato::OrdinalType tDim = 0; tDim < tSpaceDim; tDim++)
            {
                tCellCoords(aCellIndex, jNode, tDim)= tCellPoints[jNode][tDim];
                //printf("Coords[%d][%d][%d] = %f\n", aCellIndex, jNode, tDim, tCellPoints[jNode][tDim]);
            }
        }
    }, "test local_element_coords function - returns cell coordinates");

    Plato::ScalarArray3D tGold("gold", tNumCells, tNodesPerCell, tSpaceDim);
    auto tHostGold = Kokkos::create_mirror(tGold);
    tHostGold(0,0,0) = 0.0; tHostGold(0,0,1) = 0.0; tHostGold(0,1,0) = 1.0; tHostGold(0,1,1) = 0.0; tHostGold(0,2,0) = 1.0; tHostGold(0,2,1) = 1.0;
    tHostGold(1,0,0) = 1.0; tHostGold(1,0,1) = 1.0; tHostGold(1,1,0) = 0.0; tHostGold(1,1,1) = 1.0; tHostGold(1,2,0) = 0.0; tHostGold(1,2,1) = 0.0;

    const Plato::Scalar tTolerance = 1e-4;
    auto tHostCellCoords = Kokkos::create_mirror(tCellCoords);
    Kokkos::deep_copy(tHostCellCoords, tCellCoords);
    for(Plato::OrdinalType tCell=0; tCell < tNumCells; tCell++)
    {
        for(Plato::OrdinalType tNode=0; tNode < tNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim=0; tDim < tSpaceDim; tDim++)
            {
                TEST_FLOATING_EQUALITY(tHostCellCoords(tCell, tNode, tDim), tHostGold(tCell, tNode, tDim), tTolerance);
            }
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LocalElementCoords_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    //printf("\n");
    auto tNumCells = tMesh->nelems();
    constexpr auto tNodesPerCell = tSpaceDim + 1;
    Plato::ScalarArray3D tCellCoords("normals", tNumCells, tNodesPerCell, tSpaceDim);

    Plato::NodeCoordinate<tSpaceDim> tCoords(tMesh.getRawPtr());
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        auto tCellPoints = Plato::local_element_coords<tSpaceDim>(aCellIndex, tCoords);
        for (Plato::OrdinalType jNode = 0; jNode < tNodesPerCell; jNode++)
        {
            for (Plato::OrdinalType tDim = 0; tDim < tSpaceDim; tDim++)
            {
                tCellCoords(aCellIndex, jNode, tDim) = tCellPoints[jNode][tDim];
                //printf("Coords[%d][%d][%d] = %f\n", aCellIndex, jNode, tDim, tCellPoints[jNode][tDim]);
            }
        }
    }, "test local_element_coords function - returns cell coordinates");

    Plato::ScalarArray3D tGold("gold", tNumCells, tNodesPerCell, tSpaceDim);
    auto tHostGold = Kokkos::create_mirror(tGold);
    // ELEM ONE
    tHostGold(0,0,0) = 0.0; tHostGold(0,0,1) = 0.0; tHostGold(0,0,2) = 0.0;
    tHostGold(0,1,0) = 1.0; tHostGold(0,1,1) = 1.0; tHostGold(0,1,2) = 0.0;
    tHostGold(0,2,0) = 0.0; tHostGold(0,2,1) = 1.0; tHostGold(0,2,2) = 0.0;
    tHostGold(0,3,0) = 1.0; tHostGold(0,3,1) = 1.0; tHostGold(0,3,2) = 1.0;
    // ELEM TWO
    tHostGold(1,0,0) = 0.0; tHostGold(1,0,1) = 0.0; tHostGold(1,0,2) = 0.0;
    tHostGold(1,1,0) = 0.0; tHostGold(1,1,1) = 1.0; tHostGold(1,1,2) = 0.0;
    tHostGold(1,2,0) = 0.0; tHostGold(1,2,1) = 1.0; tHostGold(1,2,2) = 1.0;
    tHostGold(1,3,0) = 1.0; tHostGold(1,3,1) = 1.0; tHostGold(1,3,2) = 1.0;
    // ELEM THREE
    tHostGold(2,0,0) = 0.0; tHostGold(2,0,1) = 0.0; tHostGold(2,0,2) = 0.0;
    tHostGold(2,1,0) = 0.0; tHostGold(2,1,1) = 1.0; tHostGold(2,1,2) = 1.0;
    tHostGold(2,2,0) = 0.0; tHostGold(2,2,1) = 0.0; tHostGold(2,2,2) = 1.0;
    tHostGold(2,3,0) = 1.0; tHostGold(2,3,1) = 1.0; tHostGold(2,3,2) = 1.0;
    // ELEM FOUR
    tHostGold(3,0,0) = 0.0; tHostGold(3,0,1) = 0.0; tHostGold(3,0,2) = 0.0;
    tHostGold(3,1,0) = 1.0; tHostGold(3,1,1) = 0.0; tHostGold(3,1,2) = 1.0;
    tHostGold(3,2,0) = 1.0; tHostGold(3,2,1) = 1.0; tHostGold(3,2,2) = 1.0;
    tHostGold(3,3,0) = 0.0; tHostGold(3,3,1) = 0.0; tHostGold(3,3,2) = 1.0;
    // ELEM FIVE
    tHostGold(4,0,0) = 1.0; tHostGold(4,0,1) = 0.0; tHostGold(4,0,2) = 0.0;
    tHostGold(4,1,0) = 1.0; tHostGold(4,1,1) = 0.0; tHostGold(4,1,2) = 1.0;
    tHostGold(4,2,0) = 1.0; tHostGold(4,2,1) = 1.0; tHostGold(4,2,2) = 1.0;
    tHostGold(4,3,0) = 0.0; tHostGold(4,3,1) = 0.0; tHostGold(4,3,2) = 0.0;
    // ELEM SIX
    tHostGold(5,0,0) = 1.0; tHostGold(5,0,1) = 0.0; tHostGold(5,0,2) = 0.0;
    tHostGold(5,1,0) = 1.0; tHostGold(5,1,1) = 1.0; tHostGold(5,1,2) = 1.0;
    tHostGold(5,2,0) = 1.0; tHostGold(5,2,1) = 1.0; tHostGold(5,2,2) = 0.0;
    tHostGold(5,3,0) = 0.0; tHostGold(5,3,1) = 0.0; tHostGold(5,3,2) = 0.0;

    const Plato::Scalar tTolerance = 1e-4;
    auto tHostCellCoords = Kokkos::create_mirror(tCellCoords);
    Kokkos::deep_copy(tHostCellCoords, tCellCoords);
    for(Plato::OrdinalType tCell=0; tCell < tNumCells; tCell++)
    {
        for(Plato::OrdinalType tNode=0; tNode < tNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim=0; tDim < tSpaceDim; tDim++)
            {
                TEST_FLOATING_EQUALITY(tHostCellCoords(tCell, tNode, tDim), tHostGold(tCell, tNode, tDim), tTolerance);
            }
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ComputeNormals_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    //printf("\n");
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumEdges = tSpaceDim + 1;
    Plato::ScalarArray3D tNormalVectors("normals", tNumCells, tNumEdges, tSpaceDim);

    Plato::NodeCoordinate<tSpaceDim> tCoords(tMesh.getRawPtr());
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        for(Plato::OrdinalType tEdgeIndex=0; tEdgeIndex < tNumEdges; tEdgeIndex++)
        {
            auto tNormalVec = Plato::unit_normal_vector(aCellIndex, tEdgeIndex, tCoords);
            for( Plato::OrdinalType tDim=0; tDim < tSpaceDim; tDim++)
            {
                tNormalVectors(aCellIndex, tEdgeIndex, tDim) = tNormalVec[tDim];
                //printf("N[%d][%d][%d] = %f\n", aCellIndex, tEdgeIndex, tDim, tNormalVec[tDim]);
            }
        }
    }, "test get_side_vector function - returns normal vectors");

    Plato::ScalarArray3D tGold("gold", tNumCells, tNumEdges, tSpaceDim);
    auto tHostGold = Kokkos::create_mirror(tGold);
    // Edge ID One
    tHostGold(0,0,0) = 0.0; tHostGold(0,0,1) = -1.0;
    tHostGold(1,0,0) = 0.0; tHostGold(1,0,1) =  1.0;
    // Edge ID Two
    tHostGold(0,1,0) =  1.0; tHostGold(0,1,1) = 0.0;
    tHostGold(1,1,0) = -1.0; tHostGold(1,1,1) = 0.0;
    // Edge ID Three
    auto tMu = 1.0/sqrt(2.0);
    tHostGold(0,2,0) = -tMu; tHostGold(0,2,1) =  tMu;
    tHostGold(1,2,0) =  tMu; tHostGold(1,2,1) = -tMu;

    const Plato::Scalar tTolerance = 1e-4;
    auto tHostNormalVectors = Kokkos::create_mirror(tNormalVectors);
    Kokkos::deep_copy(tHostNormalVectors, tNormalVectors);
    for(Plato::OrdinalType tCellIndex=0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tEdgeIndex=0; tEdgeIndex < tNumEdges; tEdgeIndex++)
        {
            for(Plato::OrdinalType tDim=0; tDim < tSpaceDim; tDim++)
            {
                TEST_FLOATING_EQUALITY(tHostNormalVectors(tCellIndex, tEdgeIndex, tDim), tHostGold(tCellIndex, tEdgeIndex, tDim), tTolerance);
            }
        }
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ComputeNormals_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    //printf("\n");
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumFaces = tSpaceDim + 1;
    Plato::ScalarArray3D tNormalVectors("normals", tNumCells, tNumFaces, tSpaceDim);

    Plato::NodeCoordinate<tSpaceDim> tCoords(tMesh.getRawPtr());
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        for(Plato::OrdinalType tFaceIndex=0; tFaceIndex < tNumFaces; tFaceIndex++)
        {
            auto tNormalVec = Plato::unit_normal_vector(aCellIndex, tFaceIndex, tCoords);
            for( Plato::OrdinalType tDim=0; tDim < tSpaceDim; tDim++)
            {
                tNormalVectors(aCellIndex, tFaceIndex, tDim) = tNormalVec[tDim];
                //printf("N[%d][%d][%d] = %f\n", aCellIndex, tFaceIndex, tDim, tNormalVec[tDim]);
            }
        }
    }, "test get_side_vector function - returns normal vectors");

    Plato::ScalarArray3D tGold("gold", tNumCells, tNumFaces, tSpaceDim);
    auto tHostGold = Kokkos::create_mirror(tGold);
    auto tMu = 1.0/sqrt(2.0);
    // Face One
    tHostGold(0,0,0) =  0.0; tHostGold(0,0,1) = 0.0; tHostGold(0,0,2) = -1.0;
    tHostGold(1,0,0) = -1.0; tHostGold(1,0,1) = 0.0; tHostGold(1,0,2) =  0.0;
    tHostGold(2,0,0) = -1.0; tHostGold(2,0,1) = 0.0; tHostGold(2,0,2) =  0.0;
    tHostGold(3,0,0) = tMu;  tHostGold(3,0,1) = 0.0; tHostGold(3,0,2) = -tMu;
    tHostGold(4,0,0) = 1.0;  tHostGold(4,0,1) = 0.0; tHostGold(4,0,2) =  0.0;
    tHostGold(5,0,0) = 1.0;  tHostGold(5,0,1) = 0.0; tHostGold(5,0,2) =  0.0;
    // Face Two
    tHostGold(0,1,0) = tMu; tHostGold(0,1,1) = -tMu; tHostGold(0,1,2) =  0.0;
    tHostGold(1,1,0) = tMu; tHostGold(1,1,1) =  0.0; tHostGold(1,1,2) = -tMu;
    tHostGold(2,1,0) = 0.0; tHostGold(2,1,1) =  tMu; tHostGold(2,1,2) = -tMu;
    tHostGold(3,1,0) = 0.0; tHostGold(3,1,1) = -1.0; tHostGold(3,1,2) =  0.0;
    tHostGold(4,1,0) = 0.0; tHostGold(4,1,1) = -1.0; tHostGold(4,1,2) =  0.0;
    tHostGold(5,1,0) = 0.0; tHostGold(5,1,1) = -tMu; tHostGold(5,1,2) =  tMu;
    // Face Three
    tHostGold(0,2,0) =  0.0; tHostGold(0,2,1) = 1.0; tHostGold(0,2,2) = 0.0;
    tHostGold(1,2,0) =  0.0; tHostGold(1,2,1) = 1.0; tHostGold(1,2,2) = 0.0;
    tHostGold(2,2,0) =  0.0; tHostGold(2,2,1) = 0.0; tHostGold(2,2,2) = 1.0;
    tHostGold(3,2,0) =  0.0; tHostGold(3,2,1) = 0.0; tHostGold(3,2,2) = 1.0;
    tHostGold(4,2,0) = -tMu; tHostGold(4,2,1) = 0.0; tHostGold(4,2,2) = tMu;
    tHostGold(5,2,0) = -tMu; tHostGold(5,2,1) = tMu; tHostGold(5,2,2) = 0.0;
    // Face Four
    tHostGold(0,3,0) = -tMu; tHostGold(0,3,1) =  0.0; tHostGold(0,3,2) =  tMu;
    tHostGold(1,3,0) =  0.0; tHostGold(1,3,1) = -tMu; tHostGold(1,3,2) =  tMu;
    tHostGold(2,3,0) =  tMu; tHostGold(2,3,1) = -tMu; tHostGold(2,3,2) =  0.0;
    tHostGold(3,3,0) = -tMu; tHostGold(3,3,1) =  tMu; tHostGold(3,3,2) =  0.0;
    tHostGold(4,3,0) =  0.0; tHostGold(4,3,1) =  tMu; tHostGold(4,3,2) = -tMu;
    tHostGold(5,3,0) =  0.0; tHostGold(5,3,1) =  0.0; tHostGold(5,3,2) = -1.0;

    const Plato::Scalar tTolerance = 1e-4;
    auto tHostNormalVectors = Kokkos::create_mirror(tNormalVectors);
    Kokkos::deep_copy(tHostNormalVectors, tNormalVectors);
    for(Plato::OrdinalType tCellIndex=0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Plato::OrdinalType tFaceIndex=0; tFaceIndex < tNumFaces; tFaceIndex++)
        {
            for(Plato::OrdinalType tDim=0; tDim < tSpaceDim; tDim++)
            {
                TEST_FLOATING_EQUALITY(tHostNormalVectors(tCellIndex, tFaceIndex, tDim), tHostGold(tCellIndex, tFaceIndex, tDim), tTolerance);
            }
        }
    }
}


}
// namespace OmegaHUtilitiesTests

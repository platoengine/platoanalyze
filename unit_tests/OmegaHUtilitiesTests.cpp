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


namespace Plato
{

template<Plato::OrdinalType SpaceDim>
inline Omega_h::LO get_face_ordinal
(const Plato::OrdinalType& aCellOrdinal,
 const Plato::OrdinalType& aFaceOrdinal,
 const Omega_h::LOs& aElem2FaceMap)
{
    Omega_h::LO tOut = -100;
    auto tNumFacesPerCell = SpaceDim + 1;
    for(Plato::OrdinalType tFace = 0; tFace < tNumFacesPerCell; tFace++)
    {
        if(aElem2FaceMap[aCellOrdinal*tNumFacesPerCell+tFace] == aFaceOrdinal)
        {
            return tFace;
        }
    }
    return (tOut);
}

}

namespace OmegaHUtilitiesTests
{


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, OmegaHGrapsh)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    auto tNumFacesPerCell = tSpaceDim + 1;
    auto tElem2FaceMap = tMesh->ask_down(tSpaceDim,tSpaceDim-1);
    TEST_EQUALITY(0, Plato::get_face_ordinal<tSpaceDim>(0 /*cell*/, 0 /*face*/, tElem2FaceMap.ab2b));
    TEST_EQUALITY(1, Plato::get_face_ordinal<tSpaceDim>(0 /*cell*/, 4 /*face*/, tElem2FaceMap.ab2b));
    TEST_EQUALITY(2, Plato::get_face_ordinal<tSpaceDim>(0 /*cell*/, 1 /*face*/, tElem2FaceMap.ab2b));
    TEST_EQUALITY(0, Plato::get_face_ordinal<tSpaceDim>(1 /*cell*/, 3 /*face*/, tElem2FaceMap.ab2b));
    TEST_EQUALITY(1, Plato::get_face_ordinal<tSpaceDim>(1 /*cell*/, 2 /*face*/, tElem2FaceMap.ab2b));
    TEST_EQUALITY(2, Plato::get_face_ordinal<tSpaceDim>(1 /*cell*/, 1 /*face*/, tElem2FaceMap.ab2b));
    TEST_EQUALITY(-100, Plato::get_face_ordinal<tSpaceDim>(1 /*cell*/, 4 /*face*/, tElem2FaceMap.ab2b));
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
    constexpr auto tNodesPerCell = tSpaceDim + 1;
    Plato::ScalarArray3D tNormalVectors("normals", tNumCells, tNumEdges, tSpaceDim);

    Plato::NodeCoordinate<tSpaceDim> tCoords(tMesh.getRawPtr());
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        auto tCellPoints = Plato::local_element_coords<tSpaceDim>(aCellIndex, tCoords);

        for(Plato::OrdinalType tEdgeIndex=0; tEdgeIndex < tNumEdges; tEdgeIndex++)
        {
            auto tNormalVec = Plato::unit_normal_vector(tEdgeIndex, tCellPoints);
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
    constexpr auto tNodesPerCell = tSpaceDim + 1;
    Plato::ScalarArray3D tNormalVectors("normals", tNumCells, tNumFaces, tSpaceDim);

    Plato::NodeCoordinate<tSpaceDim> tCoords(tMesh.getRawPtr());
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellIndex)
    {
        auto tCellPoints = Plato::local_element_coords<tSpaceDim>(aCellIndex, tCoords);

        for(Plato::OrdinalType tFaceIndex=0; tFaceIndex < tNumFaces; tFaceIndex++)
        {
            auto tNormalVec = Plato::unit_normal_vector(tFaceIndex, tCellPoints);
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

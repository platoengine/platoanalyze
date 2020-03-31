/*
 * PlatoMeshMapTests.cpp
 *
 *  Created on: March 11, 2020
 */

#include "PlatoTestHelpers.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <sstream>
#include <fstream>
#include <stdio.h>

#include "Plato_InputData.hpp"
#include "Plato_Exceptions.hpp"
#include "Plato_Parser.hpp"

#define MAKE_PUBLIC
#include "Plato_MeshMap.hpp"

using SparseMatrix = Plato::Geometry::AbstractMeshMap<Plato::Scalar>::SparseMatrix;

std::vector<std::vector<Plato::Scalar>>
toFull( SparseMatrix aInMatrix );

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = typename ExecSpace::memory_space;

template <typename ViewType>
typename ViewType::HostMirror
get(ViewType aView)
{
    using RetType = typename ViewType::HostMirror;
    RetType tView = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView, aView);
    return tView;
}

namespace PlatoTestMeshMap
{

/******************************************************************************/
/*!
  \brief Compute basis function values

  1. Compute the centroid of each element in a test mesh in physical coordinates.
  2. Use GetBases() to determine the basis function values at the equivalent
     point in element coordinates.

     X_i = N(x)_I v^e_{Ii}

     v^e_{Ii}: vertex coordinates for local node I of element e
     X_i:      Input point for which basis values are determined. Element
               centroid in this test.
     x:        location of X in element coordinates.
     N(x)_I:   Basis values to be determines.

  test passes if:
    N(x)_I = {1/4, 1/4, 1/4, 1/4};
*/
/******************************************************************************/
  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, GetBasis)
  {

    // create mesh
    //
    constexpr int cMeshWidth=5;
    constexpr int cSpaceDim=3;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

    // create GetBasis functor
    //
    Plato::Geometry::GetBasis<Plato::Scalar> tGetBasis(*tMesh);

    auto tNElems = tMesh->nelems();
    Plato::ScalarMultiVector tInPoints("centroids", cSpaceDim, tNElems);
    Plato::ScalarMultiVector tBases("basis values", cSpaceDim+1, tNElems);

    auto tCoords = tMesh->coords();
    Omega_h::LOs tCells2Nodes = tMesh->ask_elem_verts();

    // map from input to output
    //
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, tNElems), LAMBDA_EXPRESSION(int aOrdinal)
    {
        Plato::Scalar tElemBases[cSpaceDim+1];
        // compute element centroid
        for(int iVert=0; iVert<(cSpaceDim+1); iVert++)
        {
            auto iVertOrdinal = tCells2Nodes[aOrdinal*(cSpaceDim+1)+iVert];
            for(int iDim=0; iDim<cSpaceDim; iDim++)
            {
                tInPoints(iDim, aOrdinal) += tCoords[iVertOrdinal*cSpaceDim+iDim];
            }
        }
        for(int iDim=0; iDim<cSpaceDim; iDim++)
        {
            tInPoints(iDim, aOrdinal) /= cSpaceDim+1;
        }
        tGetBasis(tInPoints, aOrdinal, aOrdinal, tElemBases);
        for(int iVert=0; iVert<(cSpaceDim+1); iVert++)
        {
            tBases(iVert, aOrdinal) = tElemBases[iVert];
        }
    }, "compute");

    double tol_double = 1e-14;
    auto tBases_host = get(tBases);
    for(int iElem=0; iElem<tNElems; iElem++)
    {
        for(int iNode=0; iNode<(cSpaceDim+1); iNode++)
        {
            TEST_FLOATING_EQUALITY(tBases_host(iNode, iElem), 1.0/4.0, tol_double);
        }
    }
  }
  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, SymmetryPlane)
  {

    // create input for SymmetryPlane
    //
    double rx = 0.1, ry = 0.2, rz = 0.3;
    double nx = 0.0, ny = 0.0, nz = 1.0;

    std::stringstream input;
    input << "<LinearMap>";
    input << "  <Type>SymmetryPlane</Type>";
    input << "  <Origin>";
    input << "    <X>" << rx << "</X>";
    input << "    <Y>" << ry << "</Y>";
    input << "    <Z>" << rz << "</Z>";
    input << "  </Origin>";
    input << "  <Normal>";
    input << "    <X>" << nx << "</X>";
    input << "    <Y>" << ny << "</Y>";
    input << "    <Z>" << nz << "</Z>";
    input << "  </Normal>";
    input << "</LinearMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create SymmetryPlane from input
    //
    auto tMathMapParams = tInputData.get<Plato::InputData>("LinearMap");
    Plato::Geometry::SymmetryPlane<Plato::Scalar> tMathMap(tMathMapParams);

    // create input and output views
    //
    int tNumVals = 2;
    int tNumDims = 3;
    Plato::ScalarMultiVector tXin("Xin", tNumDims, tNumVals);
    Plato::ScalarMultiVector tXout("Xin", tNumDims, tNumVals);
    auto tXin_host = Kokkos::create_mirror_view(tXin);

    double p0_X = 0.0, p0_Y = 0.0, p0_Z = 0.0;
    double p1_X = 0.0, p1_Y = 0.0, p1_Z = 0.5;

    tXin_host(0,0) = p0_X; tXin_host(1,0) = p0_Y; tXin_host(2,0) = p0_Z;
    tXin_host(0,1) = p1_X; tXin_host(1,1) = p1_Y; tXin_host(2,1) = p1_Z;
    Kokkos::deep_copy(tXin, tXin_host);

    // map from input to output
    //
    Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, tNumVals), LAMBDA_EXPRESSION(int aOrdinal)
    {
        tMathMap(aOrdinal, tXin, tXout);
    }, "compute");

    // test results
    //
    auto tXout_host = Kokkos::create_mirror_view(tXout);
    Kokkos::deep_copy(tXout_host, tXout);

    double tol_double = 1e-14;
    TEST_FLOATING_EQUALITY(/*Gold=*/ p0_X, /*Result=*/ tXout_host(0,0), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p0_Y, /*Result=*/ tXout_host(1,0), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p0_Z-2.0*(p0_Z-rz)*nz, /*Result=*/ tXout_host(2,0), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p1_X, /*Result=*/ tXout_host(0,1), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p1_Y, /*Result=*/ tXout_host(1,1), tol_double);
    TEST_FLOATING_EQUALITY(/*Gold=*/ p1_Z, /*Result=*/ tXout_host(2,1), tol_double);
  }


/******************************************************************************/
/*!
  \brief Enforce symmetry on a linear field on an asymmetric tet mesh.

  The linear tet mesh, while asymmetric, can approximate the symmetrized linear
  field accurately.  The test constructs a MeshMap with a SymmetryPlane:

    f(p(z)) = p(z)   if z >= 0.5
              p(1-z) if z < 0.5

  then constructs a field p(z) = z in {0.0,1.0} and applies the MeshMap, f(p)

  test passes if:
    f(p(z)) == z   for z > 0.5
    f(p(z)) == 1-z for z < 0.5
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, MeshMap)
  {

    // create input for MeshMap
    //
    double rx = 0.0, ry = 0.0, rz = 0.5;
    double nx = 0.0, ny = 0.0, nz = 1.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "      <Z>" << rz << "</Z>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "      <Z>" << nz << "</Z>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=5;
    constexpr int cSpaceDim=3;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(*tMesh, tMeshMapParams);

    auto tCoords = tMesh->coords();
    auto tNVerts = tMesh->nverts();

    auto tDim = tMesh->dim();
    Kokkos::View<double*, MemSpace> tInField("not symmetric", tNVerts);
    using OrdinalType = typename Kokkos::View<double*, MemSpace>::size_type;
    Kokkos::parallel_for(Kokkos::RangePolicy<OrdinalType>(0, tNVerts), LAMBDA_EXPRESSION(OrdinalType iVertOrdinal)
    {
        tInField(iVertOrdinal) = tCoords[iVertOrdinal*tDim+2];
    }, "compute field");

    Kokkos::View<double*, MemSpace> tOutField("symmetric", tNVerts);
    tMeshMap->apply(tInField, tOutField);

    auto tOutField_host = Kokkos::create_mirror_view(tOutField);
    Kokkos::deep_copy(tOutField_host, tOutField);

    auto tInField_host = Kokkos::create_mirror_view(tInField);
    Kokkos::deep_copy(tInField_host, tInField);

    double tol_double = 1e-12;
    for(OrdinalType i=0; i<tNVerts; i++)
    {
        if(tInField_host(i) > 1e-15)
        {
            if(tInField_host(i) < rz )
            {
                TEST_FLOATING_EQUALITY(2*rz-tInField_host(i), tOutField_host(i), tol_double);
            }
            else
            if(tInField_host(i) > rz )
            {
                TEST_FLOATING_EQUALITY(tInField_host(i), tOutField_host(i), tol_double);
            }
            else
            if(tInField_host(i) == 0.0 )
            {
                TEST_ASSERT(tOutField_host(i) == 0.0);
            }
        }
    }
  }
}

/******************************************************************************/
/*!
  \brief Enforce symmetry on a uniform field on an asymmetric tet mesh.

  The test constructs a MeshMap with a SymmetryPlane:

    f(p(z)) = p(z)   if z >= 0.5
              p(1-z) if z < 0.5

  then constructs a field p(z) = 1 in {-0.5,0.5} and applies the MeshMap, f(p),
  as well as a linear filter, F.

  test passes if:
    F(f(p(z))) == 1 for all z
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, MeshMapWFilter)
  {

    // create input for MeshMap
    //
    double rx = 0.0, ry = 0.0, rz = 0.5;
    double nx = 0.0, ny = 0.0, nz = 1.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <Filter>";
    input << "    <Type>Linear</Type>";
    input << "    <Radius>0.25</Radius>";
    input << "  </Filter>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "      <Z>" << rz << "</Z>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "      <Z>" << nz << "</Z>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=5;
    constexpr int cSpaceDim=3;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(*tMesh, tMeshMapParams);

    auto tCoords = tMesh->coords();
    auto tNVerts = tMesh->nverts();

    auto tDim = tMesh->dim();
    Kokkos::View<double*, MemSpace> tInField("uniform", tNVerts);
    Kokkos::View<double*, MemSpace> tOutField("also uniform", tNVerts);
    Kokkos::deep_copy(tInField, 1.0);

    tMeshMap->apply(tInField, tOutField);

    auto tOutField_host = Kokkos::create_mirror_view(tOutField);
    Kokkos::deep_copy(tOutField_host, tOutField);

    auto tInField_host = Kokkos::create_mirror_view(tInField);
    Kokkos::deep_copy(tInField_host, tInField);

    double tol_double = 1e-12;
    using OrdinalType = typename Kokkos::View<double*, MemSpace>::size_type;
    for(OrdinalType i=0; i<tNVerts; i++)
    {
        TEST_FLOATING_EQUALITY(1.0, tOutField_host(i), tol_double);
    }
  }

/******************************************************************************/
/*!
  \brief Test createTranspose() function in Plato::MeshMap.

  The test constructs a MeshMap with a SymmetryPlane:

    f(p(z)) = p(z)   if z >= 0.5
              p(1-z) if z < 0.5

  The map, f, and filter, F, are computed during construction.

  test passes if:
    (f^T)_{ij} = f_{ji}      : Transpose works
    (F^T)_{ij} = F_{ji}      : Transpose works
     F_{ii} = I              : Filter matrix rows sum to one
     F_{ij}!=0 if F_{ji}!=0  : Filter graph is symmetric
*/
/******************************************************************************/

  TEUCHOS_UNIT_TEST(PlatoTestMeshMap, TransposeMatrix)
  {

    // create input for MeshMap
    //
    double rx = 0.0, ry = 0.0, rz = 0.5;
    double nx = 0.0, ny = 0.0, nz = 1.0;

    std::stringstream input;
    input << "<MeshMap>";
    input << "  <Filter>";
    input << "    <Type>Linear</Type>";
    input << "    <Radius>0.25</Radius>";
    input << "  </Filter>";
    input << "  <LinearMap>";
    input << "    <Type>SymmetryPlane</Type>";
    input << "    <Origin>";
    input << "      <X>" << rx << "</X>";
    input << "      <Y>" << ry << "</Y>";
    input << "      <Z>" << rz << "</Z>";
    input << "    </Origin>";
    input << "    <Normal>";
    input << "      <X>" << nx << "</X>";
    input << "      <Y>" << ny << "</Y>";
    input << "      <Z>" << nz << "</Z>";
    input << "    </Normal>";
    input << "  </LinearMap>";
    input << "</MeshMap>";

    Plato::Parser* parser = new Plato::PugiParser();
    auto tInputData = parser->parseString(input.str());
    delete parser;

    // create MeshMap from input
    //
    auto tMeshMapParams = tInputData.get<Plato::InputData>("MeshMap");

    constexpr int cMeshWidth=3;
    constexpr int cSpaceDim=3;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

    Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
    auto tMeshMap = tMeshMapFactory.create(*tMesh, tMeshMapParams);

    auto tMatrix  = toFull(tMeshMap->mMatrix);
    auto tMatrixT = toFull(tMeshMap->mMatrixT);

    double tol_double = 1e-12;
    for(int i=0; i<tMatrix.size(); i++)
    {
        for(int j=0; j<tMatrix[i].size(); j++)
        {
            TEST_FLOATING_EQUALITY(tMatrix[i][j], tMatrixT[j][i], tol_double);
        }
    }

    auto tFilter  = toFull(*(tMeshMap->mFilter));
    auto tFilterT = toFull(*(tMeshMap->mFilterT));

    std::vector<Plato::Scalar> tRowSum(tFilter.size());
    for(int i=0; i<tFilter.size(); i++)
    {
        tRowSum[i] = 0.0;
        for(int j=0; j<tFilter[i].size(); j++)
        {
            tRowSum[i] += tFilter[i][j];
            TEST_FLOATING_EQUALITY(tFilter[i][j], tFilterT[j][i], tol_double);
            if( tFilter[i][j] != 0.0 )
                TEST_ASSERT(tFilter[j][i] != 0.0);
        }
        TEST_FLOATING_EQUALITY(tRowSum[i], 1.0, tol_double);
    }

    auto tMatrixTT = tMeshMap->createTranspose(tMeshMap->mMatrixT);
    auto tMatrixTTF = toFull(tMatrixTT);

    for(int i=0; i<tMatrix.size(); i++)
    {
        for(int j=0; j<tMatrix[i].size(); j++)
        {
            TEST_FLOATING_EQUALITY(tMatrix[i][j], tMatrixTTF[i][j], tol_double);
        }
    }
  }


std::vector<std::vector<Plato::Scalar>>
toFull( SparseMatrix aInMatrix )
{
    using OrdinalType = Plato::Geometry::AbstractMeshMap<Plato::Scalar>::OrdinalT;
    using Plato::Scalar;

    std::vector<std::vector<Scalar>>
        retMatrix(aInMatrix.mNumRows, std::vector<Scalar>(aInMatrix.mNumCols, 0.0));

    auto tRowMap = get(aInMatrix.mRowMap);
    auto tColMap = get(aInMatrix.mColMap);
    auto tValues = get(aInMatrix.mEntries);

    auto tNumRows = aInMatrix.mNumRows;
    for(OrdinalType iRowIndex=0; iRowIndex<tNumRows; iRowIndex++)
    {
        auto tFrom = tRowMap(iRowIndex);
        auto tTo   = tRowMap(iRowIndex+1);
        for(auto iEntryIndex=tFrom; iEntryIndex<tTo; iEntryIndex++)
        {
            auto iColIndex = tColMap(iEntryIndex);
            retMatrix[iRowIndex][iColIndex] = tValues(iEntryIndex);
        }
    }
    return retMatrix;
}

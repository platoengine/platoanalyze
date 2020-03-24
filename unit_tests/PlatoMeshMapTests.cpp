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
#include "Plato_MeshMap.hpp"

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = typename ExecSpace::memory_space;
using DeviceType = Kokkos::Device<ExecSpace, MemSpace>;


namespace PlatoTestMeshMap
{
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
    Plato::Geometry::SymmetryPlane<Plato::ScalarMultiVector> tMathMap(tMathMapParams);

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

    f(p(z)) = p(z)  if z >= 0
              p(-z) if z < 0

  then constructs a field p(z) = z in {-0.5,0.5} and applies the MeshMap, f(p)

  test passes if:
    f(p(z)) == p(z) for z > 0.0 
    f(p(z)) ==-p(z) for z < 0.0 
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


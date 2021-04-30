/*
 *  PlatoSpatialModelTests.cpp
 *  
 *   Created on: Oct 9, 2020
 **/

#include <iostream>
#include <fstream>

#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoMask.hpp"
#include "SpatialModel.hpp"

namespace PlatoUnitTests
{


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoMask)
{
  // create mesh mask input
  //
  Teuchos::RCP<Teuchos::ParameterList> tMaskParams =
    Teuchos::getParametersFromXmlString(
    "     <ParameterList name='Mask'>                                 \n"
    "       <Parameter name='Mask Type' type='string' value='Brick'/> \n"
    "       <Parameter name='Maximum Z' type='double' value='0.5'/>   \n"
    "     </ParameterList>                                            \n"
  );

  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  Plato::BrickMask<spaceDim> tBrickMask(*tMesh, *tMaskParams);

  auto tCellMask = tBrickMask.cellMask();

  auto tCellMask_host = Kokkos::create_mirror_view( tCellMask );
  Kokkos::deep_copy( tCellMask_host, tCellMask );

  /* uncomment to write mesh file.  gold data below generate by inspecting mesh.
  Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer("Cube", &(*tMesh), spaceDim);
  tWriter.write();
  */
  std::vector<int> tCellMask_gold = {
    1, 1, 1, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1
  };

  for (int i=0; i<tCellMask_gold.size(); i++) 
  {
    TEST_ASSERT(tCellMask_gold[i] == tCellMask_host(i));
  }

  auto tInactive = tBrickMask.getInactiveNodes();
  auto tInactive_host = Kokkos::create_mirror_view( tInactive );
  Kokkos::deep_copy( tInactive_host, tInactive );

  std::vector<int> tInactive_gold = { 2, 4, 5, 14, 15, 18, 19, 22, 23 };

  for (int i=0; i<tInactive_gold.size(); i++) 
  {
    TEST_ASSERT(tInactive_gold[i] == tInactive_host(i));
  }

  auto tCellCenters = tBrickMask.getCellCenters(*tMesh);
  auto tCellCenters_host = Kokkos::create_mirror_view( tCellCenters );
  Kokkos::deep_copy( tCellCenters_host, tCellCenters );

}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PlatoModel)
{
  // create mesh mask input
  //
  Teuchos::RCP<Teuchos::ParameterList> tMaskParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Layer 1'>                                           \n"
    "  <ParameterList name='Mask'>                                            \n"
    "    <Parameter name='Mask Type' type='string' value='Brick'/>            \n"
    "    <Parameter name='Maximum Z' type='double' value='0.5'/>              \n"
    "  </ParameterList>                                                       \n"
    "</ParameterList>                                                         \n"
  );

  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                     \n"
    "  <ParameterList name='Spatial Model'>                                   \n"
    "    <ParameterList name='Domains'>                                       \n"
    "      <ParameterList name='Design Volume'>                               \n"
    "        <Parameter name='Element Block' type='string' value='body'/>     \n"
    "        <Parameter name='Material Model' type='string' value='matl'/>    \n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "  <ParameterList name='Material Models'>                                 \n"
    "    <ParameterList name='matl'>                                          \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                    \n"
    "        <Parameter name='Mass Density' type='double' value='0.0'/>       \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.36'/>    \n"
    "        <Parameter name='Youngs Modulus' type='double' value='68.0e10'/> \n"
    "      </ParameterList>                                                   \n"
    "    </ParameterList>                                                     \n"
    "  </ParameterList>                                                       \n"
    "</ParameterList>                                                         \n"
  );

  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tInputParams);

  Plato::MaskFactory<spaceDim> tMaskFactory;
  auto tMask = tMaskFactory.create(tSpatialModel.Mesh, *tMaskParams);

  tSpatialModel.applyMask(tMask);

  auto tOrdinals = tSpatialModel.Domains[0].cellOrdinals();

  auto tOrdinals_host = Kokkos::create_mirror_view( tOrdinals );
  Kokkos::deep_copy( tOrdinals_host, tOrdinals );

  std::vector<int> tOrdinals_gold = {
    0, 1, 2, 3, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23,
    40, 41, 42, 43, 44, 45, 46, 47
  };

  for (int i=0; i<tOrdinals_gold.size(); i++) 
  {
    TEST_ASSERT(tOrdinals_gold[i] == tOrdinals_host(i));
  }
}

} // namespace PlatoUnitTests

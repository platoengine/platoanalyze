/*!
  These unit tests are for the HeatEquation functionality.
 \todo 
*/

#include "PlatoTestHelpers.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "Omega_h_file.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "Solutions.hpp"
#include "ImplicitFunctors.hpp"
#include "ThermalConductivityMaterial.hpp"
#include "ThermalMassMaterial.hpp"

#ifdef HAVE_AMGX
#include "alg/AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>

#include <Sacado.hpp>
#include <alg/CrsLinearProblem.hpp>
#include <alg/ParallelComm.hpp>
#include <Simp.hpp>
#include <ScalarProduct.hpp>
#include <SimplexFadTypes.hpp>
#include <WorksetBase.hpp>
#include <parabolic/VectorFunction.hpp>
#include <parabolic/PhysicsScalarFunction.hpp>
#include <StateValues.hpp>
#include "ApplyConstraints.hpp"
#include "SimplexThermal.hpp"
#include "Thermal.hpp"
#include "ComputedField.hpp"

#include <fenv.h>


TEUCHOS_UNIT_TEST( HeatEquationTests, 3D )
{ 
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Thermal Mass'>                                   \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>          \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>       \n"
    "      </ParameterList>                                                      \n"
    "      <ParameterList name='Thermal Conduction'>                             \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0e6'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);


  int numCells = tMesh->nelems();
  constexpr int nodesPerCell  = Plato::SimplexThermal<spaceDim>::mNumNodesPerCell;
  constexpr int dofsPerCell   = Plato::SimplexThermal<spaceDim>::mNumDofsPerCell;
  constexpr int dofsPerNode   = Plato::SimplexThermal<spaceDim>::mNumDofsPerNode;

  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> T_host( tMesh->nverts() );
  Plato::Scalar Tval = 0.0, dval = 1.0;
  for( auto& val : T_host ) val = (Tval += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    T_host_view(T_host.data(),T_host.size());
  auto T = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), T_host_view);


  Plato::WorksetBase<Plato::SimplexThermal<spaceDim>> worksetBase(*tMesh);

  Plato::ScalarArray3DT<Plato::Scalar>
    gradient("gradient",numCells,nodesPerCell,spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    tGrad("temperature gradient", numCells, spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    tFlux("thermal flux", numCells, spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    result("result", numCells, dofsPerCell);

  Plato::ScalarArray3DT<Plato::Scalar>
    configWS("config workset",numCells, nodesPerCell, spaceDim);
  worksetBase.worksetConfig(configWS);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    stateWS("state workset",numCells, dofsPerCell);
  worksetBase.worksetState(T, stateWS);

  Plato::ComputeGradientWorkset<spaceDim> computeGradient;

  Plato::ScalarGrad<spaceDim> scalarGrad;



  Plato::ThermalMassModelFactory<spaceDim> mmmfactory(*tParamList);
  auto thermalMassMaterialModel = mmmfactory.create("Unobtainium");

  Plato::ThermalConductionModelFactory<spaceDim> mmfactory(*tParamList);
  auto tMaterialModel = mmfactory.create("Unobtainium");

  Plato::ThermalFlux<spaceDim>      thermalFlux(tMaterialModel);
  Plato::FluxDivergence<spaceDim>  fluxDivergence;

  Plato::LinearTetCubRuleDegreeOne<spaceDim> cubatureRule;

  Plato::Scalar quadratureWeight = cubatureRule.getCubWeight();

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",numCells);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    computeGradient(cellOrdinal, gradient, configWS, cellVolume);
    cellVolume(cellOrdinal) *= quadratureWeight;

    scalarGrad(cellOrdinal, tGrad, stateWS, gradient);

    thermalFlux(cellOrdinal, tFlux, tGrad);

    fluxDivergence(cellOrdinal, result, tFlux, gradient, cellVolume);
  }, "flux divergence");


  Plato::ScalarVectorT<Plato::Scalar> 
   tTemperature("Gauss point temperature at step k", numCells);

  Plato::ScalarVectorT<Plato::Scalar> 
    thermalContent("Gauss point heat content at step k", numCells);

  Plato::ScalarMultiVectorT<Plato::Scalar> 
    massResult("mass", numCells, dofsPerCell);

  Plato::StateValues computeStateValues;

  Plato::InterpolateFromNodal<spaceDim, dofsPerNode> interpolateFromNodal;
  Plato::ThermalContent<spaceDim> computeThermalContent(thermalMassMaterialModel);
  Plato::ProjectToNode<spaceDim, dofsPerNode> projectThermalContent;

  auto basisFunctions = cubatureRule.getBasisFunctions();

  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    interpolateFromNodal(cellOrdinal, basisFunctions, stateWS, tTemperature);
    computeThermalContent(cellOrdinal, thermalContent, tTemperature);
    projectThermalContent(cellOrdinal, cellVolume, basisFunctions, thermalContent, massResult);

  }, "mass");

  
  // test cell volume
  //
  auto cellVolume_Host = Kokkos::create_mirror_view( cellVolume );
  Kokkos::deep_copy( cellVolume_Host, cellVolume );

  std::vector<Plato::Scalar> cellVolume_gold = { 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 
    0.0208333333333333, 0.0208333333333333, 0.0208333333333333, 0.0208333333333333
  };

  int numGoldCells=cellVolume_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    if(cellVolume_gold[iCell] == 0.0){
      TEST_ASSERT(fabs(cellVolume_Host(iCell)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(cellVolume_Host(iCell), cellVolume_gold[iCell], 1e-13);
    }
  }

  // test state values
  //
  auto tTemperature_Host = Kokkos::create_mirror_view( tTemperature );
  Kokkos::deep_copy( tTemperature_Host, tTemperature );

  std::vector<Plato::Scalar> tTemperature_gold = { 
     8.5 ,  7.0 , 5.25,  9.75,
     8.75,  6.5 , 6.25, 10.75,
    11.00,  8.25, 7.75, 10.0 ,
    11.75, 10.25, 9.25, 11.0 
  };

  numGoldCells=tTemperature_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    if(tTemperature_gold[iCell] == 0.0){
      TEST_ASSERT(fabs(tTemperature_Host(iCell)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tTemperature_Host(iCell), tTemperature_gold[iCell], 1e-13);
    }
  }

  // test thermal content
  //
  auto thermalContent_Host = Kokkos::create_mirror_view( thermalContent );
  Kokkos::deep_copy( thermalContent_Host, thermalContent );

  std::vector<Plato::Scalar> thermalContent_gold = { 
     2.550e6, 2.100e6, 1.575e6, 2.925e6,
     2.625e6, 1.950e6, 1.875e6, 3.225e6,
     3.300e6, 2.475e6, 2.325e6, 3.000e6,
     3.525e6, 3.075e6, 2.775e6, 3.300e6
  };

  numGoldCells=thermalContent_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
      if(thermalContent_gold[iCell] == 0.0){
        TEST_ASSERT(fabs(thermalContent_Host(iCell)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(thermalContent_Host(iCell), thermalContent_gold[iCell], 1e-13);
      }
  }

  // test gradient operator
  //
  auto gradient_Host = Kokkos::create_mirror_view( gradient );
  Kokkos::deep_copy( gradient_Host, gradient );

  std::vector<std::vector<std::vector<Plato::Scalar>>> gradient_gold = { 
    {{0.0,-2.0, 0.0},{ 2.0, 0.0,-2.0},{-2.0, 2.0, 0.0},{ 0.0, 0.0, 2.0}},
    {{0.0,-2.0, 0.0},{ 0.0, 2.0,-2.0},{-2.0, 0.0, 2.0},{ 2.0, 0.0, 0.0}},
    {{0.0, 0.0,-2.0},{-2.0, 2.0, 0.0},{ 0.0,-2.0, 2.0},{ 2.0, 0.0, 0.0}},
    {{0.0, 0.0,-2.0},{ 2.0,-2.0, 0.0},{ 0.0, 2.0, 0.0},{-2.0, 0.0, 2.0}},
    {{0.0,-2.0, 0.0},{ 2.0, 0.0,-2.0},{-2.0, 2.0, 0.0},{ 0.0, 0.0, 2.0}},
    {{0.0,-2.0, 0.0},{ 0.0, 2.0,-2.0},{-2.0, 0.0, 2.0},{ 2.0, 0.0, 0.0}}
  };

  numGoldCells=gradient_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    for(int iNode=0; iNode<spaceDim+1; iNode++){
      for(int iDim=0; iDim<spaceDim; iDim++){
        if(gradient_gold[iCell][iNode][iDim] == 0.0){
          TEST_ASSERT(fabs(gradient_Host(iCell,iNode,iDim)) < 1e-12);
        } else {
          TEST_FLOATING_EQUALITY(gradient_Host(iCell,iNode,iDim), gradient_gold[iCell][iNode][iDim], 1e-13);
        }
      }
    }
  }

  // test temperature gradient
  //
  auto tgrad_Host = Kokkos::create_mirror_view( tGrad );
  Kokkos::deep_copy( tgrad_Host, tGrad );

  std::vector<std::vector<Plato::Scalar>> tgrad_gold = { 
    { 2.000 , 16.00, 8.000 },
    { 20.00 , 16.00,-10.00 },
    { 20.00 , 4.000, 2.000 },
    { 40.00 ,-16.00, 2.000 },
    { 20.00 , 4.000, 2.000 },
    { 20.00 , 4.000, 2.000 }
  };

  for(int iCell=0; iCell<int(tgrad_gold.size()); iCell++){
    for(int iDim=0; iDim<spaceDim; iDim++){
      if(tgrad_gold[iCell][iDim] == 0.0){
        TEST_ASSERT(fabs(tgrad_Host(iCell,iDim)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tgrad_Host(iCell,iDim), tgrad_gold[iCell][iDim], 1e-13);
      }
    }
  }

  // test thermal flux
  //
  auto tflux_Host = Kokkos::create_mirror_view( tFlux );
  Kokkos::deep_copy( tflux_Host, tFlux );

  std::vector<std::vector<Plato::Scalar>> tflux_gold = { 
   {-2.0e6,-1.6e7,-8.0e6 },
   {-2.0e7,-1.6e7, 1.0e7 },
   {-2.0e7,-4.0e6,-2.0e6 },
   {-4.0e7, 1.6e7,-2.0e6 },
   {-2.0e7,-4.0e6,-2.0e6 },
   {-2.0e7,-4.0e6,-2.0e6 }
  };

  for(int iCell=0; iCell<int(tflux_gold.size()); iCell++){
    for(int iDim=0; iDim<spaceDim; iDim++){
      if(tflux_gold[iCell][iDim] == 0.0){
        TEST_ASSERT(fabs(tflux_Host(iCell,iDim)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(tflux_Host(iCell,iDim), tflux_gold[iCell][iDim], 1e-13);
      }
    }
  }

  // test residual
  //
  auto result_Host = Kokkos::create_mirror_view( result );
  Kokkos::deep_copy( result_Host, result );

  std::vector<std::vector<Plato::Scalar>> result_gold = { 
   {  666666.6666666666,  250000.0000000000, -583333.3333333333, -333333.3333333333 },
   {  666666.6666666666,-1083333.3333333333, 1250000.0000000000, -833333.3333333333 },
   {   83333.3333333333,  666666.6666666666,   83333.3333333333, -833333.3333333333 },
   {   83333.3333333333,-2333333.3333333333,  666666.6666666666, 1583333.3333333333 },
   {  166666.6666666667, -750000.0000000000,  666666.6666666666,  -83333.3333333333 },
   {  166666.6666666667,  -83333.3333333333,  750000.0000000000, -833333.3333333333 }
  };

  for(int iCell=0; iCell<int(result_gold.size()); iCell++){
    for(int iDof=0; iDof<dofsPerCell; iDof++){
      if(result_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(result_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(result_Host(iCell,iDof), result_gold[iCell][iDof], 1e-13);
      }
    }
  }

  // test residual
  //
  auto mass_result_Host = Kokkos::create_mirror_view( massResult );
  Kokkos::deep_copy( mass_result_Host, massResult );

  std::vector<std::vector<Plato::Scalar>> mass_result_gold = { 
   { 13281.25000000000, 13281.25000000000, 13281.25000000000, 13281.25000000000},
   { 10937.50000000000, 10937.50000000000, 10937.50000000000, 10937.50000000000},
   {  8203.12500000000,  8203.12500000000,  8203.12500000000,  8203.12500000000},
   { 15234.37500000000, 15234.37500000000, 15234.37500000000, 15234.37500000000},
   { 13671.87500000000, 13671.87500000000, 13671.87500000000, 13671.87500000000}
  };

  for(int iCell=0; iCell<int(mass_result_gold.size()); iCell++){
    for(int iNode=0; iNode<nodesPerCell; iNode++){
      if(mass_result_gold[iCell][iNode] == 0.0){
        TEST_ASSERT(fabs(mass_result_Host(iCell,iNode)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(mass_result_Host(iCell,iNode), mass_result_gold[iCell][iNode], 1e-13);
      }
    }
  }
}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         ElastostaticResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HeatEquationTests, HeatEquationResidual3D )
{
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Parabolic'/>        \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                \n"
    "  <ParameterList name='Parabolic'>                                          \n"
    "    <ParameterList name='Penalty Function'>                                 \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Thermal Mass'>                                   \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>          \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e3'/>       \n"
    "      </ParameterList>                                                      \n"
    "      <ParameterList name='Thermal Conduction'>                             \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0e6'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Time Integration'>                                   \n"
    "    <Parameter name='Number Time Steps' type='int' value='3'/>              \n"
    "    <Parameter name='Time Step' type='double' value='0.5'/>                 \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( tMesh->nverts(), 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);


  // create mesh based temperature from host data
  //
  std::vector<Plato::Scalar> T_host( tMesh->nverts() );
  Plato::Scalar Tval = 0.0, dval = 1.0000;
  for( auto& val : T_host ) val = (Tval += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    T_host_view(T_host.data(),T_host.size());
  auto T = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), T_host_view);

  std::vector<Plato::Scalar> Tdot_host( tMesh->nverts() );
  Tval = 0.0; dval = 0.5000;
  for( auto& val : Tdot_host ) val = (Tval += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    Tdot_host_view(Tdot_host.data(),Tdot_host.size());
  auto Tdot = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), Tdot_host_view);


  // create constraint evaluator
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::DataMap tDataMap;
  Plato::Parabolic::VectorFunction<::Plato::Thermal<spaceDim>>
    vectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));


  // compute and test value
  //
  auto timeStep = tParamList->sublist("Time Integration").get<Plato::Scalar>("Time Step");
  auto residual = vectorFunction.value(T, Tdot, z, timeStep);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<Plato::Scalar> residual_gold = { 
    -5.666620572916666e6, -5.499940625000000e6, -1.666653385416667e6,
    -6.749914257812499e6, -2.166642838541667e6, -1.666654166666667e6,
    -1.749969726562500e6, -4.999828125000000e5,  2.500064453125001e6,
    -6.749869726562500e6, -3.666615104166667e6, -9.999796874999998e5,
     2.500404296875000e5, -2.499735546875003e6, -4.998796874999979e5,
     1.166740494791667e6,  5.000119921874999e6,  3.334278645833342e5,
     5.000804687500000e5,  5.001246093749986e5,  6.000182421875003e6,
     7.000173828125002e6,  3.833386848958333e6,  6.667018229166672e5,
     1.666739322916666e6,  9.000119921875000e6,  1.666702604166667e6
  };

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-13);
    }
  }


  // compute and test gradient wrt state, T. (i.e., jacobian)
  //
  auto jacobian = vectorFunction.gradient_u(T, Tdot, z, timeStep);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
    4.999999999999999e5, -1.666666666666667e5, -1.666666666666667e5,
    0.000000000000000,    0.000000000000000,    0.000000000000000,
    0.000000000000000,   -1.666666666666667e5, -1.666666666666667e5,
    8.333333333333334e5, -2.500000000000000e5, -1.666666666666667e5,
    0.000000000000000,    0.000000000000000,    0.000000000000000,
   -2.500000000000000e5,  0.000000000000000,   -1.666666666666667e5,
    3.333333333333333e5, -8.333333333333333e4,  0.000000000000000,
   -8.333333333333333e4,  0.000000000000000,   -2.500000000000000e5,
    1.500000000000000e6, -2.500000000000000e5, -2.500000000000000e5,
    0.000000000000000,   -2.500000000000000e5, -4.999999999999999e5,
    0.000000000000000,    0.000000000000000,    0.000000000000000
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  // compute and test gradient wrt previous state, T. (i.e., jacobian)
  //
  auto jacobian_v = vectorFunction.gradient_v(T, Tdot, z, timeStep);

  auto jac_v_entries = jacobian_v->entries();
  auto jac_v_entriesHost = Kokkos::create_mirror_view( jac_v_entries );
  Kokkos::deep_copy(jac_v_entriesHost, jac_v_entries);

  std::vector<Plato::Scalar> gold_jac_v_entries = {
    2.343750000000000,    7.812500000000000e-1,  7.812500000000000e-1,
    7.812500000000000e-1, 7.812500000000000e-1,  2.343750000000000,
    7.812500000000000e-1, 7.812500000000000e-1,  7.812500000000000e-1,
    3.125000000000000,    1.171875000000000,     7.812500000000000e-1,
    7.812500000000000e-1, 1.562500000000000,     2.343750000000000,
    1.171875000000000,    7.812500000000000e-1,  7.812500000000000e-1,
    7.812500000000000e-1, 3.906250000000000e-1,  7.812500000000000e-1,
    3.906250000000000e-1, 7.812500000000000e-1,  1.171875000000000,
    4.687500000000000,    1.171875000000000,     1.171875000000000,
    7.812500000000000e-1, 1.171875000000000,     2.343750000000000,
    1.562500000000000,    2.343750000000000,     1.562500000000000
  };

  int jac_v_entriesSize = gold_jac_v_entries.size();
  for(int i=0; i<jac_v_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_v_entriesHost(i), gold_jac_v_entries[i], 1.0e-15);
  }


  // compute and test objective gradient wrt control, z
  //
  auto gradient_z = vectorFunction.gradient_z(T, Tdot, z, timeStep);
  
  auto grad_entries = gradient_z->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
   -1.41665514322916651e6,  4.16669694010416744e5, -4.16663736979166628e5,
   -4.79164274088541628e5, -4.79162516276041628e5,  2.50011523437499884e5,
    6.66671647135416744e5,  1.45833889973958326e6, -4.16637369791666642e4,
   -1.37498515625000000e6, -3.12495996093749884e5, -4.16663346354166628e5,
   -3.54164176432291686e5,  4.16738932291665697e4,  1.25011914062500175e5,
    1.56250751953125000e6,  7.70838460286458256e5, -4.16633463541666715e4,
   -4.16663346354166628e5, -1.66665445963541686e5,  4.16699869791667152e4,
    5.83335432942708256e5, -1.87497607421875000e5, -1.04162662760416657e5,
   -1.68747856445312477e6, -2.08327766927083314e5, -3.74995263671875000e5,
   -4.16663541666666628e5,  4.79171988932291744e5,  1.08334446614583302e6,
    4.16673990885416802e5,  4.16678776041666628e5,  5.83341927083333256e5
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 2.0e-14);
  }


  // compute and test objective gradient wrt node position, x
  //
  auto gradient_x = vectorFunction.gradient_x(T, Tdot, z, timeStep);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<Plato::Scalar> gold_grad_x_entries = {
   -1.516671119791666e7,  2.333309114583333e6, -1.666901041666665e5,
    4.999999999999998e5,  1.999975781250000e6, -1.500000000000000e6,
    4.833333333333333e6, -1.166666666666667e6,  3.999976562500000e6,
   -1.500000000000000e6,  2.666655729166667e6,  2.666658463541666e6,
    4.666646744791666e6,  4.666653385416666e6, -5.500000000000000e6,
    2.166622135416667e6,  1.999975781250000e6,  5.166643229166667e6,
   -5.000246093750003e5, -4.333333333333333e6, -5.000152343750005e5,
    4.999955468749998e6, -8.166666666666666e6, -4.166666666666667e6,
    5.166651432291666e6, -8.333415364583334e5, -3.333309895833333e6,
   -2.500006015625000e7,  4.999967968750000e6, -1.666669791666667e6
  };

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-13);
  }

}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalthermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HeatEquationTests, InternalThermalEnergy3D )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <ParameterList name='Spatial Model'>                                      \n"
    "    <ParameterList name='Domains'>                                          \n"
    "      <ParameterList name='Design Volume'>                                  \n"
    "        <Parameter name='Element Block' type='string' value='body'/>        \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Parabolic'/>        \n"
    "  <Parameter name='Objective' type='string' value='My Internal Thermal Energy'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                 \n"
    "  <ParameterList name='Criteria'>                                          \n"
    "    <ParameterList name='Internal Energy'>                                 \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>       \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Thermal Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                              \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>             \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>        \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                \n"
    "      </ParameterList>                                                     \n"
    "    </ParameterList>                                                       \n"
    "  </ParameterList>                                                         \n"
    "  <ParameterList name='Material Models'>                                    \n"
    "    <ParameterList name='Unobtainium'>                                      \n"
    "      <ParameterList name='Thermal Mass'>                                   \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>          \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e3'/>       \n"
    "      </ParameterList>                                                      \n"
    "      <ParameterList name='Thermal Conduction'>                             \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='1.0e6'/>\n"
    "      </ParameterList>                                                      \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Time Integration'>                                   \n"
    "    <Parameter name='Number Time Steps' type='int' value='3'/>              \n"
    "    <Parameter name='Time Step' type='double' value='0.5'/>                 \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);


  // create mesh based temperature from host data
  //
  int tNumSteps = 3;
  int tNumNodes = tMesh->nverts();
  Plato::ScalarMultiVector T("temperature history", tNumSteps, tNumNodes);
  Plato::ScalarMultiVector Tdot("temperature rate history", tNumSteps, tNumNodes);
  Plato::ScalarVector z("density", tNumNodes);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     for( int i=0; i<tNumSteps; i++){
       T(i, aNodeOrdinal) = (i+1)*aNodeOrdinal;
       Tdot(i, aNodeOrdinal) = 0.0;
     }
  }, "temperature history");


  // create objective
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Internal Energy");
  Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermal<spaceDim>>
    scalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);

  auto timeStep = tParamList->sublist("Time Integration").get<Plato::Scalar>("Time Step");
  int timeIncIndex = 1;

  // compute and test objective value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", T);
  tSolution.set("StateDot", Tdot);
  auto value = scalarFunction.value(tSolution, z, timeStep);

  Plato::Scalar value_gold = 7.95166666666666603e9;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradient_u(tSolution, z, timeIncIndex, timeStep);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
    -2.266666666666666e7, -2.200000000000000e7, -6.666666666666666e6, 
    -2.700000000000000e7, -8.666666666666666e6, -6.666666666666667e6, 
    -6.999999999999999e6, -2.000000000000001e6,  9.999999999999998e6, 
    -2.699999999999999e7, -1.466666666666667e7, -3.999999999999999e6, 
     1.000000000000002e6, -1.000000000000000e7, -2.000000000000007e6, 
     4.666666666666670e6,  2.000000000000000e7,  1.333333333333331e6, 
     2.000000000000002e6,  2.000000000000002e6,  2.399999999999999e7, 
     2.800000000000001e7,  1.533333333333333e7,  2.666666666666666e6, 
     6.666666666666670e6,  3.600000000000000e7,  6.666666666666666e6
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test objective gradient wrt control, z
  //
  auto grad_z = scalarFunction.gradient_z(tSolution, z, timeStep);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
    6.613750000000000e8, 6.175000000000000e8, 1.543750000000000e8, 
    3.813333333333334e8, 1.121250000000000e8, 5.741666666666666e7, 
    1.235000000000000e8, 3.900000000000000e7, 1.987916666666666e8, 
    5.492500000000000e8, 2.334583333333333e8, 2.275000000000000e7, 
    3.791666666666666e7, 1.168916666666667e9, 5.774166666666667e8, 
    1.803750000000000e8, 2.074583333333333e8, 1.056250000000000e8, 
    4.820833333333334e7, 1.197083333333333e8, 4.495833333333334e8, 
    7.133750000000000e8, 2.811250000000000e8, 1.191666666666667e7, 
    3.087500000000001e7, 7.659166666666666e8, 1.023750000000000e8
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradient_x(tSolution, z, timeStep);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
     2.457000000000000e9, -2.656333333333333e9, -1.261000000000000e9, 
     2.437500000000000e9, -2.710500000000000e9,  2.860000000000001e8, 
     6.825000000000000e8, -8.775000000000000e8,  1.321666666666667e9, 
     1.895833333333333e9,  1.124500000000000e9, -4.550000000000001e8, 
     4.506666666666666e8,  1.928333333333333e8,  7.821666666666667e8, 
     4.073333333333333e8,  4.030000000000000e8,  2.296666666666667e8, 
     1.603333333333333e8,  8.493333333333333e8, -2.296666666666667e8, 
     3.900000000000002e7,  2.860000000000000e8, -1.170000000000000e8, 
    -1.371500000000000e9, -5.156666666666666e8, -5.914999999999999e8, 
     2.223000000000000e9, -5.134999999999999e8, -1.180833333333333e9, 
     1.284833333333333e9, -1.852500000000000e9,  1.213333333333331e8, 
     9.100000000000001e7,  1.950000000000001e8,  1.299999999999999e8, 
    -4.333333333333340e6,  1.906666666666667e8, -2.903333333333335e8, 
     1.419166666666667e9,  4.289999999999999e8,  1.219833333333333e9, 
     6.825000000000000e8,  3.965000000000000e8,  1.759333333333333e9, 
    -3.943333333333335e8,  3.705000000000000e8,  4.918333333333333e8, 
    -1.232833333333334e9,  1.690000000000002e8, -1.928333333333334e8, 
     3.531666666666668e8,  2.665000000000002e8, -1.170000000000002e8, 
     4.333333333333328e6,  1.083333333333333e8,  1.430000000000000e8, 
     2.145000000000001e8,  1.083333333333333e8,  3.358333333333334e8, 
     6.933333333333373e7,  2.329166666666666e9, -3.228333333333337e8, 
    -4.454666666666666e9, -1.109333333333333e9,  3.293333333333337e8, 
    -2.433166666666667e9, -6.153333333333334e8,  1.042166666666667e9, 
    -2.166666666666649e7,  4.333333333333302e7,  4.766666666666681e7, 
    -4.116666666666666e7,  1.559999999999996e8,  1.755000000000002e8, 
    -5.650666666666667e9,  2.775500000000000e9, -2.901166666666666e9, 
     7.323333333333335e8,  4.571666666666666e8, -7.561666666666669e8
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}


/******************************************************************************/
/*! 
  \brief Create a 'ComputedField' object for a uniform scalar field
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( HeatEquationTests, ComputedField_UniformScalar )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  // compute fields
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                \n"
    "  <ParameterList name='Computed Fields'>                            \n"
    "    <ParameterList name='Uniform Initial Temperature'>              \n"
    "      <Parameter name='Function' type='string' value='100.0'/>      \n"
    "    </ParameterList>                                                \n"
    "    <ParameterList name='Linear X Initial Temperature'>             \n"
    "      <Parameter name='Function' type='string' value='1.0*x'/>      \n"
    "    </ParameterList>                                                \n"
    "    <ParameterList name='Linear Y Initial Temperature'>             \n"
    "      <Parameter name='Function' type='string' value='1.0*y'/>      \n"
    "    </ParameterList>                                                \n"
    "    <ParameterList name='Linear Z Initial Temperature'>             \n"
    "      <Parameter name='Function' type='string' value='1.0*z'/>      \n"
    "    </ParameterList>                                                \n"
    "    <ParameterList name='Bilinear XY Initial Temperature'>          \n"
    "      <Parameter name='Function' type='string' value='1.0*x*y'/>    \n"
    "    </ParameterList>                                                \n"
    "  </ParameterList>                                                  \n"
    "</ParameterList>                                                    \n"
  );

  auto tComputedFields = Plato::ComputedFields<spaceDim>(*tMesh, tParamList->sublist("Computed Fields"));

  int tNumNodes = tMesh->nverts();
  Plato::ScalarVector T("temperature", tNumNodes);

  tComputedFields.get("Uniform Initial Temperature", T);

  // pull temperature to host
  //
  auto T_Host = Kokkos::create_mirror_view( T );
  Kokkos::deep_copy( T_Host, T );

  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 100.0, 1e-15);
  }


  Plato::ScalarVector xcoords("x", tNumNodes);
  Plato::ScalarVector ycoords("y", tNumNodes);
  Plato::ScalarVector zcoords("z", tNumNodes);
  auto coords = tMesh->coords();
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(int nodeOrdinal)
  {
    xcoords(nodeOrdinal) = coords[nodeOrdinal*spaceDim+0];
    ycoords(nodeOrdinal) = coords[nodeOrdinal*spaceDim+1];
    zcoords(nodeOrdinal) = coords[nodeOrdinal*spaceDim+2];
  }, "get coords");

  auto xCoords_Host = Kokkos::create_mirror_view( xcoords );
  auto yCoords_Host = Kokkos::create_mirror_view( ycoords );
  auto zCoords_Host = Kokkos::create_mirror_view( zcoords );
  Kokkos::deep_copy( xCoords_Host, xcoords );
  Kokkos::deep_copy( yCoords_Host, ycoords );
  Kokkos::deep_copy( zCoords_Host, zcoords );

  tComputedFields.get("Linear X Initial Temperature", T);
  Kokkos::deep_copy( T_Host, T );
  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 1.0*xCoords_Host[iNode], 1e-15);
  }

  tComputedFields.get("Linear Y Initial Temperature", T);
  Kokkos::deep_copy( T_Host, T );
  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 1.0*yCoords_Host[iNode], 1e-15);
  }

  tComputedFields.get("Linear Z Initial Temperature", T);
  Kokkos::deep_copy( T_Host, T );
  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 1.0*zCoords_Host[iNode], 1e-15);
  }

  tComputedFields.get("Bilinear XY Initial Temperature", T);
  Kokkos::deep_copy( T_Host, T );
  for(int iNode=0; iNode<tNumNodes; iNode++){
    TEST_FLOATING_EQUALITY(T_Host[iNode], 1.0*xCoords_Host[iNode]*yCoords_Host[iNode], 1e-15);
  }
}

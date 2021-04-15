/*!
  These unit tests are for the TransientThermomech functionality.
 \todo 
*/

#include "PlatoTestHelpers.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "Omega_h_file.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "ImplicitFunctors.hpp"
#include "LinearThermoelasticMaterial.hpp"

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
#include <ApplyWeighting.hpp>
#include <SimplexFadTypes.hpp>
#include <WorksetBase.hpp>
#include <parabolic/VectorFunction.hpp>
#include <StateValues.hpp>
#include "ApplyConstraints.hpp"
#include "SimplexThermal.hpp"
#include "Thermomechanics.hpp"
#include "ComputedField.hpp"

#include <fenv.h>


TEUCHOS_UNIT_TEST( TransientThermomechTests, 3D )
{ 
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;

  static constexpr int TDofOffset = spaceDim;

  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);


  int numCells = mesh->nelems();
  constexpr int numVoigtTerms = Plato::SimplexThermomechanics<spaceDim>::mNumVoigtTerms;
  constexpr int nodesPerCell  = Plato::SimplexThermomechanics<spaceDim>::mNumNodesPerCell;
  constexpr int dofsPerCell   = Plato::SimplexThermomechanics<spaceDim>::mNumDofsPerCell;
  constexpr int dofsPerNode   = Plato::SimplexThermomechanics<spaceDim>::mNumDofsPerNode;

  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (spaceDim+1);
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;
  Plato::ScalarVector state("state", tNumDofs);
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal;

  }, "state");

  Plato::WorksetBase<Plato::SimplexThermomechanics<spaceDim>> worksetBase(*mesh);

  Plato::ScalarArray3DT<Plato::Scalar>     gradient("gradient",numCells,nodesPerCell,spaceDim);
  Plato::ScalarMultiVectorT<Plato::Scalar> tStrain("strain", numCells, numVoigtTerms);
  Plato::ScalarMultiVectorT<Plato::Scalar> tGrad("temperature gradient", numCells, spaceDim);
  Plato::ScalarMultiVectorT<Plato::Scalar> tStress("stress", numCells, numVoigtTerms);
  Plato::ScalarMultiVectorT<Plato::Scalar> tFlux("thermal flux", numCells, spaceDim);
  Plato::ScalarMultiVectorT<Plato::Scalar> result("result", numCells, dofsPerCell);
  Plato::ScalarArray3DT<Plato::Scalar>     configWS("config workset",numCells, nodesPerCell, spaceDim);
  Plato::ScalarVectorT<Plato::Scalar>      tTemperature("Gauss point temperature", numCells);
  Plato::ScalarVectorT<Plato::Scalar>      tThermalContent("Gauss point heat content at step k", numCells);
  Plato::ScalarMultiVectorT<Plato::Scalar> massResult("mass", numCells, dofsPerCell);
  Plato::ScalarMultiVectorT<Plato::Scalar> stateWS("state workset",numCells, dofsPerCell);

  worksetBase.worksetConfig(configWS);

  worksetBase.worksetState(state, stateWS);


  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                           \n"
    "  <ParameterList name='Material Models'>                                       \n"
    "    <ParameterList name='Cookie Dough'>                                        \n"
    "      <ParameterList name='Thermal Mass'>                                      \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>             \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>          \n"
    "      </ParameterList>                                                         \n"
    "      <ParameterList name='Thermoelastic'>                                     \n"
    "        <ParameterList name='Elastic Stiffness'>                               \n"
    "          <Parameter  name='Poissons Ratio' type='double' value='0.3'/>        \n"
    "          <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>     \n"
    "        </ParameterList>                                                       \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>  \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='1000.0'/> \n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>   \n"
    "      </ParameterList>                                                         \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                             \n"
    "</ParameterList>                                                               \n"
  );

  Plato::ThermalMassModelFactory<spaceDim> mmmfactory(*params);
  auto massMaterialModel = mmmfactory.create("Cookie Dough");

  Plato::ThermoelasticModelFactory<spaceDim> mmfactory(*params);
  auto materialModel = mmfactory.create("Cookie Dough");

  Plato::ComputeGradientWorkset<spaceDim>  computeGradient;
  Plato::TMKinematics<spaceDim>                   kinematics;
  Plato::TMKinetics<spaceDim>                     kinetics(materialModel);

  Plato::InterpolateFromNodal<spaceDim, dofsPerNode, TDofOffset> interpolateFromNodal;

  Plato::FluxDivergence  <spaceDim, dofsPerNode, TDofOffset> fluxDivergence;
  Plato::StressDivergence<spaceDim, dofsPerNode> stressDivergence;

  Plato::ThermalContent<spaceDim> computeThermalContent(massMaterialModel);
  Plato::ProjectToNode<spaceDim, dofsPerNode, TDofOffset> projectThermalContent;

  Plato::LinearTetCubRuleDegreeOne<spaceDim> cubatureRule;

  Plato::Scalar quadratureWeight = cubatureRule.getCubWeight();
  auto basisFunctions = cubatureRule.getBasisFunctions();

  Plato::Scalar tTimeStep = 1.0;

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",numCells);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    computeGradient(cellOrdinal, gradient, configWS, cellVolume);
    cellVolume(cellOrdinal) *= quadratureWeight;

    kinematics(cellOrdinal, tStrain, tGrad, stateWS, gradient);

    interpolateFromNodal(cellOrdinal, basisFunctions, stateWS, tTemperature);

    kinetics(cellOrdinal, tStress, tFlux, tStrain, tGrad, tTemperature);

    stressDivergence(cellOrdinal, result, tStress, gradient, cellVolume, tTimeStep/2.0);

    fluxDivergence(cellOrdinal, result, tFlux, gradient, cellVolume, tTimeStep/2.0);

    computeThermalContent(cellOrdinal, tThermalContent, tTemperature);
    projectThermalContent(cellOrdinal, cellVolume, basisFunctions, tThermalContent, massResult);

  }, "divergence");

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
    2.9999999999999997e-06, 2.3999999999999999e-06,
    1.6999999999999998e-06, 3.4999999999999995e-06,
    3.1000000000000000e-06, 2.2000000000000001e-06,
    2.0999999999999998e-06, 3.8999999999999999e-06,
    3.9999999999999998e-06, 2.9000000000000002e-06,
    2.7000000000000000e-06, 3.5999999999999998e-06
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
  auto tThermalContent_Host = Kokkos::create_mirror_view( tThermalContent );
  Kokkos::deep_copy( tThermalContent_Host, tThermalContent );

  std::vector<Plato::Scalar> tThermalContent_gold = { 
    0.90, 0.72, 0.51, 1.05, 0.93, 0.66, 0.63, 1.17, 1.20, 0.87, 0.81, 1.08 
  };

  numGoldCells=tThermalContent_gold.size();
  for(int iCell=0; iCell<numGoldCells; iCell++){
    if(tThermalContent_gold[iCell] == 0.0){
      TEST_ASSERT(fabs(tThermalContent_Host(iCell)) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(tThermalContent_Host(iCell), tThermalContent_gold[iCell], 1e-13);
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
    {8.0e-07,  6.4e-06,  3.2e-06}, 
    {8.0e-06,  6.4e-06, -4.0e-06},
    {8.0e-06,  1.6e-06,  8.0e-07},
    {1.6e-05, -6.4e-06,  8.0e-07}
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
   {8.0e-04,  6.4e-03,  3.2e-03}, 
   {8.0e-03,  6.4e-03, -4.0e-03},
   {8.0e-03,  1.6e-03,  8.0e-04},
   {1.6e-02, -6.4e-03,  8.0e-04}
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
   {-1602.56410256410254, -12099.2027243589710, -5128.20512820512704, -0.000133333333333333313,
     6169.71554487179492, -3525.64102564102450, -9695.35657051281487, -0.0000499999999999999685,
    -5688.94631410256352,  10496.6386217948711,  4006.41025641025590,  0.000116666666666666638,
     1121.79487179487114,  5128.20512820512704,  10817.1514423076878,  0.0000666666666666666428},
   {-4487.17948717948639, -7772.31089743589382, -2243.58974358974365, -0.000133333333333333313,
     480.769230769230489,  5528.72115384615336,  4407.17628205128221,  0.000216666666666666630,
    -1842.82371794871665, -2243.58974358974274, -6169.99679487179219, -0.000249999999999999951,
     5849.23397435897186,  4487.17948717948639,  4006.41025641025590,  0.000166666666666666634}
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
    {0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00468749999999999896,
     0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00468749999999999896,
     0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00468749999999999896,
     0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00468749999999999896},
    {0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00374999999999999986,
     0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00374999999999999986,
     0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00374999999999999986,
     0.000000000000000000, 0.000000000000000000, 0.000000000000000000, 0.00374999999999999986}
  };

  for(int iCell=0; iCell<int(mass_result_gold.size()); iCell++){
    for(int iDof=0; iDof<dofsPerCell; iDof++){
      if(mass_result_gold[iCell][iDof] == 0.0){
        TEST_ASSERT(fabs(mass_result_Host(iCell,iDof)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(mass_result_Host(iCell,iDof), mass_result_gold[iCell][iDof], 1e-13);
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
TEUCHOS_UNIT_TEST( TransientThermomechTests, TransientThermomechResidual3D )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  // create mesh based solution from host data
  //
  int tNumDofsPerNode = (spaceDim+1);
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;
  Plato::ScalarVector state("state", tNumDofs);
  Plato::ScalarVector stateDot("state dot", tNumDofs);
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal;
     stateDot(aNodeOrdinal*tNumDofsPerNode+0) = (4e-7)*aNodeOrdinal;
     stateDot(aNodeOrdinal*tNumDofsPerNode+1) = (3e-7)*aNodeOrdinal;
     stateDot(aNodeOrdinal*tNumDofsPerNode+2) = (2e-7)*aNodeOrdinal;
     stateDot(aNodeOrdinal*tNumDofsPerNode+3) = (1e-7)*aNodeOrdinal;

  }, "state");


  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Parabolic'/>          \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                  \n"
    "  <ParameterList name='Parabolic'>                                            \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Frozen Peas'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Frozen Peas'>                                        \n"
    "      <ParameterList name='Thermal Mass'>                                     \n"
    "        <Parameter name='Mass Density' type='double' value='0.3'/>            \n"
    "        <Parameter name='Specific Heat' type='double' value='1.0e6'/>         \n"
    "      </ParameterList>                                                        \n"
    "      <ParameterList name='Thermoelastic'>                                    \n"
    "        <ParameterList name='Elastic Stiffness'>                              \n"
    "          <Parameter  name='Poissons Ratio' type='double' value='0.3'/>       \n"
    "          <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>    \n"
    "        </ParameterList>                                                      \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/> \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='1000.0'/>\n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Time Integration'>                                     \n"
    "    <Parameter name='Number Time Steps' type='int' value='3'/>                \n"
    "    <Parameter name='Time Step' type='double' value='0.5'/>                   \n"
    "    <Parameter name='Trapezoid Alpha' type='double' value='0.5'/>             \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

  // create constraint evaluator
  //
  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *params);
  Plato::Parabolic::VectorFunction<::Plato::Thermomechanics<spaceDim>>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));


  // compute and test value
  //
  auto timeStep = params->sublist("Time Integration").get<Plato::Scalar>("Time Step");
  auto residual = vectorFunction.value(state, stateDot, z, timeStep);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<Plato::Scalar> residual_gold = { 
   -74678.38301282050, -59614.82211538460, -78204.58653846153, 0.006014583333333334,
   -69710.05929487177, -62980.04006410255, -66346.07051282052, 0.008424999999999998,
    6250.406250000000, -25480.55048076922, -6731.394230769230, 0.001677083333333333,
   -80767.10576923075, -38781.71794871794, -102564.2275641025, 0.01257343750000000,
   -12659.43349358974, -12820.45032051281, -481.6546474358953, 0.003273958333333333,
   -10255.82692307692, -3365.665865384615, -13301.58413461538, 0.001520833333333333,
   -6248.854166666652, -161.3189102564033, -26282.13461538462, 0.004729687500000000
  };

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-13);
    }
  }


  // compute and test gradient wrt state. (i.e., jacobian)
  //
  auto jacobian = vectorFunction.gradient_u(state, stateDot, z, timeStep);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
    3.52564102564102478e10, 0.00000000000000000,    0.00000000000000000,    52083.3333333333285,
    0.00000000000000000,    3.52564102564102478e10, 0.00000000000000000,    52083.3333333333285,
    0.00000000000000000,    0.00000000000000000,    3.52564102564102478e10, 52083.3333333333285,
    0.00000000000000000,    0.00000000000000000,    0.00000000000000000,    499.999999999999943,
    -6.41025641025640965e9, 3.20512820512820482e9,  0.00000000000000000,    0.00000000000000000,
    4.80769230769230652e9, -2.24358974358974304e10, 4.80769230769230652e9,  52083.3333333333285,
    0.00000000000000000,    3.20512820512820482e9, -6.41025641025640965e9,  0.00000000000000000,
    0.00000000000000000,    0.00000000000000000,    0.00000000000000000,    -166.66666666666667,
    -6.41025641025640965e9, 0.00000000000000000,    3.20512820512820482e9,  0.00000000000000000,
    0.00000000000000000,   -6.41025641025640965e9,  3.20512820512820482e9,  0.00000000000000000,
    4.80769230769230652e9,  4.80769230769230652e9, -2.24358974358974304e10, 52083.3333333333285,
    0.00000000000000000,    0.00000000000000000,    0.00000000000000000,    -166.66666666666667,
    0.00000000000000000,    3.20512820512820482e9,  3.20512820512820482e9,  0.00000000000000000,
    4.80769230769230652e9,  0.00000000000000000,   -8.01282051282051086e9,  26041.6666666666642,
    4.80769230769230652e9, -8.01282051282051086e9,  0.00000000000000000,    26041.6666666666642,
    0.00000000000000000,    0.00000000000000000,    0.00000000000000000,    0.00000000000000000
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  // compute and test gradient wrt state dot (i.e., jacobianV)
  //
  auto jacobian_v = vectorFunction.gradient_v(state, stateDot, z, timeStep);

  auto jac_v_entries = jacobian_v->entries();
  auto jac_v_entriesHost = Kokkos::create_mirror_view( jac_v_entries );
  Kokkos::deep_copy(jac_v_entriesHost, jac_v_entries);

  std::vector<Plato::Scalar> gold_jac_v_entries = {
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2343.750000000000000,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 781.2500000000000000,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 781.2500000000000000,
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 781.2500000000000000
  };

  int jac_v_entriesSize = gold_jac_v_entries.size();
  for(int i=0; i<jac_v_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_v_entriesHost(i), gold_jac_v_entries[i], 1.0e-15);
  }


  // compute and test objective gradient wrt control, z
  //
  auto gradient_z = vectorFunction.gradient_z(state, stateDot, z, timeStep);
  
  auto grad_entries = gradient_z->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
   -18669.5957532051252, -14903.7055288461488, -19551.1466346153829, 0.00150364583333333340,
   -2604.08854166666652,  8012.67988782051179,  4206.79326923076951, 0.000694010416666666560,
    1562.59114583333439, -6370.14803685897277, -1682.82772435897550, 0.000341145833333333329,
   -2804.38040865384437, -200.364783653846530, -4927.94711538461343, 0.000208723958333333358
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 2.0e-14);
  }


  // compute and test objective gradient wrt node position, x
  //
  auto gradient_x = vectorFunction.gradient_x(state, stateDot, z, timeStep);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<Plato::Scalar> gold_grad_x_entries = {
   -138461.538461538410, -151923.076923076878, -319230.769230769132, -0.0143479166666666651,
    47435.8974358974156, -33333.3333333333358,  55769.2307692307368, -0.00328541666666666522,
   -641.025641025629739,  1923.07692307693378, -11538.4615384615317, -0.00412916666666666640,
   -1282.05128205127858, -18909.6314102564065, -18589.7435897435789,  0.000199999999999999982,
    40063.4775641025481,  66666.6666666666570,  11217.4487179487187, -0.00341874999999999991,
   -26282.0512820512740, -28525.1410256410236, -1282.05128205128494, -0.000599999999999999947,
   -77564.1025641025335,  23717.9487179487187,  165705.857371794817,  0.00193333333333333291,
   -641.025641025644290, -18589.7435897435826, -58012.4663461538148, -0.000466666666666666554,
    44871.0657051281887,  41666.3124999999854,  55769.2307692307659, -0.00246249999999999985,
   -85897.4358974358765,  5449.21794871795646, -3525.28685897435935, -0.000599999999999999947,
    6089.24358974358620,  30128.2051282051179,  61538.6073717948602, -0.000808333333333333647,
    39743.2355769230635,  20192.1618589743448,  14743.5897435897496, -0.000261458333333333378
  };

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-13);
  }

}


/*!
  These unit tests are for the Derivative functionality.
 \todo 
*/

#include "PlatoTestHelpers.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "Omega_h_file.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "PlatoStaticsTypes.hpp"
#include "ImplicitFunctors.hpp"

#ifdef HAVE_AMGX
#include "alg/AmgXSparseLinearProblem.hpp"
#endif

#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <Sacado.hpp>

#include "alg/CrsLinearProblem.hpp"
#include "alg/ParallelComm.hpp"

#include "Simp.hpp"
#include "Solutions.hpp"
#include "ScalarProduct.hpp"
#include "SimplexFadTypes.hpp"
#include "SimplexMechanics.hpp"
#include "WorksetBase.hpp"
#include "elliptic/VectorFunction.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/SolutionFunction.hpp"
#include "geometric/GeometryScalarFunction.hpp"
#include "ApplyConstraints.hpp"
#include "elliptic/Problem.hpp"
#include "Mechanics.hpp"
#include "Thermal.hpp"

#include <fenv.h>

using ordType = typename Plato::ScalarMultiVector::size_type;


TEUCHOS_UNIT_TEST( DerivativeTests, 3D )
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
    "      <ParameterList name='Isotropic Linear Elastic'>                       \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>        \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>      \n"
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

  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  auto tOnlyDomain = tSpatialModel.Domains.front();

  int numCells = tMesh->nelems();
  int nodesPerCell  = Plato::SimplexMechanics<spaceDim>::mNumNodesPerCell;
  int numVoigtTerms = Plato::SimplexMechanics<spaceDim>::mNumVoigtTerms;
  int dofsPerCell   = Plato::SimplexMechanics<spaceDim>::mNumDofsPerCell;

  // create mesh based displacement from host data
  //
  std::vector<Plato::Scalar> u_host( spaceDim*tMesh->nverts() );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);


  Plato::WorksetBase<Plato::SimplexMechanics<spaceDim>> worksetBase(*tMesh);

  Plato::ScalarArray3DT<Plato::Scalar>
    gradient("gradient",numCells,nodesPerCell,spaceDim);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    strain("strain",numCells,numVoigtTerms);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    stress("stress",numCells,numVoigtTerms);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    result("result",numCells,dofsPerCell);

  Plato::ScalarArray3DT<Plato::Scalar>
    configWS("config workset",numCells, nodesPerCell, spaceDim);
  worksetBase.worksetConfig(configWS);

  Plato::ScalarMultiVectorT<Plato::Scalar>
    stateWS("state workset",numCells, dofsPerCell);
  worksetBase.worksetState(u, stateWS);

  Plato::ComputeGradientWorkset<spaceDim> computeGradient;
  Plato::Strain<spaceDim> voigtStrain;


  Plato::ElasticModelFactory<spaceDim> mmfactory(*tParamList);
  auto materialModel = mmfactory.create(tOnlyDomain.getMaterialName());
  auto tCellStiffness = materialModel->getStiffnessMatrix();

  Plato::LinearStress<spaceDim>      voigtStress(tCellStiffness);
  Plato::StressDivergence<spaceDim>  stressDivergence;

  Plato::Scalar quadratureWeight = 1.0/6.0;

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",numCells);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    computeGradient(cellOrdinal, gradient, configWS, cellVolume);
    cellVolume(cellOrdinal) *= quadratureWeight;

    voigtStrain(cellOrdinal, strain, stateWS, gradient);

    voigtStress(cellOrdinal, stress, strain);

    stressDivergence(cellOrdinal, result, stress, gradient, cellVolume);
  }, "gradient");


  // test gradient
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

  int numGoldCells=gradient_gold.size();
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

  // test strain
  //
  auto strain_Host = Kokkos::create_mirror_view( strain );
  Kokkos::deep_copy( strain_Host, strain );

  std::vector<std::vector<Plato::Scalar>> strain_gold = { 
    {0.0006, 0.0048, 0.0024, 0.0072, 0.003 , 0.0054},
    {0.006 , 0.0048,-0.0030, 0.0018, 0.003 , 0.0108},
    {0.006 , 0.0012, 0.0006, 0.0018, 0.0066, 0.0072},
    {0.012 ,-0.0048, 0.0006,-0.0042, 0.0126, 0.0072},
    {0.006 , 0.0012, 0.0006, 0.0018, 0.0066, 0.0072},
    {0.006 , 0.0012, 0.0006, 0.0018, 0.0066, 0.0072}
  };

  for(int iCell=0; iCell<int(strain_gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<numVoigtTerms; iVoigt++){
      if(strain_gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(strain_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(strain_Host(iCell,iVoigt), strain_gold[iCell][iVoigt], 1e-13);
      }
    }
  }

  // test stress
  //
  auto stress_Host = Kokkos::create_mirror_view( stress );
  Kokkos::deep_copy( stress_Host, stress );

  std::vector<std::vector<Plato::Scalar>> stress_gold = { 
   { 4961.538461538461, 8192.307692307691, 6346.153846153846, 2769.230769230769, 1153.846153846154, 2076.923076923077 },
   { 9115.384615384613, 8192.307692307690, 2192.307692307691, 692.3076923076922, 1153.846153846154, 4153.846153846153 },
   { 9115.384615384612, 5423.076923076921, 4961.538461538460, 692.3076923076922, 2538.461538461539, 2769.230769230769 },
   { 13730.76923076923, 807.6923076923071, 4961.538461538460,-1615.384615384614, 4846.153846153846, 2769.230769230769 },
   { 9115.384615384612, 5423.076923076921, 4961.538461538460, 692.3076923076924, 2538.461538461539, 2769.230769230769 },
   { 9115.384615384612, 5423.076923076921, 4961.538461538460, 692.3076923076922, 2538.461538461539, 2769.230769230769 }
  };

  for(int iCell=0; iCell<int(stress_gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<numVoigtTerms; iVoigt++){
      if(stress_gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(stress_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(stress_Host(iCell,iVoigt), stress_gold[iCell][iVoigt], 1e-13);
      }
    }
  }

  // test residual
  //
  auto result_Host = Kokkos::create_mirror_view( result );
  Kokkos::deep_copy( result_Host, result );

  std::vector<std::vector<Plato::Scalar>> result_gold = { 
   {-86.53846153846153, -341.3461538461538, -115.3846153846154,   158.6538461538462,
    -28.84615384615385, -216.3461538461538, -120.1923076923077,   254.8076923076923,
     67.30769230769231,   48.07692307692308, 115.3846153846154,   264.4230769230770},
   {-173.0769230769231, -341.3461538461538,  -28.84615384615385,  125.0000000000000,
     312.5000000000000,  -62.49999999999994,-331.7307692307691,  -144.2307692307692,
      43.26923076923080, 379.8076923076923,  173.0769230769231,    48.07692307692308},
   {-105.7692307692308,  -28.84615384615385,-206.7307692307692,  -264.4230769230767,
     110.5769230769231,  -76.92307692307692,  -9.615384615384613,-197.1153846153846,
     177.8846153846154,  379.8076923076923,  115.3846153846154,   105.7692307692308},
   {-201.9230769230769,   67.30769230769229,-206.7307692307692,   456.7307692307693,
      81.73076923076928, 269.2307692307692,  115.3846153846154,    33.65384615384622,
     -67.30769230769229,-370.1923076923075, -182.6923076923077,     4.807692307692264},
   {-115.3846153846154, -225.9615384615384,  -28.84615384615384,  274.0384615384615,
      86.53846153846152,-100.9615384615383, -264.4230769230767,   110.5769230769230,
     -76.92307692307688, 105.7692307692307,   28.84615384615384,  206.7307692307692},
   {-115.3846153846154, -225.9615384615384,  -28.84615384615384,    9.615384615384613,
     197.1153846153846, -177.8846153846153, -274.0384615384614,   -86.53846153846155,
     100.9615384615384,  379.8076923076923,  115.3846153846154,   105.7692307692308}
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
}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         ElastostaticResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, ElastostaticResidual3D )
{
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


  // create mesh based displacement from host data
  //
  std::vector<Plato::Scalar> u_host( spaceDim*tMesh->nverts() );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for( auto& val : u_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    u_host_view(u_host.data(),u_host.size());
  auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), u_host_view);



  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                             \n"
    "  <ParameterList name='Spatial Model'>                                           \n"
    "    <ParameterList name='Domains'>                                               \n"
    "      <ParameterList name='Design Volume'>                                       \n"
    "        <Parameter name='Element Block' type='string' value='body'/>             \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>     \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>              \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                      \n"
    "  <ParameterList name='Elliptic'>                                                \n"
    "    <ParameterList name='Penalty Function'>                                      \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                     \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>                \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                        \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Material Models'>                                         \n"
    "    <ParameterList name='Unobtainium'>                                           \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                            \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>             \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>           \n"
    "      </ParameterList>                                                           \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "  <ParameterList name='Criteria'>                                                \n"
    "    <ParameterList name='Internal Elastic Energy'>                               \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>             \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "    </ParameterList>                                                             \n"
    "  </ParameterList>                                                               \n"
    "</ParameterList>                                                                 \n"
  );

  // create constraint evaluator
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::DataMap tDataMap;
  Plato::Elliptic::VectorFunction<::Plato::Mechanics<spaceDim>>
    esVectorFunction(tSpatialModel, tDataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));


  // compute and test constraint value
  //
  auto residual = esVectorFunction.value(u,z);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<Plato::Scalar> residual_gold = { 
    -1903.846153846153,  -894.2307692307692,-1038.461538461538,
    -2062.499999999999, -1024.038461538461,  -692.3076923076922,
     -379.8076923076920, -379.8076923076922,  182.6923076923077,
    -2379.807692307691,  -793.2692307692305, -894.2307692307687,
     -798.0769230769225, -235.5769230769230,  283.6538461538459,
     -538.4615384615381,  -19.23076923076923, -19.23076923076923,
     -605.7692307692301,  259.6153846153844, -259.6153846153845,
     -173.0769230769229,  173.0769230769230, -173.0769230769230,
     -485.5769230769228,  336.5384615384618, -139.4230769230768,
     -615.3846153846150,-1120.192307692307, -1754.807692307692,
     -264.4230769230765, -610.5769230769226, -394.2307692307692,
        0.0000000000000,    0.0000000000000, -346.1538461538459,
       28.84615384615405, 374.9999999999998, -317.3076923076922,
     1274.038461538463,  -673.0769230769218, -312.4999999999985,
     1341.346153846153,  -302.8846153846144,  663.4615384615385,
      913.4615384615381,  668.2692307692305,  552.8846153846155,
     1033.653846153846,  1336.538461538461,   514.4230769230774,
      437.5000000000005,  379.8076923076925,  451.9230769230770,
      451.9230769230770,  221.1538461538464,  221.1538461538462,
      302.8846153846157, -490.3846153846151,   72.11538461538484,
      971.1538461538465, -399.0384615384608,  783.6538461538462,
     1269.230769230769,   721.1538461538468,  721.1538461538469,
      658.6538461538461,   96.15384615384637, 572.1153846153854,
       48.07692307692318, 134.6153846153847,   48.07692307692324,
       62.49999999999966, 365.3846153846159,  149.0384615384621,
     1365.384615384615,  1610.576923076923,   860.5769230769234,
       48.07692307692358, 264.4230769230770,  264.4230769230767
  };

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-13);
    }
  }


  // compute and test constraint gradient wrt state, u. (i.e., jacobian)
  //
  auto jacobian = esVectorFunction.gradient_u(u,z);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
    3.52564102564102504e+05, 0.00000000000000000e+00, 0.00000000000000000e+00,
    0.00000000000000000e+00, 3.52564102564102563e+05, 0.00000000000000000e+00,
    0.00000000000000000e+00, 0.00000000000000000e+00, 3.52564102564102563e+05, 

   -6.41025641025641016e+04, 3.20512820512820508e+04, 0.00000000000000000e+00,
    4.80769230769230708e+04,-2.24358974358974316e+05, 4.80769230769230708e+04,
    0.00000000000000000e+00, 3.20512820512820508e+04,-6.41025641025641016e+04, 

   -6.41025641025641016e+04, 0.00000000000000000e+00, 3.20512820512820508e+04,
    0.00000000000000000e+00,-6.41025641025641016e+04, 3.20512820512820508e+04, 
    4.80769230769230708e+04, 4.80769230769230708e+04,-2.24358974358974316e+05,

    0.00000000000000000e+00, 3.20512820512820508e+04, 3.20512820512820508e+04,
    4.80769230769230708e+04, 0.00000000000000000e+00, -8.01282051282051252e+04,
    4.80769230769230708e+04,-8.01282051282051252e+04, 0.00000000000000000e+00, 

    0.00000000000000000e+00,-8.01282051282051252e+04, 4.80769230769230708e+04,
   -8.01282051282051252e+04, 0.00000000000000000e+00, 4.80769230769230708e+04, 
    3.20512820512820508e+04, 3.20512820512820508e+04, 0.00000000000000000e+00
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  // compute and test gradient wrt control, z
  //
  auto gradient_z = esVectorFunction.gradient_z(u,z);
  
  auto grad_entries = gradient_z->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
    -475.9615384615383,  -223.5576923076923,   -259.6153846153846, 
       1.201923076923094, 141.8269230769231,      1.201923076923091, 
     -94.95192307692304,  -94.95192307692307,    45.67307692307691, 
    -149.0384615384614,    -8.413461538461540,   -8.413461538461529, 
      -8.413461538461519,  -8.413461538461512, -149.0384615384615, 
     341.3461538461538,    88.94230769230769,   125.0000000000000, 
     123.7980769230769,   -16.82692307692301,   123.7980769230769, 
     262.0192307692307,   121.3942307692307,    121.3942307692308
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 2.0e-14);
  }


  // compute and test gradient wrt node position, x
  //
  auto gradient_x = esVectorFunction.gradient_x(u,z);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<Plato::Scalar> gold_grad_x_entries = {
  -4153.84615384615245,   -2278.84615384615336,  -3192.30769230769147, 
   1423.07692307692287,    -500.000000000000000,   557.692307692307281, 
    -19.2307692307692832,    28.8461538461539817, -115.384615384615515, 
      9.61538461538469846, -153.846153846154266,  -307.692307692307509, 
    586.538461538461434,    999.999999999999773,   355.769230769230717, 
   -480.769230769230717,   -730.769230769230717,    67.3076923076923208, 
   -403.846153846153470,    423.076923076922867,  1028.84615384615358, 
    -96.1538461538464730,  -230.769230769230717,  -701.923076923076565, 
   1384.61538461538430,     692.307692307692150,   557.692307692307395, 
  -1134.61538461538430,    -451.923076923076678,  -182.692307692307651, 
    586.538461538461434,    403.846153846153697,   557.692307692307622, 
    990.384615384615017,    490.384615384615472,    67.3076923076923208
  };

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-13);
  }

}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, InternalElasticEnergy3D )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Objective' type='string' value='My Internal Elastic Energy'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Internal Elastic Energy'>                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

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


  // create mesh based displacement from host data
  //
  ordType tNumDofs = spaceDim*tMesh->nverts();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumDofs);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view( u );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for(ordType i=0; i<tNumDofs; i++)
  {
      u_host(i) = (disp += dval);
  }
  Kokkos::deep_copy(u, u_host);


  // create objective
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Internal Elastic Energy");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<spaceDim>>
    eeScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = eeScalarFunction.value(tSolution,z);

  Plato::Scalar value_gold = 46.125;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  tSolution.set("State", U);
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
    -1903.84615384615,-894.230769230769,-1038.46153846154,
    -2062.5,-1024.03846153846,-692.307692307692,
    -379.807692307692,-379.807692307692,182.692307692308,
    -2379.80769230769,-793.269230769231,-894.230769230768,
    -798.076923076923,-235.576923076923,283.653846153846,
    -538.461538461538,-19.2307692307693,-19.2307692307692,
    -605.769230769231,259.615384615385,-259.615384615384,
    -173.076923076923,173.076923076923,-173.076923076923,
    -485.576923076923,336.538461538462,-139.423076923077,
    -615.384615384615,-1120.19230769231,-1754.80769230769,
    -264.423076923077,-610.576923076923,-394.230769230769,
    0,0,-346.153846153846,
    28.8461538461541,375,-317.307692307692,
    1274.03846153846,-673.076923076922,-312.499999999999,
    1341.34615384615,-302.884615384615,663.461538461538,
    913.461538461539,668.269230769231,552.884615384616,
    1033.65384615385,1336.53846153846,514.423076923077,
    437.5,379.807692307693,451.923076923077,
    451.923076923077,221.153846153846,221.153846153846,
    302.884615384616,-490.384615384615,72.1153846153848,
    971.153846153847,-399.038461538461,783.653846153847,
    1269.23076923077,721.153846153847,721.153846153847,
    658.653846153846,96.1538461538463,572.115384615385,
    48.0769230769231,134.615384615385,48.0769230769233,
    62.4999999999999,365.384615384616,149.038461538462,
    1365.38461538462,1610.57692307692,860.576923076924,
    48.0769230769235,264.423076923077,264.423076923077
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test criterion gradient wrt control, z
  //
  tSolution.set("State", U);
  auto grad_z = eeScalarFunction.gradient_z(tSolution,z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
    3.55564903846154,3.68509615384615,0.921274038461539,
    3.02668269230769,1.01213942307692,0.488942307692308,
    0.868269230769231,0.271153846153846,1.44483173076923,
    3.00504807692308,1.09290865384615,0.20625,
    0.382211538461539,6.56538461538462,3.53221153846154,
    1.54435096153846,1.59483173076923,0.813100961538463,
    0.327764423076925,0.544831730769234,2.0985576923077,
    3.52536057692308,1.43473557692308,0.054807692307693,
    0.175600961538464,3.49903846153846,0.453966346153847
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  tSolution.set("State", U);
  auto grad_x = eeScalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
    24.4384615384615,-15.6721153846154,-7.28653846153846,
    22.9543269230769,-17.8572115384615,2.26730769230769,
    1.50721153846154,-4.72355769230769,7.71634615384616,
    11.7620192307692,7.68605769230769,-2.97115384615385,
    -1.67596153846154,0.95048076923077,6.41971153846154,
    -0.449999999999998,2.64807692307692,1.95576923076923,
    -1.32115384615384,5.93076923076923,-1.49423076923077,
    -0.305769230769231,2.06538461538461,-0.928846153846153,
    -3.35048076923077,0.608653846153847,-4.38894230769231,
    11.0711538461538,-1.39471153846154,-4.79567307692307,
    6.23509615384616,-7.68028846153845,-0.337499999999995,
    0.825,1.24038461538462,-0.403846153846154,
    -0.017307692307691,2.14615384615385,-2.54423076923077,
    -4.5360576923077,5.59326923076924,4.96586538461539,
    -10.9110576923077,5.16778846153846,9.76730769230769,
    -10.2432692307692,1.66009615384615,4.22163461538462,
    -8.33509615384615,2.30480769230769,-1.75817307692308,
    1.36009615384616,1.70625000000001,-3.82788461538462,
    -1.49711538461538,0.649038461538465,0.787499999999998,
    1.78701923076923,-0.908653846153841,1.57932692307692,
    2.93076923076923,3.9735576923077,0.304326923076927,
    -15.0663461538461,-7.28653846153847,1.61826923076923,
    -8.52836538461539,-4.92403846153846,5.43894230769231,
    -0.057692307692308,0.115384615384612,0.21923076923077,
    0.073557692307696,0.23365384615384,0.852403846153852,
    -21.9346153846154,14.1216346153846,-14.1764423076923,
    3.28557692307693,1.64567307692307,-3.20048076923077
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}

/******************************************************************************/
/*!
  \brief Compute value and gradients (wrt state, control, and configuration) of
         Solution criterion in 3D
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, Solution2D )
{
  // create test mesh
  //
  constexpr int meshWidth=4;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1.0, 1.0, meshWidth, meshWidth);


  // create mesh based density from host data
  //
  Plato::OrdinalType tNumVerts = tMesh->nverts();
  Plato::ScalarVector z("density", tNumVerts);
  Kokkos::deep_copy(z, 1.0);


  // create displacement field, u = x
  //
  auto tCoords = tMesh->coords();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumVerts*spaceDim);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, tNumVerts), LAMBDA_EXPRESSION(int aNodeOrdinal)
  {
    U(0, aNodeOrdinal*spaceDim + 0) = tCoords[aNodeOrdinal*spaceDim + 0];
    U(0, aNodeOrdinal*spaceDim + 1) = tCoords[aNodeOrdinal*spaceDim + 1];
  }, "set disp");


  // setup the problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>         \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Fancy Material'/> \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                     \n"
    "    <ParameterList name='Fancy Material'>                                     \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                          \n"
    "  <ParameterList name='Criteria'>                                           \n"
    "    <ParameterList name='Displacement'>                                     \n"
    "      <Parameter name='Type' type='string' value='Solution'/>               \n"
    "      <Parameter name='Normal' type='Array(double)' value='{1.0,1.0}'/>     \n"
    "      <Parameter name='Domain' type='string' value='y-'/>                   \n"
    "      <Parameter name='Magnitude' type='bool' value='false'/>               \n"
    "    </ParameterList>                                                        \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  // create objective
  //
  Plato::DataMap tDataMap;
  std::string tMyFunction("Displacement");
  Plato::Elliptic::SolutionFunction<::Plato::Mechanics<spaceDim>>
    scalarFunction(tSpatialModel, tDataMap, *tParamList, tMyFunction);


  // compute and test objective value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = scalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 0.5;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  TEST_FLOATING_EQUALITY(grad_u_Host(2),   1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(3),   1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(44),  1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(45),  1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(46),  1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(47),  1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(48),  1.0 / 5, 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(49),  1.0 / 5, 1e-15);


  // compute and test objective gradient wrt control, z
  //
  auto grad_z = scalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  for(int iNode=0; iNode<int(grad_z_Host.size()); iNode++){
    TEST_ASSERT(grad_z_Host[iNode] == 0.0);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradient_x(tSolution, z);

  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  for(int iNode=0; iNode<int(grad_x_Host.size()); iNode++){
    TEST_ASSERT(grad_x_Host[iNode] == 0.0);
  }
}
/******************************************************************************/
/*!
  \brief Compute value and gradients (wrt state, control, and configuration) of
         Solution magnitude criterion in 3D
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, Solution2D_Mag )
{
  // create test mesh
  //
  constexpr int meshWidth=4;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1.0, 1.0, meshWidth, meshWidth);


  // create mesh based density from host data
  //
  Plato::OrdinalType tNumVerts = tMesh->nverts();
  Plato::ScalarVector z("density", tNumVerts);
  Kokkos::deep_copy(z, 1.0);


  // create displacement field, u = x
  //
  auto tCoords = tMesh->coords();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumVerts*spaceDim);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, tNumVerts), LAMBDA_EXPRESSION(int aNodeOrdinal)
  {
    U(0, aNodeOrdinal*spaceDim + 0) = tCoords[aNodeOrdinal*spaceDim + 0];
    U(0, aNodeOrdinal*spaceDim + 1) = tCoords[aNodeOrdinal*spaceDim + 1];
  }, "set disp");


  // setup the problem
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>         \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Fancy Material'/> \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Fancy Material'>                                     \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Displacement'>                                       \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Solution'/> \n"
    "      <Parameter name='Normal' type='Array(double)' value='{1.0,1.0}'/>       \n"
    "      <Parameter name='Domain' type='string' value='y-'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                          \n"
    "</ParameterList>                                                            \n"
  );

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);


  // add named face
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);



  // create objective
  //
  Plato::DataMap tDataMap;
  std::string tMyFunction("Displacement");
  Plato::Elliptic::SolutionFunction<::Plato::Mechanics<spaceDim>>
    scalarFunction(tSpatialModel, tDataMap, *tParamList, tMyFunction);


  // compute and test objective value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = scalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 0.5;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  auto grad_u = scalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  TEST_FLOATING_EQUALITY(grad_u_Host(2),   1.0 * (1.0*0.25) / (0.25 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(3),   1.0 * (1.0*0.25) / (0.25 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(44),  1.0 * (1.0*0.50) / (0.50 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(45),  1.0 * (1.0*0.50) / (0.50 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(46),  1.0 * (1.0*0.75) / (0.75 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(47),  1.0 * (1.0*0.75) / (0.75 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(48),  1.0 * (1.0*1.00) / (1.00 * 5), 1e-15);
  TEST_FLOATING_EQUALITY(grad_u_Host(49),  1.0 * (1.0*1.00) / (1.00 * 5), 1e-15);


  // compute and test objective gradient wrt control, z
  //
  auto grad_z = scalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  for(int iNode=0; iNode<int(grad_z_Host.size()); iNode++){
    TEST_ASSERT(grad_z_Host[iNode] == 0.0);
  }

  // compute and test objective gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradient_x(tSolution, z);

  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  for(int iNode=0; iNode<int(grad_x_Host.size()); iNode++){
    TEST_ASSERT(grad_x_Host[iNode] == 0.0);
  }
}


/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, StressPNorm3D )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                  \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Globalized Stress'>                                  \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Stress P-Norm'/>  \n"
    "      <Parameter name='Exponent' type='double' value='12.0'/>                 \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

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


  // create mesh based displacement from host data
  //
  ordType tNumDofs = spaceDim*tMesh->nverts();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumDofs);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view( u );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for(ordType i=0; i<tNumDofs; i++)
  {
      u_host(i) = (disp += dval);
  }
  Kokkos::deep_copy(u, u_host);


  // create objective
  //
  Plato::DataMap dataMap;
  std::string tMyFunction("Globalized Stress");
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<spaceDim>>
    eeScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = eeScalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 14525.25169157000;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
   -1503430.743610086,    -87943.46429698351,  -253405.7951760115,
    -419311.2039113952,   -45900.11226120282,   -36121.19195937364,
    -127338.4661959158,   -19117.73429961895,    92407.24120276771,
     -50087.19124766427,   12774.33242060617,    -3333.510853871279,
     -23558.49126781739,    3380.937063817981,   14893.84056534542,
      -9519.888210062301,   3348.092294628147,    3560.944140991056,
      -5240.218786677828,   8738.583477630442,   -4060.103637689512,
        -47.36161489014854,  173.0873916025785,   -137.7668652438236,
      -5174.016020690595,  26242.73416532713,   -14003.94012109160
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test criterion gradient wrt control, z
  //
  auto grad_z = eeScalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   2972.321313190315,        831.6877699932147,    207.9219424983036,
     85.16009409571750,       30.23970839774167,     9.793975869956283,
     10.67428715222397,        0.1739233563059383,  32.70029626817538,
   1960.655369272931,          3.837915000705916,    0.06595545018012441,
      0.1109427260128550,   3191.518788255666,     633.0928752380449,
     29.38214582086229,       11.06169917284905,     0.3508304842757221,
      0.002762687660171751,    0.002953941115371589, 4.977103004996600,
   1375.578043139706,        394.9305857993233,      9.661752822222865e-7,
      6.955608440499901e-4, 2737.188926384874,       1.820787841892929
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = eeScalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
    25140.94641961542,   -14100.50148913174,     1367.291551437471,
    5454.778632091190,    -2261.028458537603,     300.7995048791828,
     515.4758244370678,    -264.9379458613482,    171.0440039078628,
     207.3155368336267,      90.12589972024526,   -18.43667104475125,
      14.65117107126713,      9.528794705847330,   16.35882582027728,
       9.135793400021047,     6.459878463616483,    3.202460912352750,
      -0.3251353455947416,    7.854573864154236,   -3.859633813615325,
       0.01424862142881591,   0.1087242510851228,  -0.06517564756324878,
     -40.80884087925770,    -15.10656437218265,    -1.627492863831697
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         EffectiveEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, EffectiveEnergy3D_NormalCellProblem )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                         \n"
    "  <ParameterList name='Spatial Model'>                                                       \n"
    "    <ParameterList name='Domains'>                                                           \n"
    "      <ParameterList name='Design Volume'>                                                   \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                         \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>                 \n"
    "      </ParameterList>                                                                       \n"
    "    </ParameterList>                                                                         \n"
    "  </ParameterList>                                                                           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                                 \n"
    "  <ParameterList name='Criteria'>                                                              \n"
    "    <ParameterList name='Effective Energy'>                                                    \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>                           \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Effective Energy'/>          \n"
    "      <Parameter name='Assumed Strain' type='Array(double)' value='{1.0,0.0,0.0,0.0,0.0,0.0}'/>\n"
    "      <ParameterList name='Penalty Function'>                                                  \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                                 \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>                            \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                                    \n"
    "      </ParameterList>                                                                         \n"
    "    </ParameterList>                                                                           \n"
    "  </ParameterList>                                                                             \n"
    "  <ParameterList name='Cell Problem Forcing'>                                                \n"
    "    <Parameter name='Column Index' type='int' value='0'/>                                    \n"
    "  </ParameterList>                                                                           \n"
    "  <ParameterList name='Material Models'>                                                     \n"
    "    <ParameterList name='Unobtainium'>                                                       \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                        \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>                         \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>                       \n"
    "      </ParameterList>                                                                       \n"
    "    </ParameterList>                                                                         \n"
    "  </ParameterList>                                                                           \n"
    "</ParameterList>                                                                             \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto numVerts = tMesh->nverts();

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( numVerts, 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);



  Plato::ScalarMultiVector solution("solution", /*numSteps=*/1, spaceDim*numVerts);

  // create mesh based displacement
  //
  std::vector<Plato::Scalar> solution_gold = {
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000,  0.0173669389188933626, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0411192268700119809,  0.00965747852017450475, 
    0.0000000000000000000,  0.0355244194737336372,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000,  0.00494803288197820032, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0511217432387267717,  0.0000000000000000000, 
   -0.0739957825759057081, -0.0162917876901311660,  0.0000000000000000000, 
    0.0000000000000000000, -0.0427458935980062973,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.142903245392837525,   0.0000000000000000000,  0.0000000000000000000, 
   -0.0172027445162594265,  0.00106322367588203379, 0.00106322367588205396, 
    0.0406024627433991536,  0.0154135960960958291,  0.0000000000000000000, 
    0.149780596048765896,   0.0000000000000000000,  0.0000000000000000000, 
    0.148483915692292107,   0.0000000000000000000,  0.00236800003074055191, 
    0.0000000000000000000,  0.0000000000000000000, -0.0113344145950257241, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000, -0.0450892189148590500,  0.0000000000000000000, 
    0.0000000000000000000, -0.0439820074688949750, -0.0125202591190575266, 
   -0.181877235524798508,   0.0000000000000000000, -0.00324619162477584461, 
   -0.117196040731724044,   0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000, -0.0121695070062508570, 
   -0.225131719618738901,   0.0000000000000000000,  0.0000000000000000000, 
    0.0000000000000000000,  0.0000000000000000000,  0.0000000000000000000 };


  // push gold data from host to device
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    tHostView(solution_gold.data(), solution_gold.size());
  auto step0 = Kokkos::subview(solution, 0, Kokkos::ALL());
  Kokkos::deep_copy(step0, tHostView);

  // create criterion
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Effective Energy");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<spaceDim>>
    eeScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", solution);
  auto value = eeScalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 1346153.84615384578;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
    112179.4871794871,      48076.92307692306,    48076.92307692306,
    168269.2307692307,      72115.38461538460,    0.000000000000000, 
    56089.74358974357,      24038.46153846153,   -48076.92307692306, 
    336538.4615384614,      0.000000000000000,    0.000000000000000, 
    168269.2307692307,      0.000000000000000,   -72115.38461538460, 
    112179.4871794871,     -24038.46153846153,   -24038.46153846153, 
    168269.2307692307,     -72115.38461538460,    0.000000000000000, 
    56089.74358974357,     -48076.92307692306,    24038.46153846153, 
    168269.2307692307,      0.000000000000000,    72115.38461538460,
   -1.455191522836685e-11,  0.000000000000000,    144230.7692307692, 
   -168269.2307692307,      0.000000000000000,    72115.38461538460, 
   -56089.74358974357,     -24038.46153846153,    48076.92307692306, 
    0.000000000000000,     -72115.38461538460,    72115.38461538460,
   -4.365574568510056e-11,  0.000000000000000,    0.000000000000000, 
   -1.455191522836685e-11,  0.000000000000000,   -144230.7692307692, 
    0.000000000000000,     -72115.38461538460,   -72115.38461538460,
   -1.455191522836685e-11, -144230.7692307692,    0.000000000000000,
   -168269.2307692307,     -72115.38461538460,    0.000000000000000, 
   -112179.4871794871,     -48076.92307692306,   -48076.92307692306, 
   -168269.2307692307,      0.000000000000000,   -72115.38461538460, 
   -336538.4615384614,      0.000000000000000,    0.000000000000000,
   -1.455191522836685e-11,  144230.7692307692,    0.000000000000000,
    0.000000000000000,      72115.38461538460,   -72115.38461538460,
   -56089.74358974357,      48076.92307692306,   -24038.46153846153,
   -168269.2307692307,      72115.38461538460,    0.000000000000000,
    0.000000000000000,      72115.38461538460,    72115.38461538460, 
   -112179.4871794871,      24038.46153846153,    24038.46153846153
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(fabs(grad_u_gold[iNode]) < 1e-10){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test criterion gradient wrt control, z
  //
  auto grad_z = eeScalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
    51415.0373984630496, 63626.4105765648710, 14999.0837995235233, 
    76149.3934262146504, 24360.5475584209853, 10370.2306261117192,
    20703.0116870963320, 10506.9555951488292, 53433.8245831477980,
    92117.0824403273000, 21902.6070614885975, 14770.1235645518063,
    31203.5021472171320, 188920.219746857270, 92794.1264836712799,
    55120.4014680251421, 87513.1858953364572, 60597.6279469306246,
    47551.5814146090779, 50405.0419535135588, 69397.7622101499728,
    86836.1418519924773, 31203.5021472171284, 10277.9953601771122,
    18245.0711901639443, 53355.1550102794354, 8378.22301064559724 };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = eeScalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   -106445.574316611834,     -157158.660963500093,     -144060.939883280807, 
   -163071.856024346896,     -222065.945634533884,     -4.54747350886464119e-11, 
   -55216.7877938377933,     -70071.6843249799276,      118322.772423217699, 
   -335191.527413369680,     -2.54658516496419907e-11, -8.52651282912120223e-12,
   -168055.897552366805,     -1.81898940354585648e-12,  143098.431691335514, 
   -116566.465917270718,      39525.3193318040503,      40995.3379179461335, 
   -174216.476848969964,      118267.067436839250,      4.54747350886464119e-12,
   -58309.6335107410923,      79140.0925266976701,     -42516.7069225371306, 
   -169079.626776330930,      1.09139364212751389e-11, -168056.776846833644,
   -4.84305928694084287e-11,  4.36557456851005554e-11, -373311.568021968415, 
    167903.506027127150,     -3.63797880709171295e-12, -127195.711052508646,
    54579.5763299848841,      72665.4801528035023,     -117071.140959200449, 
    6.82121026329696178e-12,  185243.311753953109,     -177461.297782159119, 
   -1.45519152283668518e-11,  7.27595761418342590e-11,  4.18367562815546989e-11,
    2.91038304567337036e-11,  1.45519152283668518e-11,  362612.108049641713, 
    0.000000000000000000,     152077.959759364254,      166904.448650544538,
    2.91038304567337036e-11,  336527.149233340984,      0.00000000000000000,
    162587.036085733649,      217499.896431886649,     -2.18278728425502777e-11, 
    106750.357367091114,      145207.569527156214,      132109.848446936900, 
    168594.806837717682,      1.73727698893344495e-11,  163490.727644186351, 
    336465.950341075426,     -4.00177668780088425e-11,  0.00000000000000000, 
    2.91038304567337036e-11, -347226.609205667686,     -1.45519152283668518e-11,
    7.27595761418342590e-12, -189194.940956436098,      173509.668579676159, 
    57672.4220468881977,     -77888.4610626804642,      45110.5027503607125, 
    174064.085323730309,     -102364.346798012397,     -1.45519152283668518e-11, 
    2.18278728425502777e-11, -148763.542020734254,     -163590.030911914480, 
    117536.105794497213,     -31419.6551873009557,     -32889.6737734430353
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(fabs(grad_x_gold[iNode]) < 1e-10){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         EffectiveEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, EffectiveEnergy3D_ShearCellProblem )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                         \n"
    "  <ParameterList name='Spatial Model'>                                                       \n"
    "    <ParameterList name='Domains'>                                                           \n"
    "      <ParameterList name='Design Volume'>                                                   \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                         \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>                 \n"
    "      </ParameterList>                                                                       \n"
    "    </ParameterList>                                                                         \n"
    "  </ParameterList>                                                                           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                                 \n"
    "  <ParameterList name='Criteria'>                                                              \n"
    "    <ParameterList name='Effective Energy'>                                                    \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>                           \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Effective Energy'/>          \n"
    "      <Parameter name='Assumed Strain' type='Array(double)' value='{0.0,0.0,0.0,1.0,0.0,0.0}'/>\n"
    "      <ParameterList name='Penalty Function'>                                                  \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                                 \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>                            \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                                    \n"
    "      </ParameterList>                                                                         \n"
    "    </ParameterList>                                                                           \n"
    "  </ParameterList>                                                                             \n"
    "  <ParameterList name='Cell Problem Forcing'>                                                \n"
    "    <Parameter name='Column Index' type='int' value='3'/>                                    \n"
    "  </ParameterList>                                                                           \n"
    "  <ParameterList name='Material Models'>                                                     \n"
    "    <ParameterList name='Unobtainium'>                                                       \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                        \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>                         \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>                       \n"
    "      </ParameterList>                                                                       \n"
    "    </ParameterList>                                                                         \n"
    "  </ParameterList>                                                                           \n"
    "</ParameterList>                                                                             \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  auto numVerts = tMesh->nverts();

  // create mesh based density from host data
  //
  std::vector<Plato::Scalar> z_host( numVerts, 1.0 );
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    z_host_view(z_host.data(),z_host.size());
  auto z = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), z_host_view);



  Plato::ScalarMultiVector solution("solution", /*numSteps=*/1, spaceDim*numVerts);

  // create mesh based displacement
  //
  std::vector<Plato::Scalar> solution_gold = {
0, 0, 0, 0, 0, 0, 0, 0, 0, 0.125000000000000028, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3.43954359943216618e-18, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0.125000000000000028, 
-7.03590337326575184e-18, -5.50739874438680158e-18, 0, 0, 
2.12992705742669426e-18, 0, 0, 0, 0, 1.11083808367229628e-18, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.125000000000000028, 0, 0, 0, 
-2.36483063822016535e-18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0
  };

  // push gold data from host to device
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    tHostView(solution_gold.data(), solution_gold.size());
  auto step0 = Kokkos::subview(solution, 0, Kokkos::ALL());
  Kokkos::deep_copy(step0, tHostView);

  // create criterion
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Effective Energy");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Mechanics<spaceDim>>
    eeScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", solution);
  auto value = eeScalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 384615.384615384275;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
0, 32051.28205128205, 32051.28205128205, 0, 0, 48076.92307692307, 0, 
-32051.28205128205, 16025.64102564102, 0, 0, 0, 0, 
-48076.92307692307, 0, 0, -16025.64102564102, -16025.64102564102, 0, 
0, -48076.92307692307, 0, 16025.64102564102, -32051.28205128205, 0, 
48076.92307692307, 0, 0, 96153.84615384616, 0, 0, 48076.92307692307, 
0, 0, 32051.28205128205, -16025.64102564102, 0, 48076.92307692307, 
-48076.92307692307, 0, 0, 7.275957614183426e-12, 0, 
-96153.84615384616, 0, 0, -48076.92307692307, -48076.92307692307, 0, 
0, -96153.84615384616, 0, 0, -48076.92307692307, 0, 
-32051.28205128205, -32051.28205128205, 0, -48076.92307692307, 0, 0, 
0, 0, 0, 0, 96153.84615384616, 0, -48076.92307692307, 
48076.92307692307, 0, -16025.64102564102, 32051.28205128205, 0, 0, 
48076.92307692307, 0, 48076.92307692307, 48076.92307692307, 0, 
16025.64102564102, 16025.64102564102
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(fabs(grad_u_gold[iNode]) < 1e-10){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test criterion gradient wrt control, z
  //
  auto grad_z = eeScalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   12019.2307692307695, 16025.6410256410272, 4006.41025641025590, 
   24038.4615384615427, 8012.82051282051179, 4006.41025641025590, 
   8012.82051282051179, 4006.41025641025590, 16025.6410256410272, 
   24038.4615384615427, 8012.82051282051179, 4006.41025641025590, 
   8012.82051282051179, 48076.9230769230635, 24038.4615384615427, 
   16025.6410256410272, 24038.4615384615427, 16025.6410256410272, 
   12019.2307692307695, 16025.6410256410272, 24038.4615384615427, 
   24038.4615384615427, 8012.82051282051179, 4006.41025641025590, 
   8012.82051282051179, 16025.6410256410272, 4006.41025641025590
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = eeScalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
    -32051.2820512820472, -32051.2820512820472, -32051.2820512820472, 
    -48076.9230769230708, -48076.9230769230708, 0, -16025.6410256410236, 
    -16025.6410256410236, 32051.2820512820472, -96153.8461538461561, 
    2.61113508235193827e-13, 0, -48076.9230769230708, 0, 
    48076.9230769230708, -32051.2820512820472, 16025.6410256410236, 
    16025.6410256410236, -48076.9230769230708, 48076.9230769230708, 0, 
    -16025.6410256410236, 32051.2820512820472, -16025.6410256410236, 
    -48076.9230769230708, -2.61113508235193827e-13, 
    -48076.9230769230708, -4.89905329768894236e-14, 0, 
    -96153.8461538461561, 48076.9230769230708, 0, -48076.9230769230708, 
    16025.6410256410236, 16025.6410256410236, -32051.2820512820472, 
    3.56037847330864148e-14, 48076.9230769230708, 
    -48076.9230769230708, 7.27595761418342590e-12, 
    -7.27595761418342590e-12, 0, 0, 0, 96153.8461538461561, 0, 
    48076.9230769230708, 48076.9230769230708, 0, 96153.8461538461561, 0, 
    48076.9230769230708, 48076.9230769230708, 0, 32051.2820512820472, 
    32051.2820512820472, 32051.2820512820472, 48076.9230769230708, 0, 
    48076.9230769230708, 96153.8461538461561, 0, 0, 0, 
    -96153.8461538461561, 0, 0, -48076.9230769230708, 
    48076.9230769230708, 16025.6410256410236, -32051.2820512820472, 
    16025.6410256410236, 48076.9230769230708, -48076.9230769230708, 0, 0, 
    -48076.9230769230708, -48076.9230769230708, 32051.2820512820472, 
    -16025.6410256410236, -16025.6410256410236
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(fabs(grad_x_gold[iNode]) < 1e-10){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-10);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         ThermostaticResidual in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, ThermostaticResidual3D )
{ 
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Elliptic'>                                             \n"
    "    <ParameterList name='Penalty Function'>                                   \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                  \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Thermal Conduction'>                               \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='100.0'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Internal Elastic Energy'>                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

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
  std::vector<Plato::Scalar> t_host( tMesh->nverts() );
  Plato::Scalar disp = 0.0, dval = 0.1;
  for( auto& val : t_host ) val = (disp += dval);
  Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>
    t_host_view(t_host.data(),t_host.size());
  auto u = Kokkos::create_mirror_view_and_copy( Kokkos::DefaultExecutionSpace(), t_host_view);



  // create constraint evaluator
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::DataMap dataMap;
  Plato::Elliptic::VectorFunction<::Plato::Thermal<spaceDim>>
    tsVectorFunction(tSpatialModel, dataMap, *tParamList, tParamList->get<std::string>("PDE Constraint"));


  // compute and test constraint value
  //
  auto residual = tsVectorFunction.value(u,z);

  auto residual_Host = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy( residual_Host, residual );

  std::vector<Plato::Scalar> residual_gold = { 
    -56.66666666666666, -54.99999999999999, -16.66666666666667,
    -67.49999999999999, -21.66666666666666, -16.66666666666666,
    -17.50000000000000,  -5.000000000000000, 24.99999999999999,
    -67.50000000000000, -36.66666666666667,  -9.99999999999999,
      2.499999999999995,-24.99999999999998,  -5.000000000000021,
     11.66666666666666,  50.00000000000000,   3.333333333333323,
      5.000000000000004,  4.999999999999984, 60.00000000000004,
     69.99999999999999,  38.33333333333336,   6.666666666666675,
     16.66666666666663,  89.99999999999997,  16.66666666666668
  };

  for(int iNode=0; iNode<int(residual_gold.size()); iNode++){
    if(residual_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(residual_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residual_Host[iNode], residual_gold[iNode], 1e-13);
    }
  }


  // compute and test constraint gradient wrt state, u. (i.e., jacobian)
  //
  auto jacobian = tsVectorFunction.gradient_u(u,z);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> gold_jac_entries = {
   49.99999999999999,-16.66666666666666,-16.66666666666666,
    0.0             ,  0.0             ,  0.0             ,
    0.0             ,-16.66666666666666,-16.66666666666666,
   83.33333333333330,-25.00000000000000,-16.66666666666666,
    0.0             ,  0.0             ,  0.0             ,
  -25.00000000000000,  0.0             ,-16.66666666666666,
   33.33333333333333, -8.33333333333333,  0.0             ,
   -8.33333333333333,  0.0             ,-25.00000000000000,
  150.00000000000000,-25.00000000000000,-25.00000000000000,
    0.0             ,-25.00000000000000,-49.99999999999999,
    0.0             ,  0.0             ,  0.0
  };

  int jac_entriesSize = gold_jac_entries.size();
  for(int i=0; i<jac_entriesSize; i++){
    TEST_FLOATING_EQUALITY(jac_entriesHost(i), gold_jac_entries[i], 1.0e-15);
  }


  // compute and test gradient wrt control, z
  //
  auto gradient = tsVectorFunction.gradient_z(u,z);
  
  auto grad_entries = gradient->entries();
  auto grad_entriesHost = Kokkos::create_mirror_view( grad_entries );
  Kokkos::deep_copy(grad_entriesHost, grad_entries);

  std::vector<Plato::Scalar> gold_grad_entries = {
  -14.16666666666666,   4.166666666666666, -4.166666666666666, 
   -4.791666666666666, -4.791666666666666,  2.500000000000000, 
    6.666666666666666, 14.58333333333333,  -0.4166666666666666, 
  -13.75000000000000,  -3.125000000000000, -4.166666666666667, 
   -3.541666666666666,  0.4166666666666679, 1.249999999999997, 
   15.62500000000000,   7.708333333333336, -0.4166666666666667, 
   -4.166666666666667, -1.666666666666667,  0.4166666666666656, 
    5.833333333333334, -1.875000000000000, -1.041666666666667, 
  -16.87500000000000,  -2.083333333333334, -3.750000000000000, 
   -4.166666666666666,  4.791666666666666, 10.83333333333333, 
    4.166666666666666,  4.166666666666666,  5.833333333333333
  };

  int grad_entriesSize = gold_grad_entries.size();
  for(int i=0; i<grad_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_entriesHost(i), gold_grad_entries[i], 1.0e-14);
  }

  // compute and test gradient wrt node position, x
  //
  auto gradient_x = tsVectorFunction.gradient_x(u,z);
  
  auto grad_x_entries = gradient_x->entries();
  auto grad_x_entriesHost = Kokkos::create_mirror_view( grad_x_entries );
  Kokkos::deep_copy(grad_x_entriesHost, grad_x_entries);

  std::vector<Plato::Scalar> gold_grad_x_entries = {
  -151.666666666666686,   23.3333333333333499,  -1.66666666666667229, 
     4.99999999999999467, 19.9999999999999929, -14.9999999999999964, 
    48.3333333333333286, -11.6666666666666714,  40.0000000000000071, 
   -15.0000000000000000,  26.6666666666666607,  26.6666666666666643
};

  int grad_x_entriesSize = gold_grad_x_entries.size();
  for(int i=0; i<grad_x_entriesSize; i++){
    TEST_FLOATING_EQUALITY(grad_x_entriesHost(i), gold_grad_x_entries[i], 1.0e-13);
  }


}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalThermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, InternalThermalEnergy3D )
{ 
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Internal Thermal Energy'>                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Thermal Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Thermal Conduction'>                               \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='100.0'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

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
  ordType tNumDofs = tMesh->nverts();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumDofs);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view( u );
  Plato::Scalar disp = 0.0, dval = 0.1;
  for(ordType i=0; i<tNumDofs; i++)
  {
      u_host(i) = (disp += dval);
  }
  Kokkos::deep_copy(u, u_host);


  // create objective
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Internal Thermal Energy");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermal<spaceDim>>
    eeScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = eeScalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 611.666666666666;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  auto grad_u = eeScalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
 -113.3333333333333  , -110.0000000000000  , -33.33333333333334 , 
 -135.0000000000000  ,  -43.33333333333333 , -33.33333333333333 , 
  -35.00000000000000 ,   -9.999999999999998,  49.99999999999999 , 
 -135.0000000000000  ,  -73.33333333333334 , -19.99999999999999 , 
    4.999999999999993,  -49.99999999999996 , -10.00000000000003 , 
   23.33333333333333 ,  100.0000000000000  ,   6.666666666666636, 
   10.00000000000001 ,    9.99999999999997 , 120.0000000000001  , 
  140.0000000000000  ,   76.66666666666669 ,  13.33333333333336 , 
   33.33333333333329 ,  179.9999999999999  ,  33.33333333333336
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test criterion gradient wrt control, z
  //
  auto grad_z = eeScalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   50.87500000000000 , 47.50000000000001 , 11.87500000000000 ,
   29.33333333333334 ,  8.625000000000000,  4.416666666666666,
    9.500000000000000,  3.000000000000000, 15.29166666666666 ,
   42.25000000000000 , 17.95833333333334 ,  1.749999999999999,
    2.916666666666666, 89.91666666666667 , 44.41666666666667 ,
   13.87500000000000 , 15.95833333333333 ,  8.125000000000002,
    3.708333333333334,  9.208333333333339, 34.58333333333334 ,
   54.87499999999999 , 21.62500000000001 ,  0.916666666666669,
    2.374999999999999, 58.91666666666666 ,  7.875000000000002
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = eeScalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
    189.0000000000000, -204.3333333333334,  -96.99999999999997,
    187.5000000000000, -208.5000000000001,   22.00000000000002,
     52.50000000000001, -67.50000000000000, 101.6666666666667,
    145.8333333333333,   86.49999999999997, -35.00000000000000,
     34.66666666666666,  14.83333333333334,  60.16666666666667,
     31.33333333333334,  31.00000000000001,  17.66666666666666,
     12.33333333333334,  65.33333333333334, -17.66666666666667,
      2.999999999999999, 22.00000000000000,  -9.000000000000002,
   -105.5000000000000,  -39.66666666666666, -45.50000000000001
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, FluxPNorm3D )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                  \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Flux P-Norm'>                                        \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Flux P-Norm'/>  \n"
    "      <Parameter name='Exponent' type='double' value='12.0'/>                 \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Thermal Conduction'>                               \n"
    "        <Parameter name='Thermal Conductivity' type='double' value='100.0'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

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
  ordType tNumDofs = tMesh->nverts();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumDofs);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view( u );
  Plato::Scalar disp = 0.0, dval = 0.1;
  for(ordType i=0; i<tNumDofs; i++)
  {
      u_host(i) = (disp += dval);
  }
  Kokkos::deep_copy(u, u_host);



  // create objective
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Flux P-Norm");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermal<spaceDim>>
    scalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", U);
  auto value = scalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 444.0866631427854;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt state, u
  //
  tSolution.set("State", U);
  auto grad_u = scalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
  -121.7118130636210,       -8.402628906798824,    -2.660758498915989, 
    -0.03344798696042541,   -0.002037610215525936, -0.001781170602554662, 
    -0.003041532729695166,  -5.050930310677827e-6,  0.02320609420829947, 
   -75.88535722079789,      -1.308256330626368,    -0.00001183857667807545, 
     5.995437722669788e-6,   3.556025967524545,    -2.098814435019494, 
     0.001683242825810107,   0.005889361126491251, -0.0003613381503557746, 
     2.342203318998678e-6,   0.0002564538923140549, 0.9048692212497024, 
    14.99208220618024,       5.181659302537033,     2.904509920748619e-7, 
     6.118314328490756e-6, 186.8076430023642,       0.6349853856300433
  };

  for(int iNode=0; iNode<int(grad_u_gold.size()); iNode++){
    if(grad_u_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_u_Host[iNode]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(grad_u_Host[iNode], grad_u_gold[iNode], 1e-13);
    }
  }


  // compute and test criterion gradient wrt control, z
  //
  auto grad_z = scalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
   105.2893297588999,        6.512518326309412,     1.628129581577354, 
     0.01125162852821403,    0.0007942956716896681, 0.0004666278049140509, 
     0.001684178438260436,   0.0001822327241451453, 0.009432160604870156, 
    84.39181736274085,       0.6670203358970709,    2.010529010655542e-6, 
     2.509074030348219e-6, 107.0962504385074,       4.884922663631097, 
     0.001399891275939550,   0.002617540834021655,  0.0003206243285508645, 
     9.696548053968441e-6,   0.0005276215833734041, 0.8443271287750858, 
    24.65690798374454,       3.255913149884657,     5.522257190584811e-8, 
     8.782235515878686e-7, 104.4974671615945,       0.3333672998334975
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = scalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   539.2780958447977,      -344.3254644667109,         51.30519608139448, 
    30.35370155226329,      -13.44365133910082,         1.680525781359764, 
     9.557593722324338,      -4.257193045977536,        1.617571420834767, 
     0.06078243908477833,     0.04221148137571083,     -0.02386572317775400, 
     0.003666589190819278,    0.0007099999693335820,    0.0007931864745622613, 
     0.003251256001833290,    0.0009335797177886892,    0.00008997372650920974, 
     0.005150465829598891,    0.002200974863685567,    -0.002236017567744606, 
     4.461655107765414e-6,    0.0001204782967012946, 
    -0.0001168584633119755,  -0.04615901982588366,     -0.02941544225033991, 
     0.01672873982832506,   374.5747628696193,       -186.5714100889252, 
     5.431619301086632,       0.4840866327084146,      -3.961748259004554, 
     2.171832760560541,      -3.298465252935005e-7,     6.405629870294046e-6, 
     0.00001286593933992018, -1.777115766619513e-7, 
    -1.840364441525098e-6,   -7.574624338084612e-6, 
   -20.46907727348694,       44.81554408077304,        36.34671896862719, 
     8.398344211022307,      -1.192340496837429,        1.505360601912677, 
    -0.003055497909891946,   -0.0004267854516753923,    0.0003756639687016820, 
    -0.01084107572206273,    -0.002150213408694142,     0.003688080724534325, 
     0.0001490782155791348,  -0.00001664571881389717, 
     0.0006438775089781428,  -3.268801220809508e-6, 
     4.200746621914837e-6,    3.190216649208285e-6, 
    -0.00009288986744798501,  0.0004080010998888928, 
     0.0001365110296481171,  -0.2505471535468692,       2.887162020614315, 
    -1.173911589542639,     -61.19398924435590,         8.634730918059427, 
    13.58656329864013,      -19.64033440714168,         6.120046121120866, 
     0.04900105104713143,    -1.462766547177769e-7, 
     1.955457457226577e-7,    6.691130582506415e-8, 
    -3.073668111453556e-6,    4.342332759457572e-6, 
     4.344541561312716e-6, -861.2007333539822,        489.4700276363467, 
  -111.2467574119000,         0.09524778942965621,      1.809708187258011, 
    -1.269970591057625
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
  }
}



/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalElasticEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( DerivativeTests, Volume3D )
{ 
  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                   \n"
    "  <ParameterList name='Criteria'>                                             \n"
    "    <ParameterList name='Volume'>                                             \n"
    "      <Parameter name='Linear' type='bool' value='true'/>                     \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Volume'/>   \n"
    "      <ParameterList name='Penalty Function'>                                 \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>           \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "    <ParameterList name='Internal Elastic Energy'>                            \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>          \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/>  \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );

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


  // create mesh based displacement from host data
  //
  ordType tNumDofs = spaceDim*tMesh->nverts();
  Plato::ScalarMultiVector U("states", /*numSteps=*/1, tNumDofs);
  auto u = Kokkos::subview(U, 0, Kokkos::ALL());
  auto u_host = Kokkos::create_mirror_view( u );
  Plato::Scalar disp = 0.0, dval = 0.0001;
  for(ordType i=0; i<tNumDofs; i++)
  {
      u_host(i) = (disp += dval);
  }
  Kokkos::deep_copy(u, u_host);


  // create objective
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

  Plato::DataMap dataMap;
  std::string tMyFunction("Volume");
  Plato::Geometric::GeometryScalarFunction<::Plato::Geometrical<spaceDim>>
    volScalarFunction(tSpatialModel, dataMap, *tParamList, tMyFunction);


  // compute and test criterion value
  //
  auto value = volScalarFunction.value(z);

  Plato::Scalar value_gold = 1.0;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);


  // compute and test criterion gradient wrt control, z
  //
  auto grad_z = volScalarFunction.gradient_z(z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
    0.03125000000000000, 0.04166666666666666, 0.01041666666666667,
    0.06250000000000000, 0.02083333333333333, 0.01041666666666667, 
    0.02083333333333333, 0.01041666666666667, 0.04166666666666666, 
    0.06250000000000000, 0.02083333333333333, 0.01041666666666667, 
    0.02083333333333333, 0.1249999999999999,  0.06250000000000000, 
    0.04166666666666666, 0.06250000000000000, 0.04166666666666666, 
    0.03125000000000000, 0.04166666666666666, 0.06250000000000000, 
    0.06250000000000000, 0.02083333333333333, 0.01041666666666667, 
    0.02083333333333333, 0.04166666666666666, 0.01041666666666667
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test criterion gradient wrt node position, x
  //
  auto grad_x = volScalarFunction.gradient_x(z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
   -0.08333333333333333, -0.08333333333333333, -0.08333333333333333, 
   -0.1250000000000000, -0.1250000000000000, 0, -0.04166666666666666, 
   -0.04166666666666666, 0.08333333333333333, -0.2500000000000000, 
    0.00000000000000000, 0.0, -0.1250000000000000, 0.0, 
    0.1250000000000000, -0.08333333333333333, 0.04166666666666666, 
    0.04166666666666666, -0.1250000000000000, 0.1250000000000000, 0.0, 
   -0.04166666666666666, 0.08333333333333333, -0.04166666666666666, 
   -0.1250000000000000, 0.0, -0.1250000000000000, 
    0.0000000000000000, 0.0000000000000000,
   -0.2500000000000000, 0.1250000000000000, 0.0, -0.1250000000000000, 
    0.04166666666666666, 0.04166666666666666, -0.08333333333333333, 0.0, 
    0.1250000000000000, -0.1250000000000000, 0.000000000000000000,
    0.0000000000000000, 0.0000000000000000,
    0.0000000000000000, 0.0, 0.2500000000000000, 0.0, 
    0.1250000000000000, 0.1250000000000000,    0.0000000000000000,
    0.2500000000000000, 0.0, 0.1250000000000000, 0.1250000000000000, 0.0, 
    0.08333333333333333, 0.08333333333333333, 0.08333333333333333, 
    0.1250000000000000, 0.0, 0.1250000000000000, 0.2500000000000000, 
    0.0000000000000000, 0.0, 0.0000000000000000,
   -0.2500000000000000,  0.0000000000000000, 0.0, 
   -0.1250000000000000, 0.1250000000000000, 0.04166666666666666, 
   -0.08333333333333333, 0.04166666666666666, 0.1250000000000000, 
   -0.1250000000000000, 0.0, 0.0, -0.1250000000000000, -0.1250000000000000, 
   0.08333333333333333, -0.04166666666666666, -0.04166666666666666
  };

  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    if(grad_x_gold[iNode] == 0.0){
      TEST_ASSERT(fabs(grad_x_Host[iNode]) < 1e-13);
    } else {
      TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-13);
    }
  }
}


// Reference Strain Test
TEUCHOS_UNIT_TEST( DerivativeTests, referenceStrain3D )
{ 
  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                          \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Unobtainium'>                                        \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
    "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
    "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n" 
    "        <Parameter  name='e11' type='double' value='-0.01'/>                  \n"
    "        <Parameter  name='e22' type='double' value='-0.01'/>                  \n"
    "        <Parameter  name='e33' type='double' value=' 0.02'/>                  \n"      
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "</ParameterList>                                                              \n"
  );
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  int numCells = tMesh->nelems();
  int numVoigtTerms = Plato::SimplexMechanics<spaceDim>::mNumVoigtTerms;
  
  Plato::ScalarMultiVectorT<Plato::Scalar>
    stress("stress",numCells,numVoigtTerms);


  Plato::ScalarMultiVector elasticStrain("strain", numCells, numVoigtTerms);
  auto tHostStrain = Kokkos::create_mirror(elasticStrain);
  tHostStrain(0,0) = 0.0006; tHostStrain(1,0) = 0.006 ; tHostStrain(2,0) = 0.006 ; 
  tHostStrain(0,1) = 0.0048; tHostStrain(1,1) = 0.0048; tHostStrain(2,1) = 0.0012; 
  tHostStrain(0,2) = 0.0024; tHostStrain(1,2) =-0.0030; tHostStrain(2,2) = 0.0006; 
  tHostStrain(0,3) = 0.0072; tHostStrain(1,3) = 0.0018; tHostStrain(2,3) = 0.0018; 
  tHostStrain(0,4) = 0.003 ; tHostStrain(1,4) = 0.0030; tHostStrain(2,4) = 0.0066; 
  tHostStrain(0,5) = 0.0054; tHostStrain(1,5) = 0.0108; tHostStrain(2,5) = 0.0072; 
  
  tHostStrain(3,0) = 0.012 ; tHostStrain(4,0) = 0.006 ; tHostStrain(5,0) = 0.006 ;
  tHostStrain(3,1) =-0.0048; tHostStrain(4,1) = 0.0012; tHostStrain(5,1) = 0.0012;
  tHostStrain(3,2) = 0.0006; tHostStrain(4,2) = 0.0006; tHostStrain(5,2) = 0.0006;
  tHostStrain(3,3) =-0.0042; tHostStrain(4,3) = 0.0018; tHostStrain(5,3) = 0.0018;
  tHostStrain(3,4) = 0.0126; tHostStrain(4,4) = 0.0066; tHostStrain(5,4) = 0.0066;
  tHostStrain(3,5) = 0.0072; tHostStrain(4,5) = 0.0072; tHostStrain(5,5) = 0.0072;
  Kokkos::deep_copy(elasticStrain , tHostStrain );

  Plato::ElasticModelFactory<spaceDim> mmfactory(*tParamList);
  auto materialModel = mmfactory.create("Unobtainium");

  Plato::LinearStress<spaceDim>      voigtStress(materialModel);

  Plato::ScalarVectorT<Plato::Scalar> cellVolume("cell volume",numCells);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(int cellOrdinal)
  {
    voigtStress(cellOrdinal, stress, elasticStrain);
  }, "referenceStrain");

  // test Inherent Strain stress
  //
  auto stress_Host = Kokkos::create_mirror_view( stress );
  Kokkos::deep_copy( stress_Host, stress );

  std::vector<std::vector<Plato::Scalar>> stress_gold = { 
   { 12653.8461538462, 15884.6153846154,-9038.46153846154, 2769.23076923077, 1153.84615384615, 2076.92307692308},
   { 16807.6923076923, 15884.6153846154,-13192.3076923077, 692.307692307692, 1153.84615384615, 4153.84615384615},
   { 16807.6923076923, 13115.3846153846,-10423.0769230769, 692.307692307692, 2538.46153846154, 2769.23076923077},
   { 21423.0769230769, 8500.00000000000,-10423.0769230769,-1615.38461538462, 4846.15384615385, 2769.23076923077},
   { 16807.6923076923, 13115.3846153846,-10423.0769230769, 692.307692307692, 2538.46153846154, 2769.23076923077},
   { 16807.6923076923, 13115.3846153846,-10423.0769230769, 692.307692307692, 2538.46153846154, 2769.23076923077}
  };


  for(int iCell=0; iCell<int(stress_gold.size()); iCell++){
    for(int iVoigt=0; iVoigt<numVoigtTerms; iVoigt++){
      if(stress_gold[iCell][iVoigt] == 0.0){
        TEST_ASSERT(fabs(stress_Host(iCell,iVoigt)) < 1e-12);
      } else {
        TEST_FLOATING_EQUALITY(stress_Host(iCell,iVoigt), stress_gold[iCell][iVoigt], 1e-13);
      }
    }
  }

}


TEUCHOS_UNIT_TEST( DerivativeTests, ElastostaticResidual2D_InhomogeneousEssentialConditions )
{
    Teuchos::RCP<Teuchos::ParameterList> tElasticityParams =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                          \n"
      "  <ParameterList name='Spatial Model'>                                        \n"
      "    <ParameterList name='Domains'>                                            \n"
      "      <ParameterList name='Design Volume'>                                    \n"
      "        <Parameter name='Element Block' type='string' value='body'/>          \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>  \n"
      "      </ParameterList>                                                        \n"
      "    </ParameterList>                                                          \n"
      "  </ParameterList>                                                            \n"
      "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>           \n"
      "  <Parameter name='Physics' type='string' value='Mechanical'/>                \n"
      "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                  \n"
      "  <ParameterList name='Elliptic'>                                             \n"
      "    <ParameterList name='Penalty Function'>                                   \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                     \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                  \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-6'/>          \n"
      "    </ParameterList>                                                          \n"
      "  </ParameterList>                                                            \n"
      "  <ParameterList name='Material Models'>                                      \n"
      "    <ParameterList name='Unobtainium'>                                        \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                         \n"
      "        <Parameter name='Poissons Ratio' type='double' value='0.3'/>          \n"
      "        <Parameter name='Youngs Modulus' type='double' value='1.0e6'/>        \n"
      "      </ParameterList>                                                        \n"
      "    </ParameterList>                                                          \n"
      "  </ParameterList>                                                            \n"
      "</ParameterList>                                                              \n"
    );

    // SETUP INPUT PARAMETERS
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 3;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    Plato::Elliptic::Problem<Plato::Mechanics<tSpaceDim>>
        tElasticityProblem(*tMesh, tMeshSets, *tElasticityParams, tMachine);

    // SET ESSENTIAL/DIRICHLET BOUNDARY CONDITIONS 
    Plato::OrdinalType tDispDofX = 0;
    Plato::OrdinalType tDispDofY = 1;
    auto tNumDofsPerNode = tElasticityProblem.numDofsPerNode();
    auto tDirichletIndicesBoundaryX0 = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(*tMesh, "x0", tNumDofsPerNode, tDispDofX);
    auto tDirichletIndicesBoundaryY0 = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(*tMesh, "y0", tNumDofsPerNode, tDispDofY);
    auto tDirichletIndicesBoundaryX1 = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(*tMesh, "x1", tNumDofsPerNode, tDispDofX);

    Plato::Scalar tValueToSet = 0;
    auto tNumDirichletDofs = tDirichletIndicesBoundaryX0.size() + tDirichletIndicesBoundaryY0.size() + tDirichletIndicesBoundaryX1.size();
    Plato::ScalarVector tDirichletValues("Dirichlet Values", tNumDirichletDofs);
    Plato::LocalOrdinalVector tDirichletDofs("Dirichlet Dofs", tNumDirichletDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tDirichletValues(aIndex) = tValueToSet;
        tDirichletDofs(aIndex) = tDirichletIndicesBoundaryX0(aIndex);
    }, "set dirichlet values and indices");

    auto tOffset = tDirichletIndicesBoundaryX0.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryY0.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryY0(aIndex);
    }, "set dirichlet values and indices");

    tValueToSet = 6e-4;
    tOffset += tDirichletIndicesBoundaryY0.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX1.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX1(aIndex);
    }, "set dirichlet values and indices");

    // SOLVE ELASTOSTATICS EQUATIONS
    auto tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("Control", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    tElasticityProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);
    auto tElasticitySolution = tElasticityProblem.solution(tControl);

    // TEST RESULTS    
    const Plato::OrdinalType tTimeStep = 0;
    auto tState = tElasticitySolution.get("State");
    auto tSolution = Kokkos::subview(tState, tTimeStep, Kokkos::ALL());
    auto tHostSolution = Kokkos::create_mirror_view(tSolution);
    Kokkos::deep_copy(tHostSolution, tSolution);

    std::vector<Plato::Scalar> tGold = {0.0, 0.0, 2e-4, 0.0, 2e-4, -8.5714285714e-5, 0.0, -8.5714285714e-5, 0.0, -1.7142857143e-4, 0.0, -2.5714285714e-4, 
                                        2e-4, -2.5714285714e-4, 2e-4, -1.7142857143e-4, 4e-4, -1.7142857143e-4, 4e-4, -2.5714285714e-4, 6e-4, -2.5714285714e-4,
                                        6e-4, -1.7142857143e-4, 6e-4, -8.5714285714e-5, 4e-4, -8.5714285714e-5, 4e-4, 0.0, 6e-4, 0.0};

    constexpr Plato::Scalar tTolerance = 1e-4;
    for(Plato::OrdinalType tDofIndex=0; tDofIndex < tHostSolution.size(); tDofIndex++)
    {
        if(tGold[tDofIndex] == 0.0){
            TEST_ASSERT(fabs(tHostSolution(tDofIndex)) < 1e-12);
        } else {
            //printf("solution(%d,%d) = %.10e\n", tTimeStep, tDofIndex, tHostSolution(tDofIndex));
            TEST_FLOATING_EQUALITY(tHostSolution(tDofIndex), tGold[tDofIndex], tTolerance);
        }
    }
}

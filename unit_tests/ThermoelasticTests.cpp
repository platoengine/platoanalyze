/*!
  These unit tests are for the Thermoelastic functionality.
 \todo 
*/

#include "PlatoTestHelpers.hpp"
#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_matrix.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_teuchos.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

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

#include "Simp.hpp"
#include "Solutions.hpp"
#include "ScalarProduct.hpp"
#include "SimplexFadTypes.hpp"
#include "WorksetBase.hpp"
#include "elliptic/VectorFunction.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/Problem.hpp"
#include "StateValues.hpp"
#include "ApplyConstraints.hpp"
#include "SimplexThermomechanics.hpp"
#include "Thermomechanics.hpp"
#include "ComputedField.hpp"
#include "ImplicitFunctors.hpp"
#include "LinearThermoelasticMaterial.hpp"

#include <fenv.h>

/******************************************************************************/
/*! 
  \brief Compute value and both gradients (wrt state and control) of 
         InternalthermalEnergy in 3D.
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( ThermoelasticTests, InternalThermoelasticEnergy3D )
{ 
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
  Plato::ScalarMultiVector states("states", /*numSteps=*/1, tNumDofs);
  auto state = Kokkos::subview(states, 0, Kokkos::ALL());
  Plato::ScalarVector z("control", tNumDofs);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(const int & aNodeOrdinal)
  {
     z(aNodeOrdinal) = 1.0;

     state(aNodeOrdinal*tNumDofsPerNode+0) = (1e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+1) = (2e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+2) = (3e-7)*aNodeOrdinal;
     state(aNodeOrdinal*tNumDofsPerNode+3) = (4e-7)*aNodeOrdinal;

  }, "state");


  // create material model
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                         \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>          \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>                  \n"
    "  <ParameterList name='Elliptic'>                                            \n"
    "    <ParameterList name='Penalty Function'>                                  \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                    \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>                 \n"
    "      <Parameter name='Minimum Value' type='double' value='0.0'/>            \n"
    "    </ParameterList>                                                         \n"
    "  </ParameterList>                                                           \n"
    "  <ParameterList name='Criteria'>                                            \n"
    "    <ParameterList name='Internal Thermoelastic Energy'>                     \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>         \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Thermoelastic Energy'/>  \n"
    "      <ParameterList name='Penalty Function'>                                \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>               \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>          \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                  \n"
    "      </ParameterList>                                                       \n"
    "    </ParameterList>                                                         \n"
    "  </ParameterList>                                                           \n"
    "  <ParameterList name='Spatial Model'>                                        \n"
    "    <ParameterList name='Domains'>                                            \n"
    "      <ParameterList name='Design Volume'>                                    \n"
    "        <Parameter name='Element Block' type='string' value='body'/>          \n"
    "        <Parameter name='Material Model' type='string' value='Beef Jerky'/>   \n"
    "      </ParameterList>                                                        \n"
    "    </ParameterList>                                                          \n"
    "  </ParameterList>                                                            \n"
    "  <ParameterList name='Material Models'>                                      \n"
    "    <ParameterList name='Beef Jerky'>                                          \n"
    "      <ParameterList name='Thermoelastic'>                                     \n"
    "        <ParameterList name='Elastic Stiffness'>                               \n"
    "          <Parameter  name='Poissons Ratio' type='double' value='0.3'/>        \n"
    "          <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>     \n"
    "        </ParameterList>                                                       \n"
    "        <Parameter  name='Thermal Expansivity' type='double' value='1.0e-5'/>  \n"
    "        <Parameter  name='Thermal Conductivity' type='double' value='910.0'/>  \n"
    "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>   \n"
    "      </ParameterList>                                                         \n"
    "    </ParameterList>                                                           \n"
    "  </ParameterList>                                                           \n"
    "</ParameterList>                                                             \n"
  );

  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *params);
  Plato::Elliptic::VectorFunction<::Plato::Thermomechanics<spaceDim>>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));


  // compute and test constraint value
  //
  auto residual = vectorFunction.value(state, z);

  auto residualHost = Kokkos::create_mirror_view( residual );
  Kokkos::deep_copy(residualHost, residual);

  std::vector<Plato::Scalar> residual_gold = { 
    -74678.38301282050,    -59614.82211538460,     -78204.58653846153,
    -0.002062666666666666, -69710.05929487177,     -62980.04006410255,
    -66346.07051282052,    -0.002002000000000000,   6250.406250000000,
    -25480.55048076922,    -6731.394230769230,     -0.0006066666666666668,
    -80767.10576923075,    -38781.71794871794,     -102564.2275641025,
    -0.002457000000000000, -12659.43349358974,     -12820.45032051281,
    -481.6546474358953,    -0.0007886666666666667, -10255.82692307692,
    -3365.665865384615,    -13301.58413461538,     -0.0006066666666666667,
    -6248.854166666652,    -161.3189102564033,     -26282.13461538462
  };

  for(int iVal=0; iVal<int(residual_gold.size()); iVal++){
    if(residual_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(residualHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(residualHost[iVal], residual_gold[iVal], 1e-13);
    }
  }

  // compute and test constraint value
  //
  auto jacobian = vectorFunction.gradient_u(state, z);

  auto jac_entries = jacobian->entries();
  auto jac_entriesHost = Kokkos::create_mirror_view( jac_entries );
  Kokkos::deep_copy(jac_entriesHost, jac_entries);

  std::vector<Plato::Scalar> jacobian_gold = { 
   3.52564102564102478e10, 0.00000000000000000,    0.00000000000000000,    52083.3333333333285, 
   0.00000000000000000,    3.52564102564102478e10, 0.00000000000000000,    52083.3333333333285,
   0.00000000000000000,    0.00000000000000000,    3.52564102564102478e10, 52083.3333333333285,
   0.00000000000000000,    0.00000000000000000,    0.00000000000000000,    454.999999999999943,
  -6.41025641025640965e9,  3.20512820512820482e9,  0.00000000000000000,    0.00000000000000000,
   4.80769230769230652e9, -2.24358974358974304e10, 4.80769230769230652e9,  52083.3333333333285, 
   0.00000000000000000,    3.20512820512820482e9, -6.41025641025640965e9,  0.00000000000000000, 
   0.00000000000000000,    0.00000000000000000,    0.00000000000000000,   -151.666666666666657,
  -6.41025641025640965e9,  0.00000000000000000,    3.20512820512820482e9,  0.00000000000000000,
   0.00000000000000000,   -6.41025641025640965e9,  3.20512820512820482e9,  0.00000000000000000,
   4.80769230769230652e9,  4.80769230769230652e9, -2.24358974358974304e10, 52083.3333333333285,
   0.00000000000000000,    0.00000000000000000,    0.00000000000000000,   -151.666666666666657,
   0.00000000000000000,    3.20512820512820482e9,  3.20512820512820482e9,  0.00000000000000000,
   4.80769230769230652e9,  0.00000000000000000,   -8.01282051282051086e9,  26041.6666666666642,
   4.80769230769230652e9, -8.01282051282051086e9,  0.00000000000000000,    26041.6666666666642,
   0.00000000000000000,    0.00000000000000000,   0.00000000000000000,     0.00000000000000000
  };

  for(int iVal=0; iVal<int(jacobian_gold.size()); iVal++){
    if(jacobian_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(jac_entriesHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(jac_entriesHost[iVal], jacobian_gold[iVal], 1e-13);
    }
  }

  // compute and test constraint gradient_z
  //
  auto gradient_z = vectorFunction.gradient_z(state, z);

  auto gradz_entries = gradient_z->entries();
  auto gradz_entriesHost = Kokkos::create_mirror_view( gradz_entries );
  Kokkos::deep_copy(gradz_entriesHost, gradz_entries);

  std::vector<Plato::Scalar> gradient_z_gold = { 
   -18669.5957532051252, -14903.7055288461488, -19551.1466346153829, -0.000515666666666666552,
   -2604.08854166666652,  8012.67988782051179,  4206.79326923076951,  0.000151666666666666649,
    1562.59114583333439, -6370.14803685897277, -1682.82772435897550, -0.000151666666666666649,
   -2804.38040865384437, -200.364783653846530, -4927.94711538461343, -0.000174416666666666633
  };

  for(int iVal=0; iVal<int(gradient_z_gold.size()); iVal++){
    if(gradient_z_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(gradz_entriesHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(gradz_entriesHost[iVal], gradient_z_gold[iVal], 1e-13);
    }
  }

  // compute and test constraint gradient_x
  //
  auto gradient_x = vectorFunction.gradient_x(state, z);

  auto gradx_entries = gradient_x->entries();
  auto gradx_entriesHost = Kokkos::create_mirror_view( gradx_entries );
  Kokkos::deep_copy(gradx_entriesHost, gradx_entries);

  std::vector<Plato::Scalar> gradient_x_gold = { 
   -138461.538461538410,    -151923.076923076878,     -319230.769230769132,
   -0.00552066666666666504,  47435.8974358974156,     -33333.3333333333358,
    55769.2307692307368,     0.000849333333333333342, -641.025641025629739,
    1923.07692307693378,     -11538.4615384615317,    -0.0000606666666666664969,
   -1282.05128205127858,    -18909.6314102564065,     -18589.7435897435789,
    0.000181999999999999979, 40063.4775641025481,      66666.6666666666570,
    11217.4487179487187,     0.000727999999999999915, -26282.0512820512740,
   -28525.1410256410236,    -1282.05128205128494,     -0.000545999999999999936,
   -77564.1025641025335,     23717.9487179487187,      165705.857371794817,
    0.00175933333333333280, -641.025641025644290,     -18589.7435897435826,
   -58012.4663461538148,    -0.000424666666666666617,  44871.0657051281887,
    41666.3124999999854,     55769.2307692307659,      0.00145599999999999983,
   -85897.4358974358765,     5449.21794871795646,     -3525.28685897435935,
   -0.000545999999999999827, 6089.24358974358620,      30128.2051282051179,
    61538.6073717948602,     0.000970666666666666444,  39743.2355769230635,
    20192.1618589743448,     14743.5897435897496,      0.000970666666666666336
  };

  for(int iVal=0; iVal<int(gradient_x_gold.size()); iVal++){
    if(gradient_x_gold[iVal] == 0.0){
      TEST_ASSERT(fabs(gradx_entriesHost[iVal]) < 1e-12);
    } else {
      TEST_FLOATING_EQUALITY(gradx_entriesHost[iVal], gradient_x_gold[iVal], 1e-13);
    }
  }

  // create objective
  //
  std::string tMyFunction("Internal Thermoelastic Energy");
  Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermomechanics<spaceDim>>
    scalarFunction(tSpatialModel, tDataMap, *params, tMyFunction);

  // compute and test objective value
  //
  Plato::Solutions tSolution;
  tSolution.set("State", states);
  auto value = scalarFunction.value(tSolution, z);

  Plato::Scalar value_gold = 3.99325969691123239;
  TEST_FLOATING_EQUALITY(value, value_gold, 1e-13);

  // compute and test objective gradient wrt state, u
  //
  tSolution.set("State", states);
  auto grad_u = scalarFunction.gradient_u(tSolution, z, /*stepIndex=*/0);

  auto grad_u_Host = Kokkos::create_mirror_view( grad_u );
  Kokkos::deep_copy( grad_u_Host, grad_u );

  std::vector<Plato::Scalar> grad_u_gold = { 
   -149357.8701923077,   -119230.2067307692,   -156409.7147435897,
   -0.1760003333333333,  -139421.5977564102,   -125960.8092948718,
   -132692.2243589743,   -0.2540039999999999,   12500.40624999999,
   -50961.31971153845,   -13462.16346153846,   -0.06371333333333332,
   -161536.3365384615,   -77563.76923076922,   -205128.3301282051,
   -0.3903306666666666,  -25319.68990384614,   -25640.96314102563,
   -962.4238782051143,   -0.1682440000000000,  -20512.23717948717,
   -6731.050480769230,   -26602.86618589742,   -0.07413000000000002,
   -12498.85416666665,   -321.5753205128094,   -52564.18589743588,
   -0.08460733333333334, -21473.91105769230,    11537.62820512823,
   -13140.64022435896,   -0.05244733333333334, -109934.6370192308,
    44872.06570512819,    61218.95913461538,   -0.2585966666666666
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
  tSolution.set("State", states);
  auto grad_z = scalarFunction.gradient_z(tSolution, z);

  auto grad_z_Host = Kokkos::create_mirror_view( grad_z );
  Kokkos::deep_copy( grad_z_Host, grad_z );

  std::vector<Plato::Scalar> grad_z_gold = {
    0.3006646220307975,  0.2828518382060641,  0.07071297869214102,
    0.2014571253646834,  0.06703482120772307, 0.03167447648441282,
    0.05844522383912947, 0.02116164326532563, 0.1211208163690948,
    0.3125138830947090,  0.1180923674785154,  0.02885361640063332,
    0.04527148502001922, 0.5649346204235975,  0.2730600666633808,
    0.1003915937669949,  0.1293972579725719,  0.09587150695223083,
    0.02509555855275397, 0.05871878443368219, 0.2281261532124168,
    0.3215640965314669,  0.1200080860211962,  0.006538731463594921,
    0.02452832023762705, 0.3393603844997411,  0.04580963872672825
  };

  for(int iNode=0; iNode<int(grad_z_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_z_Host[iNode], grad_z_gold[iNode], 1e-13);
  }

  // compute and test objective gradient wrt node position, x
  //
  tSolution.set("State", states);
  auto grad_x = scalarFunction.gradient_x(tSolution, z);
  
  auto grad_x_Host = Kokkos::create_mirror_view( grad_x );
  Kokkos::deep_copy(grad_x_Host, grad_x);

  std::vector<Plato::Scalar> grad_x_gold = {
    1.415436782967118,     -1.088392815648369,   -0.5040375881616612,
    1.447588668646153,     -1.250091771703753,    0.1578841810878153,
    0.1306132382529746,    -0.3847761196036412,   0.5916655752193332,
    1.020192508092308,      0.5763455956072205,  -0.2397432740062565,
   -0.2497054688307384,     0.04301339975588712,  0.4116322178948820,
   -0.02582992877120004,    0.1731726564366770,   0.1257363320594461,
   -9.21562170871081942e-5, 0.4083635774612511,  -0.1133975593030360,
   -0.02791564539653321,    0.1617290307673025,  -0.07330046252834882,
   -0.6810599201684920,    -0.1161559894293126,  -0.2451581005350564,
  };



  for(int iNode=0; iNode<int(grad_x_gold.size()); iNode++){
    TEST_FLOATING_EQUALITY(grad_x_Host[iNode], grad_x_gold[iNode], 1e-11);
  }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, VolAvgStressPNormAxial_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tNumElemX = 1;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    const Plato::Scalar tBoxWidth = 5.0;
    auto tMesh = PlatoUtestHelpers::build_3d_box_mesh(tBoxWidth,tBoxWidth,tBoxWidth,tNumElemX,tNumElemY,tNumElemZ);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Mechanical'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "      <Parameter name='Iterations' type='int' value='500'/>                              \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='10.0e0'/>                 \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
     "   <ParameterList name='volume avg stress pnorm'> \n"
     "     <Parameter name='Type' type='string' value='Division' /> \n"
     "       <Parameter name='Numerator' type='string' value='pnorm numerator' /> \n"
     "       <Parameter name='Denominator' type='string' value='pnorm denominator' /> \n"
     "   </ParameterList> \n"
     "   <ParameterList name='pnorm numerator'> \n"
     "     <Parameter name='Type' type='string' value='Scalar Function' /> \n"
     "     <Parameter name='Scalar Function Type' type='string' value='Stress P-Norm' /> \n"
     "     <ParameterList name='Penalty Function'> \n"
     "       <Parameter name='Type' type='string' value='SIMP' /> \n"
     "       <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "     </ParameterList> \n"
     "     <Parameter name='Function' type='string' value='x/5.0' /> \n"
     "     <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "   </ParameterList> \n"
     "   <ParameterList name='pnorm denominator'> \n"
     "     <Parameter name='Type' type='string' value='Scalar Function' /> \n"
     "     <Parameter name='Scalar Function Type' type='string' value='vol avg stress p-norm denominator' /> \n"
     "     <ParameterList name='Penalty Function'> \n"
     "       <Parameter name='Type' type='string' value='SIMP' /> \n"
     "       <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "     </ParameterList> \n"
     "     <Parameter name='Function' type='string' value='x/5.0' /> \n"
     "     <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "   </ParameterList> \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Disp Boundary Condition'>                             \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                   \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.1'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::Mechanics<tSpaceDim>;

    Plato::Elliptic::Problem<PhysicsT> tEllipticProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tEllipticProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tEllipticProblem.solution(tControls);


    const Plato::Scalar tSpatialWeightingFactor = 0.5;
    const Plato::Scalar tElasticModulus = 10.0;
    const Plato::Scalar tStrain = 0.1 / tBoxWidth;
    const Plato::Scalar tStress = tElasticModulus * tStrain;
    const Plato::Scalar tBoxVolume = tSpatialWeightingFactor * (tBoxWidth*tBoxWidth*tBoxWidth);

    constexpr Plato::Scalar tTolerance = 1e-4;

    std::string tCriterionName1("volume avg stress pnorm");
    auto tCriterionValue1 = tEllipticProblem.criterionValue(tControls, tCriterionName1);
    TEST_FLOATING_EQUALITY(tCriterionValue1, tStress, tTolerance);

    std::string tCriterionName2("pnorm numerator");
    auto tCriterionValue2 = tEllipticProblem.criterionValue(tControls, tCriterionName2);
    TEST_FLOATING_EQUALITY(tCriterionValue2, tStress * tBoxVolume, tTolerance);

    std::string tCriterionName3("pnorm denominator");
    auto tCriterionValue3 = tEllipticProblem.criterionValue(tControls, tCriterionName3);
    TEST_FLOATING_EQUALITY(tCriterionValue3, tBoxVolume, tTolerance);

    // auto tCriterionGrad = tEllipticProblem.criterionGradient(tControls, tCriterionName);
    // std::vector<Plato::Scalar> tGold = { -8.23158e-01,-2.74211e-01,-2.74205e-01,-2.74211e-01,-5.46915e-01,
    //                                      -1.09598e+00,-5.46915e-01,-1.09091e+00,-1.07737e+00,-5.40880e-01,
    //                                      -1.08590e+00,-5.40880e-01,-1.05793e+00,-5.26599e-01,-1.04844e+00,
    //                                      -5.26599e-01,-1.07226e+00,-5.33831e-01,-1.06304e+00,-5.33831e-01,
    //                                      -5.04852e-01,-1.00493e+00,-5.04852e-01,-1.01433e+00,-5.12007e-01,
    //                                      -1.01919e+00,-1.03386e+00,-5.19301e-01,-1.04332e+00,-5.19301e-01,
    //                                      -5.12007e-01,-1.02878e+00,-4.98050e-01,-1.00060e+00,-4.98050e-01,
    //                                      -9.91065e-01,-9.80656e-01,-4.92349e-01,-9.88215e-01,-4.92349e-01,
    //                                      -2.44243e-01,-7.35025e-01,-2.44243e-01,-2.43315e-01};
    // auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    // Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    // TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    // for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    // {
    //     //printf("%12.5e\n", tHostGrad(tIndex));
    //     TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    // }

    // 6. Output Data
    if (false)
    {
        tEllipticProblem.output("VolAvgStressPNormAxial_3D");
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, VolAvgStressPNormShear_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tNumElemX = 1;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    const Plato::Scalar tBoxWidth = 5.0;
    auto tMesh = PlatoUtestHelpers::build_3d_box_mesh(tBoxWidth,tBoxWidth,tBoxWidth,tNumElemX,tNumElemY,tNumElemZ);
    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                                     \n"
      "  <ParameterList name='Spatial Model'>                                                   \n"
      "    <ParameterList name='Domains'>                                                       \n"
      "      <ParameterList name='Design Volume'>                                               \n"
      "        <Parameter name='Element Block' type='string' value='body'/>                     \n"
      "        <Parameter name='Material Model' type='string' value='Unobtainium'/>             \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <Parameter name='Physics'          type='string'  value='Mechanical'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Solver Package' type='string' value='amesos2'/>                   \n"
      "      <Parameter name='Iterations' type='int' value='500'/>                              \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Elastic'>                                    \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.1'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e1'/>                  \n"
      "      </ParameterList>                                                                   \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
     "   <ParameterList name='volume avg stress pnorm'> \n"
     "     <Parameter name='Type' type='string' value='Division' /> \n"
     "       <Parameter name='Numerator' type='string' value='pnorm numerator' /> \n"
     "       <Parameter name='Denominator' type='string' value='pnorm denominator' /> \n"
     "   </ParameterList> \n"
     "   <ParameterList name='pnorm numerator'> \n"
     "     <Parameter name='Type' type='string' value='Scalar Function' /> \n"
     "     <Parameter name='Scalar Function Type' type='string' value='Stress P-Norm' /> \n"
     "     <ParameterList name='Penalty Function'> \n"
     "       <Parameter name='Type' type='string' value='SIMP' /> \n"
     "       <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "     </ParameterList> \n"
     "     <Parameter name='Function' type='string' value='1.0*x/5.0' /> \n"
     "     <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "   </ParameterList> \n"
     "   <ParameterList name='pnorm denominator'> \n"
     "     <Parameter name='Type' type='string' value='Scalar Function' /> \n"
     "     <Parameter name='Scalar Function Type' type='string' value='vol avg stress p-norm denominator' /> \n"
     "     <ParameterList name='Penalty Function'> \n"
     "       <Parameter name='Type' type='string' value='SIMP' /> \n"
     "       <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "     </ParameterList> \n"
     "     <Parameter name='Function' type='string' value='1.0*x/5.0' /> \n"
     "     <Parameter name='Exponent' type='double' value='1.0' /> \n"
     "   </ParameterList> \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition1'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition1'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.1'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Fixed Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y1'/>                         \n"
      "       <Parameter  name='Value'    type='double' value='0.1'/>                           \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::Mechanics<tSpaceDim>;

    Plato::Elliptic::Problem<PhysicsT> tEllipticProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tEllipticProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tEllipticProblem.solution(tControls);


    const Plato::Scalar tSpatialWeightingFactor = 0.5;
    const Plato::Scalar tElasticModulus = 10.0;
    const Plato::Scalar tPoissonsRatio  = 0.1;
    const Plato::Scalar tShearModulus   = tElasticModulus / (2.0 * (1.0 + tPoissonsRatio));
    const Plato::Scalar tStrain = 2.0 * 0.1 / tBoxWidth;
    const Plato::Scalar tStress = tShearModulus * tStrain;
    const Plato::Scalar tBoxVolume = tSpatialWeightingFactor * (tBoxWidth*tBoxWidth*tBoxWidth);

    constexpr Plato::Scalar tTolerance = 1e-4;

    std::string tCriterionName1("volume avg stress pnorm");
    auto tCriterionValue1 = tEllipticProblem.criterionValue(tControls, tCriterionName1);
    TEST_FLOATING_EQUALITY(tCriterionValue1, tStress, tTolerance);

    std::string tCriterionName2("pnorm numerator");
    auto tCriterionValue2 = tEllipticProblem.criterionValue(tControls, tCriterionName2);
    TEST_FLOATING_EQUALITY(tCriterionValue2, tStress * tBoxVolume, tTolerance);

    std::string tCriterionName3("pnorm denominator");
    auto tCriterionValue3 = tEllipticProblem.criterionValue(tControls, tCriterionName3);
    TEST_FLOATING_EQUALITY(tCriterionValue3, tBoxVolume, tTolerance);

    // 6. Output Data
    if (false)
    {
        tEllipticProblem.output("VolAvgStressPNormShear_3D");
    }
}
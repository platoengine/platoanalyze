/*
 * ThermoplasticityTests.cpp
 *
 *  Created on: Jan 25, 2021
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoUtilities.hpp"
#include "PlatoTestHelpers.hpp"
#include "Plato_Diagnostics.hpp"

#include "PlasticityProblem.hpp"

#include <Teuchos_XMLParameterListCoreHelpers.hpp>

namespace ThermoplasticityTests
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_SimplySupportedBeamTractionForce2D_Elastic)
{
    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,10,2);
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.24'/>                   \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e7'/>                  \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='50.0'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e-20'/> \n"
      "        <Parameter  name='Reference Temperature' type='double' value='10.0'/>            \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0'/>        \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0'/>        \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1e10'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                          \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-9'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='10'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "   <ParameterList  name='Mechanical Natural Boundary Conditions'>                        \n"
      "   <ParameterList  name='Traction Vector Boundary Condition'>                            \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{0.0, -100.0}'/>            \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_Y1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{0.0e-9}'/>                 \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_Y1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                              \n"
      "       <Parameter  name='Sides'    type='string' value='Pinned'/>                        \n"
      "       <Parameter  name='Function'    type='string' value='110.0'/>                      \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    Omega_h::LOs tPinnedNodeLOs({32});
    tMeshSets[Omega_h::NODE_SET]["Pinned"] = tPinnedNodeLOs;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls).State;

    // 5. Test results
    Plato::ScalarMultiVector tPressure("Pressure", tSolution.extent(0), tNumVertices);
    Plato::ScalarMultiVector tDisplacements("Displacements", tSolution.extent(0), tNumVertices * tSpaceDim);
    Plato::blas2::extract<PhysicsT::mNumDofsPerNode, PhysicsT::mPressureDofOffset>(tSolution, tPressure);
    Plato::blas2::extract<PhysicsT::mNumDofsPerNode, tSpaceDim>(tNumVertices, tSolution, tDisplacements);

    // 5.1 test pressure
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostPressure = Kokkos::create_mirror(tPressure);
    Kokkos::deep_copy(tHostPressure, tPressure);
    std::vector<std::vector<Plato::Scalar>> tGoldPress =
        {
         {-2.115362e+02, -1.874504e+03, -2.294189e+02, -1.516672e+03, -2.785281e+03, -2.925495e+03, -2.970340e+03, -4.293099e+02,
          -1.685521e+03, -5.322904e+02, -1.665030e+03, -2.835582e+03, -6.988780e+02, -1.668066e+03, -2.687101e+03, -2.258380e+03,
          -2.495897e+03, -1.672543e+03, -9.116663e+02, -1.675849e+03, -1.168386e+03, -1.974995e+03, -1.677669e+03, -1.470044e+03,
          -1.702233e+03, -1.860586e+03, -1.668134e+03, -1.143118e+03, -1.319865e+03, -1.653114e+03, -2.204908e+03, -1.995014e+03,
          -2.705687e+03}
        };
    Plato::OrdinalType tTimeStep = 0;
    for(Plato::OrdinalType tOrdinal=0; tOrdinal< tPressure.extent(1); tOrdinal++)
    {
        //printf("X(%d,%d) = %e\n", tTimeStep, tOrdinal, tHostPressure(tTimeStep, tOrdinal));
        TEST_FLOATING_EQUALITY(tHostPressure(tTimeStep, tOrdinal), tGoldPress[0][tOrdinal], tTolerance);
    }

    // 5.2 test displacement
    auto tHostDisplacements = Kokkos::create_mirror(tDisplacements);
    Kokkos::deep_copy(tHostDisplacements, tDisplacements);
    std::vector<std::vector<Plato::Scalar>> tGoldDisp =
        {
         {0.0, -6.267770e-02, 0.0, -6.250715e-02, 5.054901e-04, -6.174772e-02, -3.494325e-04, -6.164854e-02, -1.189951e-03,
          -6.163677e-02, 0.0, -6.243005e-02, -2.395852e-03, -5.908745e-02, 9.381758e-04, -5.919208e-02, -7.291411e-04, -5.909716e-02,
          1.326328e-03, -5.503616e-02, -1.099687e-03, -5.494402e-02, -3.525908e-03, -5.492911e-02, 1.629318e-03, -4.941788e-02,  -1.472318e-03,
          -4.933201e-02, -4.573797e-03, -4.931350e-02, -6.306177e-03, -3.454268e-02, -5.510012e-03, -4.243363e-02, -1.845476e-03, -4.245746e-02,
          1.819180e-03, -4.253584e-02, -2.219095e-03, -3.457328e-02, 1.868041e-03, -3.464274e-02, -6.934208e-03, -2.594957e-02, -2.593272e-03,
          -2.598862e-02, 1.747752e-03, -2.604802e-02, -2.966076e-03, -1.706881e-02, 1.432299e-03, -1.711426e-02, -7.365046e-03, -1.702033e-02,
          -7.602023e-03, 1.234104e-04,  -7.582309e-03, -8.182097e-03, -3.335936e-03, -8.239034e-03, 8.764536e-04, -8.256626e-03, -3.587180e-03,
          1.188541e-05, 0.0, 0.0}
        };
    for(Plato::OrdinalType tOrdinal=0; tOrdinal< tDisplacements.extent(1); tOrdinal++)
    {
        //printf("X(%d,%d) = %e\n", tTimeStep, tOrdinal, tHostDisplacements(tTimeStep, tOrdinal));
        TEST_FLOATING_EQUALITY(tHostDisplacements(tTimeStep, tOrdinal), tGoldDisp[0][tOrdinal], tTolerance);
    }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.saveStates("SimplySupportedBeamTractionThermoPlasticity2D_Elastic");
    }
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_CantileverBeamTractionForce2D_Plastic)
{
    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 2;
    const Plato::OrdinalType tNumElemX = 10;
    const Plato::OrdinalType tNumElemY = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,tNumElemX,tNumElemY);
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='100.0'/>                 \n"
      "    <Parameter  name='Temperature Scaling' type='double' value='100.0'/>                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='75.0e3'/>                 \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='180.'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='2.32e-5'/> \n"
      "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>             \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='2.0e3'/>      \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e-2'/>     \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='344.0'/>             \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-8'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                          \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-8'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='6'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='6'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='25'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "   <ParameterList  name='Mechanical Natural Boundary Conditions'>                        \n"
      "   <ParameterList  name='Traction Vector Boundary Condition'>                            \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{0.0, -16.0}'/>             \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{1.0e3}'/>                  \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                              \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.0'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls).State;

    // 5. Test results
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostSolution = Kokkos::create_mirror(tSolution);
    Kokkos::deep_copy(tHostSolution, tSolution);
    std::vector<std::vector<Plato::Scalar>> tGoldSolution =
        {
         {   0.00000e+00,  0.00000e+00,  0.00000e+00, -9.04362e-01, 
             0.00000e+00,  0.00000e+00,  0.00000e+00,  3.03958e-01, 
            -4.63572e-03, -5.05137e-03,  4.62963e-02, -1.10334e+00, 
            -1.75058e-04, -4.83553e-03,  4.62963e-02, -8.35101e-02, 
             4.24129e-03, -4.91978e-03,  4.62963e-02,  9.64723e-01, 
             0.00000e+00,  0.00000e+00,  0.00000e+00,  1.37128e+00, 
             8.52476e-03, -1.81340e-02,  9.25926e-02,  9.08600e-01, 
            -8.33396e-03, -1.84871e-02,  9.25926e-02, -9.63406e-01, 
             1.14822e-04, -1.81708e-02,  9.25926e-02, -4.87695e-02, 
            -1.14753e-02, -3.93501e-02,  1.38889e-01, -8.72921e-01, 
             4.67306e-04, -3.89958e-02,  1.38889e-01, -7.20877e-02, 
             1.23786e-02, -3.88950e-02,  1.38889e-01,  7.75309e-01, 
            -1.40186e-02, -6.67581e-02,  1.85185e-01, -7.85507e-01, 
             9.55764e-04, -6.63507e-02,  1.85185e-01, -1.03955e-01, 
             1.58948e-02, -6.61607e-02,  1.85185e-01,  6.24374e-01, 
             2.19648e-02, -1.36549e-01,  2.77778e-01,  3.36290e-01, 
             1.90935e-02, -9.90219e-02,  2.31481e-01,  4.80792e-01, 
             1.58876e-03, -9.93011e-02,  2.31481e-01, -1.29832e-01, 
            -1.59510e-02, -9.97607e-02,  2.31481e-01, -6.93724e-01, 
             2.36042e-03, -1.36916e-01,  2.77778e-01, -1.56718e-01, 
            -1.72787e-02, -1.37427e-01,  2.77778e-01, -6.02914e-01, 
             2.45094e-02, -1.77810e-01,  3.24074e-01,  1.91719e-01, 
             3.27136e-03, -1.78265e-01,  3.24074e-01, -1.83380e-01, 
            -1.80018e-02, -1.78827e-01,  3.24074e-01, -5.11391e-01, 
             4.32178e-03, -2.22414e-01,  3.70370e-01, -2.10001e-01, 
            -1.81192e-02, -2.23032e-01,  3.70370e-01, -4.20976e-01, 
             2.67276e-02, -2.21872e-01,  3.70370e-01,  4.79151e-02, 
             3.01400e-02, -3.14644e-01,  4.62963e-01, -1.78738e-01, 
             2.86206e-02, -2.67802e-01,  4.16667e-01, -9.47316e-02, 
             5.52150e-03, -2.68418e-01,  4.16667e-01, -2.45149e-01, 
            -1.76110e-02, -2.69100e-01,  4.16667e-01, -3.54556e-01, 
             6.84697e-03, -3.15307e-01,  4.62963e-01, -2.62759e-01, 
            -1.64809e-02, -3.16034e-01,  4.62963e-01, -3.29448e-01}
        };
    Plato::OrdinalType tTimeStep = 4;
    for(Plato::OrdinalType tOrdinal=0; tOrdinal< tSolution.extent(1); tOrdinal++)
    {
        const Plato::Scalar tValue = std::abs(tHostSolution(tTimeStep, tOrdinal)) < 1.0e-14 ? 0.0 : tHostSolution(tTimeStep, tOrdinal);
        TEST_FLOATING_EQUALITY(tValue, tGoldSolution[0][tOrdinal], tTolerance);
    }

    // Plato::OrdinalType tIdx = 0;
    // Plato::OrdinalType tDispDofX = 0;
    // Plato::OrdinalType tDispDofY = 1;
    // Plato::OrdinalType tTemperatureDof = 2;
    // Plato::OrdinalType tPressureDof = 3;
    // Plato::OrdinalType tNumDofsPerNode = 4;
    // for (Plato::OrdinalType tIndexK = 0; tIndexK < tNumVertices; tIndexK++)
    // {
    //   tIdx = tIndexK * tNumDofsPerNode + tDispDofX;       printf("%12.5e, ",  tHostSolution(tTimeStep, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tDispDofY;       printf("%12.5e, ",  tHostSolution(tTimeStep, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tTemperatureDof; printf("%12.5e, ",  tHostSolution(tTimeStep, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tPressureDof;    printf("%12.5e, \n", tHostSolution(tTimeStep, tIdx));
    // }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.saveStates("CantileverBeamTractionForce2D_Plastic");
    }
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_CantileverBeamTractionForce3D_Plastic)
{
    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tNumElemX = 10;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = PlatoUtestHelpers::build_3d_box_mesh(10.0,1.0,1.0,tNumElemX,tNumElemY,tNumElemZ);
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='100.0'/>                 \n"
      "    <Parameter  name='Temperature Scaling' type='double' value='100.0'/>                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='75.0e3'/>                 \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='180.'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='2.32e-5'/> \n"
      "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>             \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='2.0e3'/>      \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e-2'/>     \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='344.0'/>             \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-8'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                          \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-8'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='6'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='6'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='25'/>                  \n"
      "    <Parameter name='Stopping Tolerance' type='double' value='1e-8'/>                    \n"
      "    <Parameter name='Current Residual Norm Stopping Tolerance' type='double' value='1e-8'/> \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "   <ParameterList  name='Mechanical Natural Boundary Conditions'>                        \n"
      "   <ParameterList  name='Traction Vector Boundary Condition'>                            \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{0.0, 16.0, 0.0}'/>         \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{1.0e3}'/>                  \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z1'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='3'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.0'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls).State;

    // 5. Test results
    constexpr Plato::Scalar tTolerance = 1e-4;
    auto tHostSolution = Kokkos::create_mirror(tSolution);
    Kokkos::deep_copy(tHostSolution, tSolution);
    std::vector<std::vector<Plato::Scalar>> tGoldSolution =
        {
         {    0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  9.00297e-01, 
              0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00,  5.81583e-01, 
              0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00, -1.78310e+00, 
              0.00000e+00,  0.00000e+00,  0.00000e+00,  0.00000e+00, -1.47559e+00, 
             -3.06131e-03,  3.65583e-03,  0.00000e+00,  4.62963e-02, -9.27613e-01, 
             -2.44615e-03,  3.52869e-03,  0.00000e+00,  4.62963e-02, -1.07992e+00, 
              3.34423e-03,  3.52614e-03,  0.00000e+00,  4.62963e-02,  9.96540e-01, 
              3.65007e-03,  3.41307e-03,  0.00000e+00,  4.62963e-02,  1.17256e+00, 
              6.73491e-03,  1.30329e-02,  0.00000e+00,  9.25926e-02,  1.02084e+00, 
             -5.84940e-03,  1.34045e-02,  0.00000e+00,  9.25926e-02, -9.22388e-01, 
             -5.31299e-03,  1.31736e-02,  0.00000e+00,  9.25926e-02, -1.18465e+00, 
              6.35736e-03,  1.31594e-02,  0.00000e+00,  9.25926e-02,  7.81894e-01, 
             -9.28404e-03,  4.84083e-02,  0.00000e+00,  1.85185e-01, -1.01259e+00, 
             -9.71752e-03,  4.86100e-02,  0.00000e+00,  1.85185e-01, -7.65105e-01, 
              1.22947e-02,  4.79799e-02,  0.00000e+00,  1.85185e-01,  7.16642e-01, 
              1.20260e-02,  4.80492e-02,  0.00000e+00,  1.85185e-01,  5.14341e-01, 
             -7.54312e-03,  2.84113e-02,  0.00000e+00,  1.38889e-01, -1.12073e+00, 
             -8.02775e-03,  2.86317e-02,  0.00000e+00,  1.38889e-01, -8.45101e-01, 
              9.61981e-03,  2.81195e-02,  0.00000e+00,  1.38889e-01,  8.79688e-01, 
              9.28955e-03,  2.82089e-02,  0.00000e+00,  1.38889e-01,  6.47019e-01, 
              1.90129e-02,  1.29307e-01,  0.00000e+00,  3.24074e-01,  1.17015e-01, 
              1.91117e-02,  1.29297e-01,  0.00000e+00,  3.24074e-01,  2.09957e-01, 
             -1.18920e-02,  1.30287e-01,  0.00000e+00,  3.24074e-01, -5.29709e-01, 
             -1.16285e-02,  1.30143e-01,  0.00000e+00,  3.24074e-01, -6.68047e-01, 
              1.68862e-02,  9.92811e-02,  0.00000e+00,  2.77778e-01,  2.48614e-01, 
              1.70403e-02,  9.92510e-02,  0.00000e+00,  2.77778e-01,  3.77202e-01, 
              1.47678e-02,  7.19179e-02,  0.00000e+00,  2.31481e-01,  5.47124e-01, 
              1.45570e-02,  7.19678e-02,  0.00000e+00,  2.31481e-01,  3.80724e-01, 
             -1.05465e-02,  7.24856e-02,  0.00000e+00,  2.31481e-01, -8.99422e-01, 
             -1.09245e-02,  7.26680e-02,  0.00000e+00,  2.31481e-01, -6.87456e-01, 
             -1.16490e-02,  1.00121e-01,  0.00000e+00,  2.77778e-01, -6.08846e-01, 
             -1.13276e-02,  9.99584e-02,  0.00000e+00,  2.77778e-01, -7.83057e-01, 
              2.09336e-02,  1.61361e-01,  0.00000e+00,  3.70370e-01, -1.03022e-02, 
             -1.14486e-02,  1.62354e-01,  0.00000e+00,  3.70370e-01, -5.59385e-01, 
             -1.16542e-02,  1.62481e-01,  0.00000e+00,  3.70370e-01, -4.48380e-01, 
              2.09761e-02,  1.61370e-01,  0.00000e+00,  3.70370e-01,  5.68311e-02, 
              2.26167e-02,  1.94769e-01,  0.00000e+00,  4.16667e-01, -6.21169e-02, 
              2.26393e-02,  1.94746e-01,  0.00000e+00,  4.16667e-01, -1.21691e-01, 
             -1.07864e-02,  1.95898e-01,  0.00000e+00,  4.16667e-01, -4.68392e-01, 
             -1.09293e-02,  1.96025e-01,  0.00000e+00,  4.16667e-01, -3.70199e-01, 
             -9.65403e-03,  2.30120e-01,  0.00000e+00,  4.62963e-01, -4.00814e-01, 
             -9.61229e-03,  2.30046e-01,  0.00000e+00,  4.62963e-01, -4.69215e-01, 
              2.41788e-02,  2.28798e-01,  0.00000e+00,  4.62963e-01, -2.37398e-01, 
              2.40773e-02,  2.28849e-01,  0.00000e+00,  4.62963e-01, -1.67758e-01}
        };
    Plato::OrdinalType tTimeStep = 4;
    for(Plato::OrdinalType tOrdinal=0; tOrdinal< tSolution.extent(1); tOrdinal++)
    {
        const Plato::Scalar tValue = std::abs(tHostSolution(tTimeStep, tOrdinal)) < 1.0e-14 ? 0.0 : tHostSolution(tTimeStep, tOrdinal);
        TEST_FLOATING_EQUALITY(tValue, tGoldSolution[0][tOrdinal], tTolerance);
    }

    // Plato::OrdinalType tIdx = 0;
    // Plato::OrdinalType tDispDofX = 0;
    // Plato::OrdinalType tDispDofY = 1;
    // Plato::OrdinalType tDispDofZ = 2;
    // Plato::OrdinalType tTemperatureDof = 3;
    // Plato::OrdinalType tPressureDof = 4;
    // Plato::OrdinalType tNumDofsPerNode = 5;
    // for (Plato::OrdinalType tIndexK = 0; tIndexK < tNumVertices; tIndexK++)
    // {
    //   tIdx = tIndexK * tNumDofsPerNode + tDispDofX;       printf("%12.5e, ",  tHostSolution(tTimeStep, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tDispDofY;       printf("%12.5e, ",  tHostSolution(tTimeStep, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tDispDofZ;       printf("%12.5e, ",  tHostSolution(tTimeStep, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tTemperatureDof; printf("%12.5e, ",  tHostSolution(tTimeStep, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tPressureDof;    printf("%12.5e, \n", tHostSolution(tTimeStep, tIdx));
    // }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.saveStates("CantileverBeamTractionForce3D_Plastic");
    }
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_PlasticWork_2D)
{
    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 2;
    const Plato::OrdinalType tNumElemX = 10;
    const Plato::OrdinalType tNumElemY = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,tNumElemX,tNumElemY);
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='100.0'/>                 \n"
      "    <Parameter  name='Temperature Scaling' type='double' value='100.0'/>                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='75.0e3'/>                 \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='180.'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='2.32e-5'/> \n"
      "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>             \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='2.0e3'/>      \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e-2'/>     \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='344.0'/>             \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-8'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                          \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-8'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='6'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='6'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='25'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "   <ParameterList  name='Mechanical Natural Boundary Conditions'>                        \n"
      "   <ParameterList  name='Traction Vector Boundary Condition'>                            \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{0.0, -16.0}'/>             \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{1.0e3}'/>                  \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                              \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.0'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls).State;

    // 5. Test results
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::string tCriterionName("Plastic Work");
    auto tCriterionValue = tPlasticityProblem.criterionValue(tControls, tCriterionName);
    TEST_FLOATING_EQUALITY(tCriterionValue, -1.97983, tTolerance);

    auto tCriterionGrad = tPlasticityProblem.criterionGradient(tControls, tCriterionName);
    std::vector<Plato::Scalar> tGold = {  7.79752e+00, 5.60363e+00, 6.28221e+00, 1.29293e+01, 7.76341e+00, 
                                          3.04540e+00, 4.78934e+00, 5.16835e-01, 9.36347e-01,-2.90131e-02,
                                          -1.93664e-01, 1.55728e-03,-5.07596e-03,-1.34851e-02, 1.09473e-02,
                                          -5.70991e-04, 2.41109e-03,-4.15956e-04,-6.16690e-05, 2.12617e-04, 
                                          1.71787e-04, 1.09343e-04, 7.57489e-05,-1.70775e-06,-6.68193e-06,
                                          -4.98039e-06, 2.13624e-05, 6.22086e-07,-1.08766e-05,-5.07546e-06,
                                          -1.24006e-08, 1.63240e-06, 6.29494e-07};
    auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        //printf("%12.5e\n", tHostGrad(tIndex));
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.saveStates("Thermoplasticity_PlasticWork_2D");
    }
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_PlasticWork_3D)
{
    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tNumElemX = 10;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = PlatoUtestHelpers::build_3d_box_mesh(10.0,1.0,1.0,tNumElemX,tNumElemY,tNumElemZ);
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Iterations' type='int' value='5000'/>                             \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='1.0'/>                   \n"
      "    <Parameter  name='Temperature Scaling' type='double' value='1.0'/>                   \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e4'/>                  \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='180.'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='2.32e-5'/> \n"
      "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>             \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e2'/>      \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e2'/>      \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e2'/>             \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-8'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                          \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-8'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='25'/>                  \n"
      "    <Parameter name='Stopping Tolerance' type='double' value='1e-8'/>                    \n"
      "    <Parameter name='Current Residual Norm Stopping Tolerance' type='double' value='1e-8'/> \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "   <ParameterList  name='Mechanical Natural Boundary Conditions'>                        \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{1.0e3}'/>                  \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
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
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='3'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.0'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Disp Boundary Condition'>                             \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.2*t'/>                      \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls).State;

    // 5. Test results
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::string tCriterionName("Plastic Work");
    auto tCriterionValue = tPlasticityProblem.criterionValue(tControls, tCriterionName);
    TEST_FLOATING_EQUALITY(tCriterionValue, -9.18505, tTolerance);

    auto tCriterionGrad = tPlasticityProblem.criterionGradient(tControls, tCriterionName);
    std::vector<Plato::Scalar> tGold = { -8.23158e-01,-2.74211e-01,-2.74205e-01,-2.74211e-01,-5.46915e-01,
                                         -1.09598e+00,-5.46915e-01,-1.09091e+00,-1.07737e+00,-5.40880e-01,
                                         -1.08590e+00,-5.40880e-01,-1.05793e+00,-5.26599e-01,-1.04844e+00,
                                         -5.26599e-01,-1.07226e+00,-5.33831e-01,-1.06304e+00,-5.33831e-01,
                                         -5.04852e-01,-1.00493e+00,-5.04852e-01,-1.01433e+00,-5.12007e-01,
                                         -1.01919e+00,-1.03386e+00,-5.19301e-01,-1.04332e+00,-5.19301e-01,
                                         -5.12007e-01,-1.02878e+00,-4.98050e-01,-1.00060e+00,-4.98050e-01,
                                         -9.91065e-01,-9.80656e-01,-4.92349e-01,-9.88215e-01,-4.92349e-01,
                                         -2.44243e-01,-7.35025e-01,-2.44243e-01,-2.43315e-01};
    auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        //printf("%12.5e\n", tHostGrad(tIndex));
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.saveStates("Thermoplasticity_PlasticWork_3D");
    }
    //std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_PlasticWorkGradientZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    const Plato::OrdinalType tNumElemX = 10;
    const Plato::OrdinalType tNumElemY = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,tNumElemX,tNumElemY);
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='100.0'/>                 \n"
      "    <Parameter  name='Temperature Scaling' type='double' value='100.0'/>                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='75.0e3'/>                 \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='180.'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='2.32e-5'/> \n"
      "        <Parameter  name='Reference Temperature' type='double' value='100.0'/>           \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='344.0'/>             \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-8'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                          \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-8'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='25'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{1.0e3}'/>                  \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                              \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='200.0'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function' type='string' value='0.8*t'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 5. Test results
    std::string tCriterionName("Plastic Work");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    const Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_PlasticWorkGradientZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tNumElemX = 1;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = PlatoUtestHelpers::build_3d_box_mesh(1.0,1.0,1.0,tNumElemX,tNumElemY,tNumElemZ);
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
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Tolerance' type='double' value='1e-14'/>                          \n"
      "      <Parameter name='Iterations' type='int' value='1000'/>                             \n"
      "    </ParameterList>                                                                     \n"
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='10.0'/>                 \n"
      "    <Parameter  name='Temperature Scaling' type='double' value='10.0'/>                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='75.0e3'/>                 \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='180.'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='2.32e-5'/> \n"
      "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>             \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='344.0'/>             \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='2'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-2'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='1.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-1'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                          \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='2.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-2'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='2.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-2'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='25'/>                  \n"
      "    <Parameter name='Stopping Tolerance' type='double' value='1e-10'/>                   \n"
      "    <Parameter name='Current Residual Norm Stopping Tolerance' type='double' value='1e-10'/> \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "   <ParameterList  name='Mechanical Natural Boundary Conditions'>                        \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{1.0e3}'/>                  \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z1'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Applied Displacement Boundary Condition'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.08*t'/>                     \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='3'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.0'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 5. Test results
    std::string tCriterionName("Plastic Work");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    const Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    //std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_ElasticWork_2D)
{
    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 2;
    const Plato::OrdinalType tNumElemX = 10;
    const Plato::OrdinalType tNumElemY = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,tNumElemX,tNumElemY);
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='100.0'/>                 \n"
      "    <Parameter  name='Temperature Scaling' type='double' value='100.0'/>                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='75.0e3'/>                 \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='180.'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='2.32e-5'/> \n"
      "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>             \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='2.0e3'/>      \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e-2'/>     \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='344.0'/>             \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-8'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                          \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Elastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Elastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-8'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='6'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='6'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='25'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "   <ParameterList  name='Mechanical Natural Boundary Conditions'>                        \n"
      "   <ParameterList  name='Traction Vector Boundary Condition'>                            \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{0.0, -16.0}'/>             \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{1.0e3}'/>                  \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                              \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.0'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls).State;

    // 5. Test results
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::string tCriterionName("Elastic Work");
    auto tCriterionValue = tPlasticityProblem.criterionValue(tControls, tCriterionName);
    TEST_FLOATING_EQUALITY(tCriterionValue, -6.77828, tTolerance);

    auto tCriterionGrad = tPlasticityProblem.criterionGradient(tControls, tCriterionName);
    std::vector<Plato::Scalar> tGold = {   8.03858e+00, 5.07883e+00, 6.13084e+00, 1.31136e+01, 7.94362e+00, 
                                           2.98272e+00, 5.09420e+00, 5.91316e-01, 4.44496e-01, 5.25237e-01, 
                                           7.67438e-01, 1.45438e-01, 4.03751e-01, 7.01721e-01, 3.98144e-01, 
                                           1.47186e-01, 2.64768e-01, 5.10102e-01, 3.06928e-01, 3.03242e-01, 
                                           1.92345e-01, 3.97258e-02, 8.72496e-02, 7.13652e-02,-1.35716e-01,
                                           -5.74592e-02,-6.71317e-02,-1.52573e-01,-1.66383e-01,-3.56113e-01,
                                           -1.93767e-01,-2.42087e-01,-8.09962e-02};
    auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        //printf("%12.5e\n", tHostGrad(tIndex));
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.saveStates("Thermoplasticity_ElasticWork_2D");
    }
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_ElasticWork_3D)
{
    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tNumElemX = 10;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = PlatoUtestHelpers::build_3d_box_mesh(10.0,1.0,1.0,tNumElemX,tNumElemY,tNumElemZ);
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Iterations' type='int' value='5000'/>                             \n"
      "      <Parameter name='Tolerance' type='double' value='1.0e-10'/>                        \n"
      "    </ParameterList>                                                                     \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='20.0'/>                  \n"
      "    <Parameter  name='Temperature Scaling' type='double' value='10.0'/>                  \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e4'/>                  \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='180.'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='2.32e-5'/> \n"
      "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>             \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e2'/>      \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e2'/>      \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='100.0'/>             \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-8'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                          \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Elastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Elastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-8'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='2'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='2'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='25'/>                  \n"
      "    <Parameter name='Stopping Tolerance' type='double' value='1e-8'/>                    \n"
      "    <Parameter name='Current Residual Norm Stopping Tolerance' type='double' value='1e-8'/> \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "   <ParameterList  name='Mechanical Natural Boundary Conditions'>                        \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{1.0e3}'/>                  \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
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
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='3'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.0'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Disp Boundary Condition'>                             \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.02*t'/>                     \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solution
    auto tNumVertices = tMesh->nverts();
    Plato::ScalarVector tControls("Controls", tNumVertices);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls).State;

    // 5. Test results
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::string tCriterionName("Elastic Work");
    auto tCriterionValue = tPlasticityProblem.criterionValue(tControls, tCriterionName);
    TEST_FLOATING_EQUALITY(tCriterionValue, -0.316012, tTolerance);

    auto tCriterionGrad = tPlasticityProblem.criterionGradient(tControls, tCriterionName);
    std::vector<Plato::Scalar> tGold = {   8.10725e-03, 2.76799e-03, 2.77315e-03, 2.75210e-03, 5.24313e-03, 
                                           1.07304e-02, 5.24752e-03, 1.01715e-02, 8.34277e-03, 4.54118e-03, 
                                           9.66973e-03, 4.47393e-03, 3.08696e-03, 8.61307e-04, 7.64494e-05, 
                                           7.69755e-04, 7.12002e-03, 3.08811e-03, 4.99934e-03, 2.99937e-03,
                                           -1.06117e-02,-2.40471e-02,-1.05099e-02,-1.83710e-02,-6.03998e-03,
                                           -1.44660e-02,-6.42135e-03,-2.24518e-03,-2.50408e-03,-2.15290e-03,
                                           -5.94603e-03,-9.66026e-03,-1.59782e-02,-2.86427e-02,-1.58519e-02,
                                           -3.52717e-02,-4.74171e-02,-2.22469e-02,-4.07056e-02,-2.21171e-02,
                                           -1.25146e-02,-3.74843e-02,-1.25843e-02,-1.24195e-02};
    auto tHostGrad = Kokkos::create_mirror(tCriterionGrad);
    Kokkos::deep_copy(tHostGrad, tCriterionGrad);
    TEST_ASSERT( tHostGrad.size() == static_cast<Plato::OrdinalType>(tGold.size() ));
    for(Plato::OrdinalType tIndex = 0; tIndex < tHostGrad.size(); tIndex++)
    {
        //printf("%12.5e\n", tHostGrad(tIndex));
        TEST_FLOATING_EQUALITY(tHostGrad(tIndex), tGold[tIndex], tTolerance);
    }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.saveStates("Thermoplasticity_ElasticWork_3D");
    }
    //std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_ElasticWorkGradientZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    const Plato::OrdinalType tNumElemX = 10;
    const Plato::OrdinalType tNumElemY = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10.0,1.0,tNumElemX,tNumElemY);
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='100.0'/>                 \n"
      "    <Parameter  name='Temperature Scaling' type='double' value='100.0'/>                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='75.0e3'/>                 \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='180.'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='2.32e-5'/> \n"
      "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>             \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='344.0'/>             \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-8'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                          \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Elastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Elastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-8'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='25'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{1.0e3}'/>                  \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                              \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.0'/>                        \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Applied Displacement Boundary Condition'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function' type='string' value='0.8*t'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 5. Test results
    std::string tCriterionName("Elastic Work");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    const Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_ElasticWorkGradientZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tNumElemX = 1;
    const Plato::OrdinalType tNumElemY = 1;
    const Plato::OrdinalType tNumElemZ = 1;
    auto tMesh = PlatoUtestHelpers::build_3d_box_mesh(1.0,1.0,1.0,tNumElemX,tNumElemY,tNumElemZ);
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
      "    <ParameterList name='Linear Solver'>                                                 \n"
      "      <Parameter name='Tolerance' type='double' value='1e-14'/>                          \n"
      "      <Parameter name='Iterations' type='int' value='1000'/>                             \n"
      "    </ParameterList>                                                                     \n"
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='10.0'/>                  \n"
      "    <Parameter  name='Temperature Scaling' type='double' value='10.0'/>                  \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='75.0e3'/>                 \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='180.'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='2.32e-5'/> \n"
      "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>             \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>      \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>      \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='344.0'/>             \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='2'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-2'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='1.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-1'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                          \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='2.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-2'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Elastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Elastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='2.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-2'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='25'/>                  \n"
      "    <Parameter name='Stopping Tolerance' type='double' value='1e-10'/>                   \n"
      "    <Parameter name='Current Residual Norm Stopping Tolerance' type='double' value='1e-10'/> \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "   <ParameterList  name='Mechanical Natural Boundary Conditions'>                        \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{1.0e3}'/>                  \n"
      "     <Parameter  name='Sides'    type='string'        value='ss_X1'/>                    \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z1'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Applied Displacement Boundary Condition'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.08*t'/>                     \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='3'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='100.0*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 5. Test results
    std::string tCriterionName("Elastic Work");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    const Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_CriterionTest_2D_GradientZ)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <Parameter  name='Pressure Scaling'    type='double' value='100.0'/>                 \n"
      "    <Parameter  name='Temperature Scaling' type='double' value='100.0'/>                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>                    \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e6'/>                  \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='180.'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='0.0'/>     \n"
      "        <Parameter  name='Reference Temperature' type='double' value='0.0'/>             \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                                \n"
      "        <ParameterList name='J2 Plasticity'>                                                 \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0e3'/>       \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1.0e3'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>   \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-8'/>  \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/> \n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-4'/>  \n"
      "        </ParameterList>                                                                     \n"
      "      </ParameterList>                                                                       \n"
      "    </ParameterList>                                                                       \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-8'/>                     \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Criteria'>                                                        \n"
      "    <ParameterList name='Plastic Work'>                                                  \n"
      "      <Parameter name='Type'                 type='string' value='Scalar Function'/>     \n"
      "      <Parameter name='Scalar Function Type' type='string' value='Plastic Work'/>        \n"
      "      <Parameter name='Multiplier'           type='double' value='-1.0'/>                \n"
      "      <Parameter name='Exponent'             type='double' value='3.0'/>                 \n"
      "      <Parameter name='Minimum Value'        type='double' value='1.0e-8'/>              \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='4'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
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
      "     <ParameterList  name='Temperature Boundary Condition'>                              \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Applied Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X1'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='0.002*t'/>                    \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    std::string tCriterionName("Plastic Work");
    auto tApproxError = Plato::test_criterion_grad_wrt_control(tPlasticityProblem, *tMesh, tCriterionName);
    const Plato::Scalar tUpperBound = 1e-6;
    TEST_ASSERT(tApproxError < tUpperBound);
    //std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_RodElasticSolution2D)
{
    // 1. DEFINE PROBLEM
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <Parameter name='Debug'   type='bool'  value='false'/>                                 \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "      <Parameter  name='Pressure Scaling'    type='double' value='10.0'/>                \n"
      "      <Parameter  name='Temperature Scaling' type='double' value='20.0'/>                \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.333333333333333'/>      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e3'/>                  \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='300.0'/>\n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e-4'/>  \n"
      "        <Parameter  name='Reference Temperature' type='double' value='10.0'/>            \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0'/>        \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0'/>        \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1e10'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList  name='Natural Boundary Conditions'>                                    \n"
      "  </ParameterList>                                                                       \n"
      "   <ParameterList  name='Essential Boundary Conditions'>                                 \n"
      "     <ParameterList  name='X Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='0'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y1'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "      <Parameter  name='Index'    type='int'    value='2'/>                              \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='110.0'/>                      \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    constexpr Plato::Scalar tPressureScaling    = 10.0;
    constexpr Plato::Scalar tTemperatureScaling = 20.0;

    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    PlatoUtestHelpers::set_mesh_sets_2D(*tMesh, tMeshSets);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solve problem
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls).State;

    std::vector<std::vector<Plato::Scalar>> tGold =
        {
          {0.000e+00, 0.000e+00, 1.100e+02/tTemperatureScaling, -1.000e+01/tPressureScaling,
           0.000e+00, 0.000e+00, 1.100e+02/tTemperatureScaling, -1.000e+01/tPressureScaling,
           2.000e-02, 0.000e+00, 1.100e+02/tTemperatureScaling, -1.000e+01/tPressureScaling,
           2.000e-02, 0.000e+00, 1.100e+02/tTemperatureScaling, -1.000e+01/tPressureScaling}
        };
    auto tHostSolution = Kokkos::create_mirror(tSolution);
    Kokkos::deep_copy(tHostSolution, tSolution);

    const Plato::Scalar tTolerance = 1e-4;
    const Plato::OrdinalType tDim0 = tSolution.extent(0);
    const Plato::OrdinalType tDim1 = tSolution.extent(1);
    for (Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            //printf("X(%d,%d) = %f\n", tIndexI, tIndexJ, tHostInput(tIndexI, tIndexJ));
            const Plato::Scalar tValue = std::abs(tHostSolution(tIndexI, tIndexJ)) < 1.0e-14 ? 0.0 : tHostSolution(tIndexI, tIndexJ);
            TEST_FLOATING_EQUALITY(tValue, tGold[tIndexI][tIndexJ], tTolerance);
        }
    }

    // Plato::OrdinalType tIdx = 0;
    // for (Plato::OrdinalType tIndexK = 0; tIndexK < tNumVerts; tIndexK++)
    // {
    //   tIdx = tIndexK * tNumDofsPerNode + tDispDofX; printf("DofX: %12.3e, ", tHostSolution(0, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tDispDofY; printf("DofY: %12.3e, ", tHostSolution(0, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tTemperatureDof; printf("DofT: %12.3e, ", tHostSolution(0, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tPressureDof; printf("DofP: %12.3e \n", tHostSolution(0, tIdx));
    // }

    // 6. Output Data
    if(tOutputData)
    {
        tPlasticityProblem.saveStates("Thermoplasticity_RodElasticSolution2D");
    }
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_RodElasticSolution3D)
{
    // 1. DEFINE PROBLEM
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <Parameter name='Debug'   type='bool'  value='false'/>                                 \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "      <Parameter  name='Pressure Scaling'    type='double' value='10.0'/>                \n"
      "      <Parameter  name='Temperature Scaling' type='double' value='20.0'/>                \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.333333333333333'/>      \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e3'/>                  \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='300.0'/>\n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e-4'/>  \n"
      "        <Parameter  name='Reference Temperature' type='double' value='10.0'/>            \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0'/>        \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0'/>        \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1e10'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
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
      "     <ParameterList  name='Y Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='1'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Y1'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition'>                     \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z0'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Z Fixed Displacement Boundary Condition 2'>                   \n"
      "       <Parameter  name='Type'     type='string' value='Zero Value'/>                    \n"
      "       <Parameter  name='Index'    type='int'    value='2'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_Z1'/>                         \n"
      "     </ParameterList>                                                                    \n"
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='3'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='110.0'/>                      \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    constexpr Plato::Scalar tPressureScaling    = 10.0;
    constexpr Plato::Scalar tTemperatureScaling = 20.0;

    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solve problem
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls).State;

    std::vector<std::vector<Plato::Scalar>> tGold =
        {
          {0.000e+00, 0.000e+00, 0.000e+00, 1.100e+02/tTemperatureScaling, -1.000e+01/tPressureScaling,
           0.000e+00, 0.000e+00, 0.000e+00, 1.100e+02/tTemperatureScaling, -1.000e+01/tPressureScaling,
           0.000e+00, 0.000e+00, 0.000e+00, 1.100e+02/tTemperatureScaling, -1.000e+01/tPressureScaling,
           0.000e+00, 0.000e+00, 0.000e+00, 1.100e+02/tTemperatureScaling, -1.000e+01/tPressureScaling,
           2.000e-02, 0.000e+00, 0.000e+00, 1.100e+02/tTemperatureScaling, -1.000e+01/tPressureScaling,
           2.000e-02, 0.000e+00, 0.000e+00, 1.100e+02/tTemperatureScaling, -1.000e+01/tPressureScaling,
           2.000e-02, 0.000e+00, 0.000e+00, 1.100e+02/tTemperatureScaling, -1.000e+01/tPressureScaling,
           2.000e-02, 0.000e+00, 0.000e+00, 1.100e+02/tTemperatureScaling, -1.000e+01/tPressureScaling}
        };
    auto tHostSolution = Kokkos::create_mirror(tSolution);
    Kokkos::deep_copy(tHostSolution, tSolution);

    const Plato::Scalar tTolerance = 1e-4;
    const Plato::OrdinalType tDim0 = tSolution.extent(0);
    const Plato::OrdinalType tDim1 = tSolution.extent(1);
    for (Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            const Plato::Scalar tValue = std::abs(tHostSolution(tIndexI, tIndexJ)) < 1.0e-14 ? 0.0 : tHostSolution(tIndexI, tIndexJ);
            TEST_FLOATING_EQUALITY(tValue, tGold[tIndexI][tIndexJ], tTolerance);
        }
    }

    // Plato::OrdinalType tIdx = 0;
    // for (Plato::OrdinalType tIndexK = 0; tIndexK < tNumVerts; tIndexK++)
    // {
    //   tIdx = tIndexK * tNumDofsPerNode + tDispDofX; printf("DofX: %12.3e, ", tHostSolution(0, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tDispDofY; printf("DofY: %12.3e, ", tHostSolution(0, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tDispDofZ; printf("DofZ: %12.3e, ", tHostSolution(0, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tTemperatureDof; printf("DofT: %12.3e, ", tHostSolution(0, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tPressureDof; printf("DofP: %12.3e \n", tHostSolution(0, tIdx));
    // }

    // 6. Output Data
    if(tOutputData)
    {
        tPlasticityProblem.saveStates("Thermoplasticity_RodElasticSolution3D");
    }
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Thermoplasticity_ElasticSolution3D)
{
    // 1. DEFINE PROBLEM
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
      "  <Parameter name='Physics'          type='string'  value='Thermoplasticity'/>           \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <Parameter name='Debug'   type='bool'  value='false'/>                                 \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "      <Parameter  name='Pressure Scaling'    type='double' value='10.0'/>                \n"
      "      <Parameter  name='Temperature Scaling' type='double' value='20.0'/>                \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.24'/>                   \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e3'/>                  \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='300.0'/>\n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e-4'/>  \n"
      "        <Parameter  name='Reference Temperature' type='double' value='10.0'/>            \n"
      "      </ParameterList>                                                                   \n"
      "      <ParameterList name='Plasticity Model'>                                               \n"
      "        <ParameterList name='J2 Plasticity'>                                                \n"
      "          <Parameter  name='Hardening Modulus Isotropic' type='double' value='1.0'/>        \n"
      "          <Parameter  name='Hardening Modulus Kinematic' type='double' value='1.0'/>        \n"
      "          <Parameter  name='Initial Yield Stress' type='double' value='1e10'/>              \n"
      "          <Parameter  name='Elastic Properties Penalty Exponent' type='double' value='3'/>  \n"
      "          <Parameter  name='Elastic Properties Minimum Ersatz' type='double' value='1e-6'/> \n"
      "          <Parameter  name='Plastic Properties Penalty Exponent' type='double' value='2.5'/>\n"
      "          <Parameter  name='Plastic Properties Minimum Ersatz' type='double' value='1e-9'/> \n"
      "        </ParameterList>                                                                    \n"
      "      </ParameterList>                                                                      \n"
      "    </ParameterList>                                                                        \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Elliptic'>                                                        \n"
      "    <ParameterList name='Penalty Function'>                                              \n"
      "      <Parameter name='Type' type='string' value='SIMP'/>                                \n"
      "      <Parameter name='Exponent' type='double' value='3.0'/>                             \n"
      "      <Parameter name='Minimum Value' type='double' value='1.0e-9'/>                     \n"
      "      <Parameter name='Plottable' type='Array(string)' value='{principal stresses}'/>    \n"
      "    </ParameterList>                                                                     \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Time Stepping'>                                                   \n"
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='1'/>              \n"
      "  </ParameterList>                                                                       \n"
      "  <ParameterList name='Newton-Raphson'>                                                  \n"
      "    <Parameter name='Stop Measure' type='string' value='residual'/>                      \n"
      "    <Parameter name='Maximum Number Iterations' type='int' value='20'/>                  \n"
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
      "     <ParameterList  name='Fixed Temperature Boundary Condition'>                        \n"
      "       <Parameter  name='Type'     type='string' value='Time Dependent'/>                \n"
      "       <Parameter  name='Index'    type='int'    value='3'/>                             \n"
      "       <Parameter  name='Sides'    type='string' value='ns_X0'/>                         \n"
      "       <Parameter  name='Function'    type='string' value='110.0'/>                      \n"
      "     </ParameterList>                                                                    \n"
      "   </ParameterList>                                                                      \n"
      "</ParameterList>                                                                         \n"
    );

    constexpr Plato::Scalar tPressureScaling    = 10.0;
    constexpr Plato::Scalar tTemperatureScaling = 20.0;

    const bool tOutputData = false; // for debugging purpose, set true to enable Paraview output
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    Plato::DataMap    tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

    PlatoUtestHelpers::set_mesh_sets_3D(*tMesh, tMeshSets);

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);
    tPlasticityProblem.readEssentialBoundaryConditions(*tParamList);

    // 4. Solve problem
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tPlasticityProblem.solution(tControls).State;

    std::vector<std::vector<Plato::Scalar>> tGold =
        {
          {0.000e+00, 0.000e+00, 0.000e+00, 1.100e+02/tTemperatureScaling, 0.0/tPressureScaling,
           0.000e+00, 0.000e+00, 1.000e-02, 1.100e+02/tTemperatureScaling, 0.0/tPressureScaling,
           0.000e+00, 1.000e-02, 1.000e-02, 1.100e+02/tTemperatureScaling, 0.0/tPressureScaling,
           0.000e+00, 1.000e-02, 0.000e+00, 1.100e+02/tTemperatureScaling, 0.0/tPressureScaling,
           1.000e-02, 1.000e-02, 0.000e+00, 1.100e+02/tTemperatureScaling, 0.0/tPressureScaling,
           1.000e-02, 1.000e-02, 1.000e-02, 1.100e+02/tTemperatureScaling, 0.0/tPressureScaling,
           1.000e-02, 0.000e+00, 1.000e-02, 1.100e+02/tTemperatureScaling, 0.0/tPressureScaling,
           1.000e-02, 0.000e+00, 0.000e+00, 1.100e+02/tTemperatureScaling, 0.0/tPressureScaling}
        };
    auto tHostSolution = Kokkos::create_mirror(tSolution);
    Kokkos::deep_copy(tHostSolution, tSolution);

    const Plato::Scalar tTolerance = 1e-4;
    const Plato::OrdinalType tDim0 = tSolution.extent(0);
    const Plato::OrdinalType tDim1 = tSolution.extent(1);
    for (Plato::OrdinalType tIndexI = 0; tIndexI < tDim0; tIndexI++)
    {
        for (Plato::OrdinalType tIndexJ = 0; tIndexJ < tDim1; tIndexJ++)
        {
            //printf("X(%d,%d) = %f\n", tIndexI, tIndexJ, tHostInput(tIndexI, tIndexJ));
            const Plato::Scalar tValue = std::abs(tHostSolution(tIndexI, tIndexJ)) < 5.0e-14 ? 0.0 : tHostSolution(tIndexI, tIndexJ);
            TEST_FLOATING_EQUALITY(tValue, tGold[tIndexI][tIndexJ], tTolerance);
        }
    }

    // Plato::OrdinalType tIdx = 0;
    // for (Plato::OrdinalType tIndexK = 0; tIndexK < tNumVerts; tIndexK++)
    // {
    // 	tIdx = tIndexK * tNumDofsPerNode + tDispDofX; printf("DofX: %12.3e, ", tHostSolution(0, tIdx));
    // 	tIdx = tIndexK * tNumDofsPerNode + tDispDofY; printf("DofY: %12.3e, ", tHostSolution(0, tIdx));
    // 	tIdx = tIndexK * tNumDofsPerNode + tDispDofZ; printf("DofZ: %12.3e, ", tHostSolution(0, tIdx));
    // 	tIdx = tIndexK * tNumDofsPerNode + tTemperatureDof; printf("DofT: %12.3e, ", tHostSolution(0, tIdx));
    // 	tIdx = tIndexK * tNumDofsPerNode + tPressureDof; printf("DofP: %12.3e \n", tHostSolution(0, tIdx));
    // }

    // 6. Output Data
    if(tOutputData)
    {
        tPlasticityProblem.saveStates("Thermoplasticity_ElasticSolution3D");
    }
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


}
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
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
      "  <Parameter name='PDE Constraint'   type='string'  value='Elliptic'/>                   \n"
      "  <ParameterList name='Material Models'>                                                 \n"
      "    <ParameterList name='Unobtainium'>                                                   \n"
      "      <ParameterList name='Isotropic Linear Thermoelastic'>                              \n"
      "        <Parameter  name='Density' type='double' value='1000'/>                          \n"
      "        <Parameter  name='Poissons Ratio' type='double' value='0.24'/>                   \n"
      "        <Parameter  name='Youngs Modulus' type='double' value='1.0e7'/>                  \n"
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='50.0'/> \n"
      "        <Parameter  name='Thermal Expansion Coefficient' type='double' value='1.0e-20'/>  \n"
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
      "    <Parameter name='Initial Num. Pseudo Time Steps' type='int' value='2'/>              \n"
      "    <Parameter name='Maximum Num. Pseudo Time Steps' type='int' value='2'/>              \n"
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
      "     <Parameter  name='Sides'    type='string'        value='Load'/>                     \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{0.0e-9}'/>                 \n"
      "     <Parameter  name='Sides'    type='string'        value='Load'/>                     \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    // 1. Construct plasticity problem
    auto tFaceIDs = PlatoUtestHelpers::get_edge_ids_on_y1(*tMesh);
    tMeshSets[Omega_h::SIDE_SET]["Load"] = tFaceIDs;
    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);

    // 2. Get Dirichlet Boundary Conditions
    const Plato::OrdinalType tDispDofX = 0;
    const Plato::OrdinalType tDispDofY = 1;
    const Plato::OrdinalType tTemperatureDof = 2;
    constexpr auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    auto tDirichletIndicesBoundaryX0 = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(*tMesh, "x0", tNumDofsPerNode, tDispDofX);

    // 3. Set Dirichlet Boundary Conditions
    // 3.1 Symmetry degrees of freedom and 1 temperature
    Plato::Scalar tValueToSet = 0;
    auto tNumDirichletDofs = tDirichletIndicesBoundaryX0.size() + 2 + 1;
    Plato::ScalarVector tDirichletValues("Dirichlet Values", tNumDirichletDofs);
    Plato::LocalOrdinalVector tDirichletDofs("Dirichlet Dofs", tNumDirichletDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tDirichletValues(aIndex) = tValueToSet;
        tDirichletDofs(aIndex) = tDirichletIndicesBoundaryX0(aIndex);
    }, "set dirichlet values and indices");

    // 3.2. Pinned degrees of freedom
    tValueToSet = 0.0;
    const Plato::OrdinalType tPinnedNodeIndex = 32;
    auto tOffset = tDirichletIndicesBoundaryX0.size();

    auto tHostDirichletValues = Kokkos::create_mirror(tDirichletValues);
    Kokkos::deep_copy(tHostDirichletValues, tDirichletValues);
    tHostDirichletValues(tOffset + tDispDofX) = tValueToSet;
    tHostDirichletValues(tOffset + tDispDofY) = tValueToSet;
    tHostDirichletValues(tOffset + tTemperatureDof) = 110.0;
    Kokkos::deep_copy(tDirichletValues, tHostDirichletValues);

    auto tHostDirichletDofs = Kokkos::create_mirror(tDirichletDofs);
    Kokkos::deep_copy(tHostDirichletDofs, tDirichletDofs);
    tHostDirichletDofs(tOffset + tDispDofX) = tNumDofsPerNode * tPinnedNodeIndex + tDispDofX;
    tHostDirichletDofs(tOffset + tDispDofY) = tNumDofsPerNode * tPinnedNodeIndex + tDispDofY;
    tHostDirichletDofs(tOffset + tTemperatureDof) = tNumDofsPerNode * tPinnedNodeIndex + tTemperatureDof;
    Kokkos::deep_copy(tDirichletDofs, tHostDirichletDofs);

    // 3.3 set Dirichlet boundary conditions
    tPlasticityProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);

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
    Plato::OrdinalType tTimeStep = 1;
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
    const bool tOutputData = true; // for debugging purpose, set true to enable Paraview output
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
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
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
      "     <Parameter  name='Sides'    type='string'        value='Load'/>                     \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      "   <ParameterList  name='Thermal Natural Boundary Conditions'>                           \n"
      "   <ParameterList  name='Thermal Flux Boundary Condition'>                               \n"
      "     <Parameter  name='Type'     type='string'        value='Uniform'/>                  \n"
      "     <Parameter  name='Values'   type='Array(double)' value='{1.0e3}'/>                 \n"
      "     <Parameter  name='Sides'    type='string'        value='Load'/>                     \n"
      "   </ParameterList>                                                                      \n"
      "   </ParameterList>                                                                      \n"
      " </ParameterList>                                                                        \n"
      "</ParameterList>                                                                         \n"
    );

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    constexpr Plato::Scalar tPressureScaling    = 100.0;
    constexpr Plato::Scalar tTemperatureScaling = 100.0;

    // 1. Construct plasticity problem
    auto tFaceIDs = PlatoUtestHelpers::get_edge_ids_on_x1(*tMesh);
    tMeshSets[Omega_h::SIDE_SET]["Load"] = tFaceIDs;
    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;

    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);

    // 2. Get Dirichlet Boundary Conditions
    const Plato::OrdinalType tDispDofX = 0;
    const Plato::OrdinalType tDispDofY = 1;
    const Plato::OrdinalType tTemperatureDof = 2;
    const Plato::OrdinalType tPressureDof = 3;
    constexpr auto tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    auto tDirichletIndicesBoundaryX0_X = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(*tMesh, "x0", tNumDofsPerNode, tDispDofX);
    auto tDirichletIndicesBoundaryX0_Y = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(*tMesh, "x0", tNumDofsPerNode, tDispDofY);
    auto tDirichletIndicesBoundaryX0_T = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(*tMesh, "x0", tNumDofsPerNode, tTemperatureDof);

    // 3. Set Dirichlet Boundary Conditions
    // 3.1 Symmetry degrees of freedom and 1 temperature
    Plato::Scalar tValueToSet = 0.0;
    auto tNumDirichletDofs = tDirichletIndicesBoundaryX0_X.size() + tDirichletIndicesBoundaryX0_Y.size() + tDirichletIndicesBoundaryX0_T.size();
    Plato::ScalarVector tDirichletValues("Dirichlet Values", tNumDirichletDofs);
    Plato::LocalOrdinalVector tDirichletDofs("Dirichlet Dofs", tNumDirichletDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_X.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tDirichletValues(aIndex) = tValueToSet;
        tDirichletDofs(aIndex) = tDirichletIndicesBoundaryX0_X(aIndex);
    }, "set dirichlet values and indices");

    auto tOffset = tDirichletIndicesBoundaryX0_X.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Y.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tDirichletValues(aIndex + tOffset) = tValueToSet;
        tDirichletDofs(aIndex + tOffset) = tDirichletIndicesBoundaryX0_Y(aIndex);
    }, "set dirichlet values and indices");

    tOffset += tDirichletIndicesBoundaryX0_Y.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_T.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tDirichletValues(aIndex + tOffset) = 0.0/tTemperatureScaling;
        tDirichletDofs(aIndex + tOffset) = tDirichletIndicesBoundaryX0_T(aIndex);
    }, "set dirichlet values and indices");

    // 3.3 set Dirichlet boundary conditions
    tPlasticityProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);

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
         {   0.00000e+00,  0.00000e+00, 0.00000e+00, -4.60073e-01, 
             0.00000e+00,  0.00000e+00, 0.00000e+00,  2.92597e-01, 
            -5.99021e-03, -6.11782e-03, 5.55556e-02, -6.10071e-01, 
            -1.37979e-04, -6.16735e-03, 5.55556e-02, -4.73334e-02, 
             5.30779e-03, -5.97557e-03, 5.55556e-02,  5.01814e-01, 
             0.00000e+00,  0.00000e+00, 0.00000e+00,  7.56213e-01, 
             1.05837e-02, -2.24066e-02, 1.11111e-01,  4.79371e-01, 
            -1.01017e-02, -2.27897e-02, 1.11111e-01, -5.30318e-01, 
             2.39631e-04, -2.27974e-02, 1.11111e-01, -3.39245e-02, 
            -1.36176e-02, -4.79304e-02, 1.66667e-01, -5.19952e-01, 
             6.50520e-04, -4.78380e-02, 1.66667e-01, -9.47773e-02, 
             1.48799e-02, -4.73995e-02, 1.66667e-01,  3.46734e-01, 
            -1.64159e-02, -8.03862e-02, 2.22222e-01, -4.87420e-01, 
             1.24878e-03, -8.01807e-02, 2.22222e-01, -1.24111e-01, 
             1.88708e-02, -7.96757e-02, 2.22222e-01,  2.56499e-01, 
             2.57863e-02, -1.61998e-01, 3.33333e-01,  6.82856e-02, 
             2.25060e-02, -1.18226e-01, 2.77778e-01,  1.62893e-01, 
             2.01154e-03, -1.18789e-01, 2.77778e-01, -1.55874e-01, 
            -1.85247e-02, -1.19103e-01, 2.77778e-01, -4.57126e-01, 
             2.94142e-03, -1.62620e-01, 3.33333e-01, -1.88294e-01, 
            -1.99453e-02, -1.63042e-01, 3.33333e-01, -4.27379e-01, 
             2.87126e-02, -2.09949e-01, 3.88889e-01, -2.60069e-02, 
             4.03889e-03, -2.10631e-01, 3.88889e-01, -2.20413e-01, 
            -2.06769e-02, -2.11161e-01, 3.88889e-01, -3.97245e-01, 
             5.30302e-03, -2.61778e-01, 4.44444e-01, -2.51787e-01, 
            -2.07204e-02, -2.62419e-01, 4.44444e-01, -3.65485e-01, 
             3.12849e-02, -2.61036e-01, 4.44444e-01, -1.19732e-01, 
             3.53487e-02, -3.68458e-01, 5.55556e-01, -2.77992e-01, 
             3.35094e-02, -3.14218e-01, 5.00000e-01, -2.19214e-01, 
             6.74259e-03, -3.15013e-01, 5.00000e-01, -2.92422e-01, 
            -2.00609e-02, -3.15766e-01, 5.00000e-01, -3.47337e-01, 
             8.33202e-03, -3.69287e-01, 5.55556e-01, -3.19877e-01, 
            -1.87189e-02, -3.70104e-01, 5.55556e-01, -3.46729e-01}
        };
    Plato::OrdinalType tTimeStep = 4;
    for(Plato::OrdinalType tOrdinal=0; tOrdinal< tSolution.extent(1); tOrdinal++)
    {
        TEST_FLOATING_EQUALITY(tHostSolution(tTimeStep, tOrdinal), tGoldSolution[0][tOrdinal], tTolerance);
    }

    // Plato::OrdinalType tIdx = 0;
    // for (Plato::OrdinalType tIndexK = 0; tIndexK < tNumVertices; tIndexK++)
    // {
    //   tIdx = tIndexK * tNumDofsPerNode + tDispDofX;       printf("DofX: %12.5e, ",  tHostSolution(4, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tDispDofY;       printf("DofY: %12.5e, ",  tHostSolution(4, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tTemperatureDof; printf("DofT: %12.5e, ",  tHostSolution(4, tIdx));
    //   tIdx = tIndexK * tNumDofsPerNode + tPressureDof;    printf("DofP: %12.5e \n", tHostSolution(4, tIdx));
    // }

    // 6. Output Data
    if (tOutputData)
    {
        tPlasticityProblem.saveStates("CantileverBeamTractionForce2D_Plastic");
    }
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
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
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
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

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);

    // 2. Get Dirichlet Boundary Conditions
    Plato::OrdinalType tDispDofX = 0;
    Plato::OrdinalType tDispDofY = 1;
    Plato::OrdinalType tTemperatureDof = 2;
    Plato::OrdinalType tPressureDof = 3;
    constexpr Plato::OrdinalType tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    auto tDirichletIndicesBoundaryX0_Xdof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(*tMesh, "x0", tNumDofsPerNode, tDispDofX);
    auto tDirichletIndicesBoundaryY0_Ydof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(*tMesh, "y0", tNumDofsPerNode, tDispDofY);
    auto tDirichletIndicesBoundaryY1_Ydof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_2D(*tMesh, "y1", tNumDofsPerNode, tDispDofY);

    // 3. Set Dirichlet Boundary Conditions
    Plato::Scalar tValueToSet = 0;
    auto tNumDirichletDofs = tDirichletIndicesBoundaryX0_Xdof.size() + tDirichletIndicesBoundaryY0_Ydof.size() +
            tDirichletIndicesBoundaryY1_Ydof.size() + 1;
    Plato::ScalarVector tDirichletValues("Dirichlet Values", tNumDirichletDofs);
    Plato::LocalOrdinalVector tDirichletDofs("Dirichlet Dofs", tNumDirichletDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Xdof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tDirichletValues(aIndex) = tValueToSet;
        tDirichletDofs(aIndex) = tDirichletIndicesBoundaryX0_Xdof(aIndex);
    }, "set dirichlet values and indices");

    auto tOffset = tDirichletIndicesBoundaryX0_Xdof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryY0_Ydof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryY0_Ydof(aIndex);
    }, "set dirichlet values and indices");

    tOffset += tDirichletIndicesBoundaryY0_Ydof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryY1_Ydof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryY1_Ydof(aIndex);
    }, "set dirichlet values and indices");

    tOffset += tDirichletIndicesBoundaryY1_Ydof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, 1), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = 110.0 / tTemperatureScaling;
        tDirichletDofs(tIndex) = tTemperatureDof;
    }, "set dirichlet values and indices");


    tPlasticityProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);

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

    // for (Plato::OrdinalType tIndexK = 0; tIndexK < tNumVerts; tIndexK++)
    // {
    //   Plato::OrdinalType tIdx = tIndexK * tNumDofsPerNode + tPressureDof;
    //   tHostSolution(0, tIdx) *= tPressureScaling;

    //   tIdx = tIndexK * tNumDofsPerNode + tTemperatureDof;
    //   tHostSolution(0, tIdx) *= tTemperatureScaling;
    // }

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
        tPlasticityProblem.saveStates("ThermoplasticityThermoelasticSolution2D");
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
      "  <Parameter name='Physics'          type='string'  value='Plasticity'/>                 \n"
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
      "        <Parameter  name='Thermal Conductivity Coefficient' type='double' value='300.0'/>  \n"
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

    MPI_Comm myComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
    Plato::Comm::Machine tMachine(myComm);

    using PhysicsT = Plato::InfinitesimalStrainThermoPlasticity<tSpaceDim>;
    Plato::PlasticityProblem<PhysicsT> tPlasticityProblem(*tMesh, tMeshSets, *tParamList, tMachine);

    // 2. Get Dirichlet Boundary Conditions
    Plato::OrdinalType tDispDofX = 0;
    Plato::OrdinalType tDispDofY = 1;
    Plato::OrdinalType tDispDofZ = 2;
    Plato::OrdinalType tTemperatureDof = 3;
    Plato::OrdinalType tPressureDof = 4;
    constexpr Plato::OrdinalType tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    auto tDirichletIndicesBoundaryX0_Xdof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x0", tNumDofsPerNode, tDispDofX);
    auto tDirichletIndicesBoundaryY0_Ydof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "y0", tNumDofsPerNode, tDispDofY);
    auto tDirichletIndicesBoundaryZ0_Zdof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "z0", tNumDofsPerNode, tDispDofZ);

    // 3. Set Dirichlet Boundary Conditions
    Plato::Scalar tValueToSet = 0;
    auto tNumDirichletDofs = tDirichletIndicesBoundaryX0_Xdof.size() + tDirichletIndicesBoundaryY0_Ydof.size() +
            tDirichletIndicesBoundaryZ0_Zdof.size() + 1;
    Plato::ScalarVector tDirichletValues("Dirichlet Values", tNumDirichletDofs);
    Plato::LocalOrdinalVector tDirichletDofs("Dirichlet Dofs", tNumDirichletDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Xdof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tDirichletValues(aIndex) = tValueToSet;
        tDirichletDofs(aIndex) = tDirichletIndicesBoundaryX0_Xdof(aIndex);
    }, "set dirichlet values and indices");

    auto tOffset = tDirichletIndicesBoundaryX0_Xdof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryY0_Ydof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryY0_Ydof(aIndex);
    }, "set dirichlet values and indices");

    tOffset += tDirichletIndicesBoundaryY0_Ydof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryZ0_Zdof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryZ0_Zdof(aIndex);
    }, "set dirichlet values and indices");

    tOffset += tDirichletIndicesBoundaryZ0_Zdof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, 1), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = 110.0/tTemperatureScaling;
        tDirichletDofs(tIndex) = tTemperatureDof;
    }, "set dirichlet values and indices");


    tPlasticityProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);

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
            const Plato::Scalar tValue = std::abs(tHostSolution(tIndexI, tIndexJ)) < 1.0e-14 ? 0.0 : tHostSolution(tIndexI, tIndexJ);
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
        tPlasticityProblem.saveStates("ThermoplasticityThermoelasticSolution3D");
    }
    std::system("rm -f plato_analyze_newton_raphson_diagnostics.txt");
}


}
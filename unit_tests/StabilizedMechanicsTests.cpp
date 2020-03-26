/*
 * StabilizedMechanicsTests.cpp
 *
 *  Created on: Mar 26, 2020
 */

#include <Teuchos_XMLParameterListCoreHelpers.hpp>

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoUtilities.hpp"
#include "PlatoTestHelpers.hpp"
#include "EllipticVMSProblem.hpp"


namespace StabilizedMechanicsTests
{


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StabilizedMechanics_Solution3D)
{
    // 1. DEFINE PROBLEM
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);
    Plato::DataMap    tDataMap;
    Omega_h::MeshSets tMeshSets;

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                               \n"
        "<Parameter name='Physics'         type='string'  value='Stabilized Mechanical'/> \n"
        "<Parameter name='PDE Constraint'  type='string'  value='Elliptic'/>              \n"
        "<ParameterList name='Elliptic'>                                                  \n"
          "<ParameterList name='Penalty Function'>                                        \n"
            "<Parameter name='Type' type='string' value='SIMP'/>                          \n"
            "<Parameter name='Exponent' type='double' value='3.0'/>                       \n"
            "<Parameter name='Minimum Value' type='double' value='1.0e-9'/>               \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Time Stepping'>                                             \n"
          "<Parameter name='Number Time Steps' type='int' value='2'/>                     \n"
          "<Parameter name='Time Step' type='double' value='1.0'/>                        \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Newton Iteration'>                                          \n"
          "<Parameter name='Number Iterations' type='int' value='3'/>                      \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Material Model'>                                            \n"
          "<ParameterList name='Isotropic Linear Elastic'>                                 \n"
            "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>               \n"
            "<Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
    "</ParameterList>                                                                     \n"
    );

    using PhysicsT = Plato::StabilizedMechanics<tSpaceDim>;
    Plato::EllipticVMSProblem<PhysicsT> tEllipticVMSProblem(*tMesh, tMeshSets, *tParamList);

    // 2. Get Dirichlet Boundary Conditions
    Plato::OrdinalType tDispDofX = 0;
    Plato::OrdinalType tDispDofY = 1;
    Plato::OrdinalType tDispDofZ = 2;
    constexpr Plato::OrdinalType tNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    auto tDirichletIndicesBoundaryX0_Xdof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x0", tNumDofsPerNode, tDispDofX);
    auto tDirichletIndicesBoundaryX0_Ydof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x0", tNumDofsPerNode, tDispDofY);
    auto tDirichletIndicesBoundaryX0_Zdof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x0", tNumDofsPerNode, tDispDofZ);
    auto tDirichletIndicesBoundaryX1_Ydof = PlatoUtestHelpers::get_dirichlet_indices_on_boundary_3D(*tMesh, "x1", tNumDofsPerNode, tDispDofY);

    // 3. Set Dirichlet Boundary Conditions
    Plato::Scalar tValueToSet = 0;
    auto tNumDirichletDofs = tDirichletIndicesBoundaryX0_Xdof.size() + tDirichletIndicesBoundaryX0_Ydof.size() +
            tDirichletIndicesBoundaryX0_Zdof.size() + tDirichletIndicesBoundaryX1_Ydof.size();
    Plato::ScalarVector tDirichletValues("Dirichlet Values", tNumDirichletDofs);
    Plato::LocalOrdinalVector tDirichletDofs("Dirichlet Dofs", tNumDirichletDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Xdof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        tDirichletValues(aIndex) = tValueToSet;
        tDirichletDofs(aIndex) = tDirichletIndicesBoundaryX0_Xdof(aIndex);
    }, "set dirichlet values and indices");

    auto tOffset = tDirichletIndicesBoundaryX0_Xdof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Ydof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX0_Ydof(aIndex);
    }, "set dirichlet values and indices");

    tOffset += tDirichletIndicesBoundaryX0_Ydof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX0_Zdof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX0_Zdof(aIndex);
    }, "set dirichlet values and indices");

    tValueToSet = -1e-3;
    tOffset += tDirichletIndicesBoundaryX0_Zdof.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tDirichletIndicesBoundaryX1_Ydof.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        auto tIndex = tOffset + aIndex;
        tDirichletValues(tIndex) = tValueToSet;
        tDirichletDofs(tIndex) = tDirichletIndicesBoundaryX1_Ydof(aIndex);
    }, "set dirichlet values and indices");
    tEllipticVMSProblem.setEssentialBoundaryConditions(tDirichletDofs, tDirichletValues);

    // 4. Solve problem
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::fill(1.0, tControls);
    auto tSolution = tEllipticVMSProblem.solution(tControls);
    auto tSubViewT1 = Kokkos::subview(tSolution, 1, Kokkos::ALL());
    Plato::print(tSubViewT1, "solution t=1");

    Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer("SolutionMesh", tMesh.getRawPtr(), tSpaceDim);
    tMesh->add_tag(Omega_h::VERT, "State", 1, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tSubViewT1)));
    auto tTags = Omega_h::vtk::get_all_vtk_tags(tMesh.getRawPtr(), tSpaceDim);
    tWriter.write(static_cast<Omega_h::Real>(1), tTags);
}


}
// namespace StabilizedMechanicsTests

#include "PlatoTestHelpers.hpp"

#include "Teuchos_UnitTestHarness.hpp"
#include <Teuchos_XMLParameterListHelpers.hpp>

#include "Mechanics.hpp"
#include "EssentialBCs.hpp"
#include "elliptic/VectorFunction.hpp"
#include "ApplyConstraints.hpp"
#include "SimplexMechanics.hpp"
#include "LinearElasticMaterial.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "MultipointConstraints.hpp"

#ifdef HAVE_AMGX
#include <alg/AmgXSparseLinearProblem.hpp>
#endif


#include <fenv.h>
#include <memory>
#include <typeinfo>

template <typename DataType>
bool is_same(
      const Plato::ScalarVectorT<DataType> & aView,
      const std::vector<DataType>          & aVec)
 {
    auto tView_host = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView_host, aView);

    if( aView.extent(0) != aVec.size() ) return false;

    for (unsigned int i = 0; i < aVec.size(); ++i)
    {
        if(tView_host(i) != aVec[i])
        {
            return false;
        }
    }
    return true;
 }

template <typename DataType>
void print_view(const Plato::ScalarVectorT<DataType> & aView)
 {
    auto tView_host = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView_host, aView);
    std::cout << '\n';
    for (unsigned int i = 0; i < aView.extent(0); ++i)
    {
        std::cout << tView_host(i) << '\n';
    }
 }

/******************************************************************************/
/*!
  \brief test MPC nodeset or ID input
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( MultipointConstraintTests, MPCNodeSetTest )
{
  
  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;
   
  // create parameter input list
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <ParameterList  name='Multipoint Constraints'>                        \n"
    "    <ParameterList  name='Node Tie Constraint 1'>                       \n"
    "      <Parameter  name='Type'     type='string'    value='Tie'/>        \n"
    "      <Parameter  name='Child'    type='int'       value='42'/>         \n"
    "      <Parameter  name='Parent'   type='int'       value='73'/>         \n"
    "      <Parameter  name='Value'    type='double'    value='3.0'/>        \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Node Tie Constraint 2'>                       \n"
    "      <Parameter  name='Type'     type='string'    value='Tie'/>        \n"
    "      <Parameter  name='Child'    type='int'       value='24'/>         \n"
    "      <Parameter  name='Parent'   type='int'       value='3'/>          \n"
    "      <Parameter  name='Value'    type='double'    value='7.0'/>        \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Node Tie Constraint 3'>                       \n"
    "      <Parameter  name='Type'     type='string'    value='Tie'/>        \n"
    "      <Parameter  name='Child'    type='string'    value='MPC Child'/>  \n"
    "      <Parameter  name='Parent'   type='string'    value='MPC Parent'/> \n"
    "      <Parameter  name='Value'    type='double'    value='4.8'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );
  
  // assign edges MPC nodesets
  //
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  Omega_h::Read<Omega_h::I8> tMarksLoad = Omega_h::mark_class_closure(mesh.get(), Omega_h::EDGE, Omega_h::EDGE, 5 /* class id */);
  tMeshSets[Omega_h::SIDE_SET]["Load"] = Omega_h::collect_marked(tMarksLoad);

  Omega_h::Read<Omega_h::I8> tMarksFix = Omega_h::mark_class_closure(mesh.get(), Omega_h::VERT, Omega_h::EDGE, 3 /* class id */);
  tMeshSets[Omega_h::NODE_SET]["Fix"] = Omega_h::collect_marked(tMarksFix);
  
  // assign edges for MPCs
  //
  Omega_h::Read<Omega_h::I8> tMarksParent = Omega_h::mark_class_closure(mesh.get(), Omega_h::VERT, Omega_h::EDGE, 1 /* class id */); // bottom edge is parent
  tMeshSets[Omega_h::NODE_SET]["MPC Parent"] = Omega_h::collect_marked(tMarksParent);

  Omega_h::Read<Omega_h::I8> tMarksChild = Omega_h::mark_class_closure(mesh.get(), Omega_h::VERT, Omega_h::EDGE, 7 /* class id */); // top edge is child
  tMeshSets[Omega_h::NODE_SET]["MPC Child"] = Omega_h::collect_marked(tMarksChild);

  // test getting MPC type
  auto tMpcParams = params->sublist("Multipoint Constraints", false);

  for(Teuchos::ParameterList::ConstIterator tIndex = tMpcParams.begin(); tIndex != tMpcParams.end(); ++tIndex)
  {
      const Teuchos::ParameterEntry & tEntry = tMpcParams.entry(tIndex);
      const std::string & tMyName = tMpcParams.name(tIndex);

      TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error, " Parameter in Multipoint Constraints block not valid. Expect lists only.");

      Teuchos::ParameterList& tSublist = tMpcParams.sublist(tMyName);

      std::cout << '\n'; 
      if(tSublist.isType<std::string>("Child"))
      {
          std::cout << "Child is a string" << std::endl;

          std::string tChildNodeSet = tSublist.get<std::string>("Child");
          std::cout << tChildNodeSet << std::endl;

          /* auto tChildNodeSetsIter = tNodeSets.find(this->mChildNodeSet); */
          /* auto tChildNodeLids = (tChildNodeSetsIter->second); */
          /* auto tNumberChildNodes = tChildNodeLids.size(); */
      }

      if(tSublist.isType<Plato::OrdinalType>("Child"))
      {
          std::cout << "Child is an int" << std::endl;

          Plato::OrdinalType tChildNode = tSublist.get<Plato::OrdinalType>("Child");
          std::cout << tChildNode << std::endl;
      }

      if(tSublist.isType<std::string>("Parent"))
      {
          std::cout << "Parent is a string" << std::endl;

          std::string tParentNodeSet = tSublist.get<std::string>("Parent");
          std::cout << tParentNodeSet << std::endl;
      }

      if(tSublist.isType<Plato::OrdinalType>("Parent"))
      {
          std::cout << "Parent is an int" << std::endl;

          Plato::OrdinalType tParentNode = tSublist.get<Plato::OrdinalType>("Parent");
          std::cout << tParentNode << std::endl;
      }

  }

}

/******************************************************************************/
/*!
  \brief 2D Elastic problem with Tie multipoint constraints

  Construct a linear system with tie multipoint constraints.
  Test passes if nodal displacements are offset by specified amount in MPC
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( MultipointConstraintTests, Elastic2DTieMPC )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  std::cout << '\n' << "Number of Nodes: " << tNumNodes;
  std::cout << '\n' << "Number of DOFs: " << tNumNodes*tNumDofsPerNode;

  // Test output of node IDs

  std::cout << '\n' << "Edge 3 (x0) Nodes:" << std::endl;
  auto tNodesX0= PlatoUtestHelpers::get_2D_boundary_nodes_x0(*mesh);
  PlatoUtestHelpers::print_ordinals(tNodesX0);
  PlatoUtestHelpers::print_2d_coords(*mesh,tNodesX0);

  std::cout << '\n' <<"Edge 5 (x1) Nodes:" << std::endl;
  auto tNodesX1= PlatoUtestHelpers::get_2D_boundary_nodes_x1(*mesh);
  PlatoUtestHelpers::print_ordinals(tNodesX1);
  PlatoUtestHelpers::print_2d_coords(*mesh,tNodesX1);

  std::cout << '\n' << "Edge 1 (y0) Nodes:" << std::endl;
  auto tNodesY0= PlatoUtestHelpers::get_2D_boundary_nodes_y0(*mesh);
  PlatoUtestHelpers::print_ordinals(tNodesY0);
  PlatoUtestHelpers::print_2d_coords(*mesh,tNodesY0);

  std::cout << '\n' << "Edge 7 (y1) Nodes:" << std::endl;
  auto tNodesY1= PlatoUtestHelpers::get_2D_boundary_nodes_y1(*mesh);
  PlatoUtestHelpers::print_ordinals(tNodesY1);
  PlatoUtestHelpers::print_2d_coords(*mesh,tNodesY1);

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // specify parameter input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>             \n"
    "  <ParameterList name='Elliptic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList name='Material Model'>                                 \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                   \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>          \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>   \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0}'/>  \n"
    "      <Parameter name='Sides'  type='string'        value='Load'/>      \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                 \n"
    "    <ParameterList  name='X Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Multipoint Constraints'>                        \n"
    "    <ParameterList  name='Node Tie Constraint 1'>                       \n"
    "      <Parameter  name='Type'     type='string'    value='Tie'/>        \n"
    "      <Parameter  name='Child'    type='string'    value='MPC Child'/>  \n"
    "      <Parameter  name='Parent'   type='string'    value='MPC Parent'/> \n"
    "      <Parameter  name='Value'    type='double'    value='0.0'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  // assign edges for load and boundary conditions
  //
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  Omega_h::Read<Omega_h::I8> tMarksLoad = Omega_h::mark_class_closure(mesh.get(), Omega_h::EDGE, Omega_h::EDGE, 5 /* class id */);
  tMeshSets[Omega_h::SIDE_SET]["Load"] = Omega_h::collect_marked(tMarksLoad);

  Omega_h::Read<Omega_h::I8> tMarksFix = Omega_h::mark_class_closure(mesh.get(), Omega_h::VERT, Omega_h::EDGE, 3 /* class id */);
  tMeshSets[Omega_h::NODE_SET]["Fix"] = Omega_h::collect_marked(tMarksFix);
  
  // assign edges for MPCs
  //
  Omega_h::Read<Omega_h::I8> tMarksParent = Omega_h::mark_class_closure(mesh.get(), Omega_h::VERT, Omega_h::EDGE, 1 /* class id */); // bottom edge is parent
  tMeshSets[Omega_h::NODE_SET]["MPC Parent"] = Omega_h::collect_marked(tMarksParent);

  Omega_h::Read<Omega_h::I8> tMarksChild = Omega_h::mark_class_closure(mesh.get(), Omega_h::VERT, Omega_h::EDGE, 7 /* class id */); // top edge is child
  tMeshSets[Omega_h::NODE_SET]["MPC Child"] = Omega_h::collect_marked(tMarksChild);
  
  // parse essential BCs
  //
  Plato::LocalOrdinalVector bcDofs;
  Plato::ScalarVector bcValues;
  Plato::EssentialBCs<SimplexPhysics>
      tEssentialBoundaryConditions(params->sublist("Essential Boundary Conditions",false));
  tEssentialBoundaryConditions.get(tMeshSets, bcDofs, bcValues);

  // create vector function
  //
  Plato::Elliptic::VectorFunction<SimplexPhysics>
    vectorFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  int tNumCells = vectorFunction.numCells();
  std::cout << '\n' << "Number of Elements: " << tNumCells << std::endl;

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);
  
  // parse multipoint constraints
  //
  std::shared_ptr<Plato::MultipointConstraints> tMPCs = std::make_shared<Plato::MultipointConstraints>(*mesh, tMeshSets, tNumDofsPerNode, params->sublist("Multipoint Constraints", false));
  tMPCs->setupTransform();
  
  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver' type='string' value='AztecOO'/>        \n"
    "  <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "  <Parameter name='Iterations' type='int' value='50'/>            \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(mesh->nverts(), tMachine, tNumDofsPerNode, tMPCs);

  // apply essential BCs
  //
  Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(jacobian, residual, bcDofs, bcValues);

  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test difference between constrained nodes
  //
  Plato::OrdinalType checkChildNode = 4;
  Plato::OrdinalType checkParentNode = 7;
  Plato::Scalar      checkValue = 0.0;

  Plato::OrdinalType checkChildDof0 = checkChildNode*tNumDofsPerNode;
  Plato::OrdinalType checkChildDof1 = checkChildNode*tNumDofsPerNode + 1;

  Plato::OrdinalType checkParentDof0 = checkParentNode*tNumDofsPerNode;
  Plato::OrdinalType checkParentDof1 = checkParentNode*tNumDofsPerNode + 1;

  Plato::Scalar checkDifferenceDof0 = stateView_host(checkChildDof0) - stateView_host(checkParentDof0);
  Plato::Scalar checkDifferenceDof1 = stateView_host(checkChildDof1) - stateView_host(checkParentDof1);

  TEST_FLOATING_EQUALITY(checkDifferenceDof0, checkValue, 1.0e-12);
  TEST_FLOATING_EQUALITY(checkDifferenceDof1, checkValue, 1.0e-12);

}

/******************************************************************************/
/*!
  \brief 2D Elastic problem with PBC multipoint constraints

  Construct a linear system with PBC multipoint constraints.
  Test passes if nodal displacements are offset by specified amount in MPC
*/
/******************************************************************************/
TEUCHOS_UNIT_TEST( MultipointConstraintTests, Elastic3DPbcMPC )
{
  feclearexcept(FE_ALL_EXCEPT);
  feenableexcept(FE_INVALID | FE_OVERFLOW);

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=3;
  auto mesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = mesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);
  
  // Test output of node IDs
  auto tNodesAll = PlatoUtestHelpers::get_boundary_nodes(*mesh);
  PlatoUtestHelpers::print_ordinals(tNodesAll);
  PlatoUtestHelpers::print_3d_coords(*mesh,tNodesAll);

  auto tMeshObject = *mesh;

  std::cout << '\n' << "Face 12 (x0) Nodes:" << std::endl;
  auto tX0Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 12 /* class id */);
  auto tX0Ordinals = Omega_h::collect_marked(tX0Marks);
  PlatoUtestHelpers::print_ordinals(tX0Ordinals);
  PlatoUtestHelpers::print_3d_coords(*mesh,tX0Ordinals);

  std::cout << '\n' << "Face 14 (x1) Nodes:" << std::endl;
  auto tX1Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 14 /* class id */);
  auto tX1Ordinals = Omega_h::collect_marked(tX1Marks);
  PlatoUtestHelpers::print_ordinals(tX1Ordinals);
  PlatoUtestHelpers::print_3d_coords(*mesh,tX1Ordinals);

  std::cout << '\n' << "Face 10 (y0) Nodes:" << std::endl;
  auto tY0Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 10 /* class id */);
  auto tY0Ordinals = Omega_h::collect_marked(tY0Marks);
  PlatoUtestHelpers::print_ordinals(tY0Ordinals);
  PlatoUtestHelpers::print_3d_coords(*mesh,tY0Ordinals);

  std::cout << '\n' << "Face 16 (y1) Nodes:" << std::endl;
  auto tY1Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 16 /* class id */);
  auto tY1Ordinals = Omega_h::collect_marked(tY1Marks);
  PlatoUtestHelpers::print_ordinals(tY1Ordinals);
  PlatoUtestHelpers::print_3d_coords(*mesh,tY1Ordinals);

  std::cout << '\n' << "Face 4 (z0) Nodes:" << std::endl;
  auto tZ0Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 4 /* class id */);
  auto tZ0Ordinals = Omega_h::collect_marked(tZ0Marks);
  PlatoUtestHelpers::print_ordinals(tZ0Ordinals);
  PlatoUtestHelpers::print_3d_coords(*mesh,tZ0Ordinals);

  std::cout << '\n' << "Face 22 (z1) Nodes:" << std::endl;
  auto tZ1Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 22 /* class id */);
  auto tZ1Ordinals = Omega_h::collect_marked(tZ1Marks);
  PlatoUtestHelpers::print_ordinals(tZ1Ordinals);
  PlatoUtestHelpers::print_3d_coords(*mesh,tZ1Ordinals);

  // specify parameter input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>             \n"
    "  <ParameterList name='Elliptic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList name='Material Model'>                                 \n"
    "    <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "      <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "      <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Natural Boundary Conditions'>                   \n"
    "    <ParameterList  name='Traction Vector Boundary Condition'>          \n"
    "      <Parameter name='Type'   type='string'        value='Uniform'/>   \n"
    "      <Parameter name='Values' type='Array(double)' value='{1e3, 0, 0}'/>  \n"
    "      <Parameter name='Sides'  type='string'        value='Load'/>      \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                 \n"
    "    <ParameterList  name='X Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='0'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Y Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='1'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "    <ParameterList  name='Z Fixed Displacement Boundary Condition'>     \n"
    "      <Parameter  name='Type'     type='string' value='Zero Value'/>    \n"
    "      <Parameter  name='Index'    type='int'    value='2'/>             \n"
    "      <Parameter  name='Sides'    type='string' value='Fix'/>           \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList  name='Multipoint Constraints'>                        \n"
    "    <ParameterList  name='PBC Constraint 1'>                            \n"
    "      <Parameter  name='Type'     type='string'    value='PBC'/>        \n"
    "      <Parameter  name='Child'    type='string'    value='MPC Child'/>  \n"
    "      <Parameter  name='Vector'  type='Array(double)' value='{0, 1, 0}'/>  \n"
    "      <Parameter  name='Value'    type='double'    value='0.0'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  // assign edges for load and boundary conditions
  //
  Plato::DataMap tDataMap;
  Omega_h::MeshSets tMeshSets;
  Omega_h::Read<Omega_h::I8> tMarksLoad = Omega_h::mark_class_closure(mesh.get(), Omega_h::EDGE, Omega_h::FACE, 14 /* class id */);
  tMeshSets[Omega_h::SIDE_SET]["Load"] = Omega_h::collect_marked(tMarksLoad);

  Omega_h::Read<Omega_h::I8> tMarksFix = Omega_h::mark_class_closure(mesh.get(), Omega_h::VERT, Omega_h::FACE, 12 /* class id */);
  tMeshSets[Omega_h::NODE_SET]["Fix"] = Omega_h::collect_marked(tMarksFix);
  
  // assign faces for MPCs
  //
  Omega_h::Read<Omega_h::I8> tMarksChild = Omega_h::mark_class_closure(mesh.get(), Omega_h::VERT, Omega_h::FACE, 10 /* class id */); // y0 face is child
  tMeshSets[Omega_h::NODE_SET]["MPC Child"] = Omega_h::collect_marked(tMarksChild);
  
  // parse essential BCs
  //
  Plato::LocalOrdinalVector bcDofs;
  Plato::ScalarVector bcValues;
  Plato::EssentialBCs<SimplexPhysics>
      tEssentialBoundaryConditions(params->sublist("Essential Boundary Conditions",false));
  tEssentialBoundaryConditions.get(tMeshSets, bcDofs, bcValues);

  // create vector function
  //
  Plato::Elliptic::VectorFunction<SimplexPhysics>
    vectorFunction(*mesh, tMeshSets, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  int tNumCells = vectorFunction.numCells();
  std::cout << '\n' << "Number of Elements: " << tNumCells << std::endl;

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);
  
  // parse multipoint constraints
  //
  std::shared_ptr<Plato::MultipointConstraints> tMPCs = std::make_shared<Plato::MultipointConstraints>(*mesh, tMeshSets, tNumDofsPerNode, params->sublist("Multipoint Constraints", false));
  tMPCs->setupTransform();
  
  // create solver
  //
  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  Teuchos::RCP<Teuchos::ParameterList> tSolverParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Linear Solver'>                              \n"
    "  <Parameter name='Solver' type='string' value='AztecOO'/>        \n"
    "  <Parameter name='Display Iterations' type='int' value='0'/>     \n"
    "  <Parameter name='Iterations' type='int' value='50'/>            \n"
    "  <Parameter name='Tolerance' type='double' value='1e-14'/>       \n"
    "</ParameterList>                                                  \n"
  );
  Plato::SolverFactory tSolverFactory(*tSolverParams);

  auto tSolver = tSolverFactory.create(mesh->nverts(), tMachine, tNumDofsPerNode, tMPCs);

  // apply essential BCs
  //
  Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(jacobian, residual, bcDofs, bcValues);

  // solve linear system
  //
  tSolver->solve(*jacobian, state, residual);

  // create mirror view of displacement solution
  //
  Plato::ScalarVector statesView("State",tNumDofs);
  Kokkos::deep_copy(statesView, state);

  auto stateView_host = Kokkos::create_mirror_view(statesView);
  Kokkos::deep_copy(stateView_host, statesView);

  // test difference between constrained nodes
  //
  Plato::OrdinalType checkChildNode = 22;
  Plato::OrdinalType checkParentNode = 15;
  Plato::Scalar      checkValue = 0.0;

  Plato::OrdinalType checkChildDof0 = checkChildNode*tNumDofsPerNode;
  Plato::OrdinalType checkChildDof1 = checkChildNode*tNumDofsPerNode + 1;
  Plato::OrdinalType checkChildDof2 = checkChildNode*tNumDofsPerNode + 2;

  Plato::OrdinalType checkParentDof0 = checkParentNode*tNumDofsPerNode;
  Plato::OrdinalType checkParentDof1 = checkParentNode*tNumDofsPerNode + 1;
  Plato::OrdinalType checkParentDof2 = checkParentNode*tNumDofsPerNode + 2;

  Plato::Scalar checkDifferenceDof0 = stateView_host(checkChildDof0) - stateView_host(checkParentDof0);
  Plato::Scalar checkDifferenceDof1 = stateView_host(checkChildDof1) - stateView_host(checkParentDof1);
  Plato::Scalar checkDifferenceDof2 = stateView_host(checkChildDof2) - stateView_host(checkParentDof2);

  TEST_FLOATING_EQUALITY(checkDifferenceDof0, checkValue, 1.0e-8);
  TEST_FLOATING_EQUALITY(checkDifferenceDof1, checkValue, 1.0e-8);
  TEST_FLOATING_EQUALITY(checkDifferenceDof2, checkValue, 1.0e-8);

}

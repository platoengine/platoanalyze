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

#include "SpatialModel.hpp"

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
  \brief test Omega-H parsing of node sets
*/
/******************************************************************************/
/* TEUCHOS_UNIT_TEST( MultipointConstraintTests, OmegaHNodeSetParsingTest ) */
/* { */
/*   // specify mesh */
/*   // */
/*   Teuchos::RCP<Teuchos::ParameterList> tMeshParams = */
/*     Teuchos::getParametersFromXmlString( */
/*     "<ParameterList name='Mesh File'>                                      \n" */
/*     "  <Parameter  name='Input Mesh' type='string' value='test_mesh_rve.exo'/> \n" */
/*     "  <Parameter  name='Node Set 1' type='string' value='ChildFaceX'/> \n" */
/*     "  <Parameter  name='Node Set 2' type='string' value='ChildEdge15'/> \n" */
/*     "</ParameterList>                                                      \n" */
/*   ); */

/*   // read in mesh */
/*   // */
/*   auto tLibOmegaH = PlatoUtestHelpers::getLibraryOmegaH(); */
/*   auto tInputMesh = tMeshParams->get<std::string>("Input Mesh"); */

/*   Omega_h::Mesh mesh = Omega_h::read_mesh_file(tInputMesh, tLibOmegaH->world()); */
  
/*   // get mesh data */
/*   // */
/*   constexpr int spaceDim=3; */
/*   using SimplexPhysics = ::Plato::Mechanics<spaceDim>; */

/*   int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode; */
/*   int tNumNodes = mesh.nverts(); */
/*   int tNumDofs = tNumNodes*tNumDofsPerNode; */

/*   std::cout << '\n' << "Number of Nodes: " << tNumNodes << std::endl; */

/*   // get mesh sets */
/*   // */
/*   Omega_h::Assoc tAssoc; */
/*   tAssoc[Omega_h::ELEM_SET] = mesh.class_sets; */
/*   tAssoc[Omega_h::NODE_SET] = mesh.class_sets; */
/*   tAssoc[Omega_h::SIDE_SET] = mesh.class_sets; */
/*   Omega_h::MeshSets tMeshSets = Omega_h::invert(&mesh, tAssoc); */

/*   // parse mesh node set 1 */
/*   // */
/*   auto& tNodeSets = tMeshSets[Omega_h::NODE_SET]; */
/*   std::string tChildNodeSet1 = tMeshParams->get<std::string>("Node Set 1"); */
/*   auto tChildNodeSets1Iter = tNodeSets.find(tChildNodeSet1); */
/*   if(tChildNodeSets1Iter == tNodeSets.end()) */
/*   { */
/*       std::ostringstream tMsg; */
/*       tMsg << "Could not find Node Set with name = '" << tChildNodeSet1.c_str() */
/*               << "'. Node Set is not defined in input geometry/mesh file.\n"; */
/*       THROWERR(tMsg.str()) */
/*   } */
/*   auto tChildNodeLids1 = (tChildNodeSets1Iter->second); */
/*   auto tNumberChildNodes1 = tChildNodeLids1.size(); */

/*   std::cout << '\n' << "Nodes in Set 1: " << tNumberChildNodes1 << std::endl; */
  
/*   // parse mesh node set 2 */
/*   // */
/*   std::string tChildNodeSet2 = tMeshParams->get<std::string>("Node Set 2"); */
/*   auto tChildNodeSets2Iter = tNodeSets.find(tChildNodeSet2); */
/*   if(tChildNodeSets2Iter == tNodeSets.end()) */
/*   { */
/*       std::ostringstream tMsg; */
/*       tMsg << "Could not find Node Set with name = '" << tChildNodeSet2.c_str() */
/*               << "'. Node Set is not defined in input geometry/mesh file.\n"; */
/*       THROWERR(tMsg.str()) */
/*   } */
/*   auto tChildNodeLids2 = (tChildNodeSets2Iter->second); */
/*   auto tNumberChildNodes2 = tChildNodeLids2.size(); */

/*   std::cout << '\n' << "Nodes in Set 2: " << tNumberChildNodes2 << std::endl; */

/* } */

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

  // specify parameter input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>             \n"
    "  <ParameterList name='Elliptic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
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
    "      <Parameter  name='Value'    type='double'    value='4.2'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  // create test mesh
  //
  constexpr int meshWidth=2;
  constexpr int spaceDim=2;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
  int tNumDofs = tNumNodes*tNumDofsPerNode;

  // create mesh based density
  //
  Plato::ScalarVector control("density", tNumDofs);
  Kokkos::deep_copy(control, 1.0);

  // create mesh based state
  //
  Plato::ScalarVector state("state", tNumDofs);
  Kokkos::deep_copy(state, 0.0);

  // assign edges for load and boundary conditions
  //
  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Omega_h::Read<Omega_h::I8> tMarksLoad = Omega_h::mark_class_closure(tMesh.get(), Omega_h::EDGE, Omega_h::EDGE, 5 /* class id */);
  tMeshSets[Omega_h::SIDE_SET]["Load"] = Omega_h::collect_marked(tMarksLoad);

  Omega_h::Read<Omega_h::I8> tMarksFix = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::EDGE, 3 /* class id */);
  tMeshSets[Omega_h::NODE_SET]["Fix"] = Omega_h::collect_marked(tMarksFix);
  
  // assign edges for MPCs
  //
  Omega_h::Read<Omega_h::I8> tMarksParent = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::EDGE, 1 /* class id */); // bottom edge is parent
  tMeshSets[Omega_h::NODE_SET]["MPC Parent"] = Omega_h::collect_marked(tMarksParent);

  Omega_h::Read<Omega_h::I8> tMarksChild = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::EDGE, 7 /* class id */); // top edge is child
  tMeshSets[Omega_h::NODE_SET]["MPC Child"] = Omega_h::collect_marked(tMarksChild);
  
  // parse essential BCs
  //
  Plato::LocalOrdinalVector mBcDofs;
  Plato::ScalarVector mBcValues;
  Plato::EssentialBCs<SimplexPhysics>
      tEssentialBoundaryConditions(params->sublist("Essential Boundary Conditions",false), tMeshSets);
  tEssentialBoundaryConditions.get(mBcDofs, mBcValues);

  // create vector function
  //
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *params);
  Plato::Elliptic::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);
  
  // parse multipoint constraints
  //
  std::shared_ptr<Plato::MultipointConstraints> tMPCs = std::make_shared<Plato::MultipointConstraints>(tSpatialModel, tNumDofsPerNode, params->sublist("Multipoint Constraints", false));
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

  auto tSolver = tSolverFactory.create(tMesh->nverts(), tMachine, tNumDofsPerNode, tMPCs);

  // apply essential BCs
  //
  Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(jacobian, residual, mBcDofs, mBcValues);

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
  Plato::Scalar      checkValue = 4.2;

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
  auto tMesh = PlatoUtestHelpers::getBoxMesh(spaceDim, meshWidth);

  using SimplexPhysics = ::Plato::Mechanics<spaceDim>;

  int tNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;
  int tNumNodes = tMesh->nverts();
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
  /* auto tNodesAll = PlatoUtestHelpers::get_boundary_nodes(*tMesh); */
  /* PlatoUtestHelpers::print_ordinals(tNodesAll); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tNodesAll); */

  /* auto tMeshObject = *tMesh; */

  /* std::cout << '\n' << "Face 12 (x0) Nodes:" << std::endl; */
  /* auto tX0Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 12 /1* class id *1/); */
  /* auto tX0Ordinals = Omega_h::collect_marked(tX0Marks); */
  /* PlatoUtestHelpers::print_ordinals(tX0Ordinals); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tX0Ordinals); */

  /* std::cout << '\n' << "Face 14 (x1) Nodes:" << std::endl; */
  /* auto tX1Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 14 /1* class id *1/); */
  /* auto tX1Ordinals = Omega_h::collect_marked(tX1Marks); */
  /* PlatoUtestHelpers::print_ordinals(tX1Ordinals); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tX1Ordinals); */

  /* std::cout << '\n' << "Face 10 (y0) Nodes:" << std::endl; */
  /* auto tY0Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 10 /1* class id *1/); */
  /* auto tY0Ordinals = Omega_h::collect_marked(tY0Marks); */
  /* PlatoUtestHelpers::print_ordinals(tY0Ordinals); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tY0Ordinals); */

  /* std::cout << '\n' << "Face 16 (y1) Nodes:" << std::endl; */
  /* auto tY1Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 16 /1* class id *1/); */
  /* auto tY1Ordinals = Omega_h::collect_marked(tY1Marks); */
  /* PlatoUtestHelpers::print_ordinals(tY1Ordinals); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tY1Ordinals); */

  /* std::cout << '\n' << "Face 4 (z0) Nodes:" << std::endl; */
  /* auto tZ0Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 4 /1* class id *1/); */
  /* auto tZ0Ordinals = Omega_h::collect_marked(tZ0Marks); */
  /* PlatoUtestHelpers::print_ordinals(tZ0Ordinals); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tZ0Ordinals); */

  /* std::cout << '\n' << "Face 22 (z1) Nodes:" << std::endl; */
  /* auto tZ1Marks = Omega_h::mark_class_closure(&tMeshObject, Omega_h::VERT, Omega_h::FACE, 22 /1* class id *1/); */
  /* auto tZ1Ordinals = Omega_h::collect_marked(tZ1Marks); */
  /* PlatoUtestHelpers::print_ordinals(tZ1Ordinals); */
  /* PlatoUtestHelpers::print_3d_coords(*tMesh,tZ1Ordinals); */

  // specify parameter input
  //
  Teuchos::RCP<Teuchos::ParameterList> params =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                    \n"
    "  <ParameterList name='Spatial Model'>                                    \n"
    "    <ParameterList name='Domains'>                                        \n"
    "      <ParameterList name='Design Volume'>                                \n"
    "        <Parameter name='Element Block' type='string' value='body'/>      \n"
    "        <Parameter name='Material Model' type='string' value='Unobtainium'/> \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
    "  </ParameterList>                                                        \n"
    "  <Parameter name='PDE Constraint' type='string' value='Elliptic'/>     \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='true'/>             \n"
    "  <ParameterList name='Elliptic'>                                       \n"
    "    <ParameterList name='Penalty Function'>                             \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>               \n"
    "      <Parameter name='Exponent' type='double' value='1.0'/>            \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "  <ParameterList name='Material Models'>                                  \n"
    "    <ParameterList name='Unobtainium'>                                    \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                     \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.3'/>     \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>  \n"
    "      </ParameterList>                                                    \n"
    "    </ParameterList>                                                      \n"
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
    "      <Parameter  name='Parent'   type='string'    value='Design Volume'/>  \n"
    "      <Parameter  name='Vector'  type='Array(double)' value='{0, 1, 0}'/>  \n"
    "      <Parameter  name='Value'    type='double'    value='0.0'/>        \n"
    "    </ParameterList>                                                    \n"
    "  </ParameterList>                                                      \n"
    "</ParameterList>                                                        \n"
  );

  // assign edges for load and boundary conditions
  //
  Plato::DataMap tDataMap;
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(spaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Omega_h::Read<Omega_h::I8> tMarksLoad = Omega_h::mark_class_closure(tMesh.get(), Omega_h::EDGE, Omega_h::FACE, 14 /* class id */);
  tMeshSets[Omega_h::SIDE_SET]["Load"] = Omega_h::collect_marked(tMarksLoad);

  Omega_h::Read<Omega_h::I8> tMarksFix = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::FACE, 12 /* class id */);
  tMeshSets[Omega_h::NODE_SET]["Fix"] = Omega_h::collect_marked(tMarksFix);
  
  // assign faces for MPCs
  //
  Omega_h::Read<Omega_h::I8> tMarksChild = Omega_h::mark_class_closure(tMesh.get(), Omega_h::VERT, Omega_h::FACE, 10 /* class id */); // y0 face is child
  tMeshSets[Omega_h::NODE_SET]["MPC Child"] = Omega_h::collect_marked(tMarksChild);
  
  // parse essential BCs
  //
  Plato::LocalOrdinalVector mBcDofs;
  Plato::ScalarVector mBcValues;
  Plato::EssentialBCs<SimplexPhysics>
      tEssentialBoundaryConditions(params->sublist("Essential Boundary Conditions",false), tMeshSets);
  tEssentialBoundaryConditions.get(mBcDofs, mBcValues);

  // create vector function
  //
  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *params);
  Plato::Elliptic::VectorFunction<SimplexPhysics>
    vectorFunction(tSpatialModel, tDataMap, *params, params->get<std::string>("PDE Constraint"));

  // compute residual
  //
  auto residual = vectorFunction.value(state, control);
  Plato::blas1::scale(-1.0, residual);

  // compute jacobian
  //
  auto jacobian = vectorFunction.gradient_u(state, control);
  
  // parse multipoint constraints
  //
  std::shared_ptr<Plato::MultipointConstraints> tMPCs = std::make_shared<Plato::MultipointConstraints>(tSpatialModel, tNumDofsPerNode, params->sublist("Multipoint Constraints", false));
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

  auto tSolver = tSolverFactory.create(tMesh->nverts(), tMachine, tNumDofsPerNode, tMPCs);

  // apply essential BCs
  //
  Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(jacobian, residual, mBcDofs, mBcValues);

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


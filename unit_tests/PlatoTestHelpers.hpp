/*
 * PlatoTestHelpers.hpp
 *
 *  Created on: Mar 31, 2018
 */

#ifndef PLATOTESTHELPERS_HPP_
#define PLATOTESTHELPERS_HPP_

#include <fstream>
#include <Teuchos_RCP.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>


#include "Omega_h_build.hpp"
#include "Omega_h_map.hpp"
#include "Omega_h_assoc.hpp"
#include "Omega_h_mark.hpp"
#include "Omega_h_matrix.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_library.hpp"
#include "AnalyzeMacros.hpp"

#include "PlatoStaticsTypes.hpp"

namespace PlatoUtestHelpers
{

/***************************************************************************//**
 * \fn get_box_mesh_sets
 * \brief Return collection of element, side, and node sets for a box mesh model.
 *
 * 1D Mesh Sets Description:
 *  Node Set Names   : 'x-', 'x+', 'body'
 *  Side Set Names   : 'x-', 'x+'
 *  Element Set Names: 'body'
 * 2D Mesh Sets Description:
 *  Node Set Names   : 'x-', 'y-', 'x+', 'y+', 'body'
 *  Side Set Names   : 'x-', 'y-', 'x+', 'y+'
 *  Element Set Names: 'body'
 * 3D Mesh Sets Description:
 *  Node Set Names   : 'x-', 'y-', 'z-', 'x+', 'y+', 'z+', 'body'
 *  Side Set Names   : 'x-', 'y-', 'z-', 'x+', 'y+', 'z+'
 *  Element Set Names: 'body'
 *
 * \param [in] aMesh mesh database
 * \return collection of element, side, and node sets
*******************************************************************************/
Omega_h::MeshSets
inline get_box_mesh_sets
(Omega_h::Mesh & aMesh)
{
    auto tNumSpaceDim = aMesh.dim();
    auto tAssoc = Omega_h::get_box_assoc(tNumSpaceDim);
    const auto tMeshSets = Omega_h::invert(&aMesh, tAssoc);
    return tMeshSets;
}

/***************************************************************************//**
 * \fn get_box_side_sets
 * \brief Return side set for a box mesh model.
 *
 * 1D Mesh Sets Description:
 *  Node Set Names   : 'x-', 'x+', 'body'
 * 2D Mesh Sets Description:
 *  Node Set Names   : 'x-', 'y-', 'x+', 'y+', 'body'
 * 3D Mesh Sets Description:
 *  Node Set Names   : 'x-', 'y-', 'z-', 'x+', 'y+', 'z+', 'body'
 *
 * \param [in] aMesh mesh database
 * \return side set
*******************************************************************************/
Omega_h::MeshDimSets
inline get_box_node_sets
(Omega_h::Mesh & aMesh)
{
    auto tNumSpaceDim = aMesh.dim();
    auto tAssoc = Omega_h::get_box_assoc(tNumSpaceDim);
    const auto tMeshSets = Omega_h::invert(&aMesh, tAssoc);
    const auto tNodeSets = tMeshSets[Omega_h::NODE_SET];
    return tNodeSets;
}

/***************************************************************************//**
 * \fn get_box_side_sets
 * \brief Return side set for a box mesh model.
 *
 * 1D Mesh Sets Description:
 *  Side Set Names   : 'x-', 'x+'
 * 2D Mesh Sets Description:
 *  Side Set Names   : 'x-', 'y-', 'x+', 'y+'
 * 3D Mesh Sets Description:
 *  Side Set Names   : 'x-', 'y-', 'z-', 'x+', 'y+', 'z+'
 *
 * \param [in] aMesh mesh database
 * \return side set
*******************************************************************************/
Omega_h::MeshDimSets
inline get_box_side_sets
(Omega_h::Mesh & aMesh)
{
    auto tNumSpaceDim = aMesh.dim();
    auto tAssoc = Omega_h::get_box_assoc(tNumSpaceDim);
    const auto tMeshSets = Omega_h::invert(&aMesh, tAssoc);
    const auto tSideSets = tMeshSets[Omega_h::SIDE_SET];
    return tSideSets;
}

/***************************************************************************//**
 * \fn get_box_elem_sets
 * \brief Return element set for a box mesh model.
 *
 * 1D Mesh Sets Description:
 *  Element Set Names: 'body'
 * 2D Mesh Sets Description:
 *  Element Set Names: 'body'
 * 3D Mesh Sets Description:
 *  Element Set Names: 'body'
 *
 * \param [in] aMesh mesh database
 * \return element set
*******************************************************************************/
Omega_h::MeshDimSets
inline get_box_elem_sets
(Omega_h::Mesh & aMesh)
{
    auto tNumSpaceDim = aMesh.dim();
    auto tAssoc = Omega_h::get_box_assoc(tNumSpaceDim);
    const auto tMeshSets = Omega_h::invert(&aMesh, tAssoc);
    const auto tElemSets = tMeshSets[Omega_h::ELEM_SET];
    return tElemSets;
}

/******************************************************************************//**
 * \brief get view from device
 *
 * \param[in] aView data on device
 * \returns Mirror on host
**********************************************************************************/
template <typename ViewType>
typename ViewType::HostMirror
get(ViewType aView)
{
    using RetType = typename ViewType::HostMirror;
    RetType tView = Kokkos::create_mirror(aView);
    Kokkos::deep_copy(tView, aView);
    return tView;
}

void finalizeOmegaH();
void initializeOmegaH(int *argc, char ***argv);
Teuchos::RCP<Omega_h::Library> getLibraryOmegaH();

/******************************************************************************/
//! returns all nodes matching x=0 on the boundary of the provided mesh
inline Omega_h::LOs getBoundaryNodes_x0(Teuchos::RCP<Omega_h::Mesh> & aMesh)
/******************************************************************************/
{
    Omega_h::Int tSpaceDim = aMesh->dim();
    try
    {
        if(tSpaceDim != static_cast<Omega_h::Int>(3))
        {
            std::ostringstream tErrorMsg;
            tErrorMsg << "\n\n ************* ERROR IN FILE: " << __FILE__ << ", FUNCTION: " << __PRETTY_FUNCTION__ << ", LINE: "
                      << __LINE__ << ", MESSAGE: " << "THIS METHOD IS ONLY IMPLEMENTED FOR 3D USE CASES." << " *************\n\n";
            throw std::invalid_argument(tErrorMsg.str().c_str());
        }
    }
    catch(const std::invalid_argument & tErrorMsg)
    {
        std::cout << tErrorMsg.what() << std::flush;
        std::abort();
    }

    // because of the way that build_box does things, the x=0 nodes end up on a face which has label (2,12); the
    // x=1 nodes end up with label (2,14)
    const Omega_h::Int tVertexDim = 0;
    const Omega_h::Int tFaceDim = tSpaceDim - static_cast<Omega_h::Int>(1);
    Omega_h::Read<Omega_h::I8> x0Marks = Omega_h::mark_class_closure(aMesh.get(), tVertexDim, tFaceDim, 12);
    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(x0Marks);

    return (tLocalOrdinals);
}

/***************************************************************************//**
 * \brief Return array of edge ids on edge y=0, assuming a unit-box mesh for illustration.
 *
 *            y=1
 *        -----------
 *        |         |
 *   x=0  |         | x=1
 *        |         |
 *        |         |
 *        -----------
 *            y=0
 *
 * \param [in] aMesh mesh database
 * \return array of edge ids
*******************************************************************************/
inline Omega_h::LOs get_edge_ids_on_y0(Omega_h::Mesh & aMesh)
{
    Omega_h::Read<Omega_h::I8> tMarks = Omega_h::mark_class_closure(&aMesh, Omega_h::EDGE, Omega_h::EDGE, 1 /* class id */);
    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tMarks);
    return (tLocalOrdinals);
}

/***************************************************************************//**
 * \brief Return array of edge ids on edge x=0, assuming a unit-box mesh for illustration.
 *
 *            y=1
 *        -----------
 *        |         |
 *   x=0  |         | x=1
 *        |         |
 *        |         |
 *        -----------
 *            y=0
 *
 * \param [in] aMesh mesh database
 * \return array of edge ids
*******************************************************************************/
inline Omega_h::LOs get_edge_ids_on_x0(Omega_h::Mesh & aMesh)
{
    Omega_h::Read<Omega_h::I8> tMarks = Omega_h::mark_class_closure(&aMesh, Omega_h::EDGE, Omega_h::EDGE, 3 /* class id */);
    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tMarks);
    return (tLocalOrdinals);
}

/***************************************************************************//**
 * \brief Return array of edge ids on edge x=1, assuming a unit-box mesh for illustration.
 *
 *            y=1
 *        -----------
 *        |         |
 *   x=0  |         | x=1
 *        |         |
 *        |         |
 *        -----------
 *            y=0
 *
 * \param [in] aMesh mesh database
 * \return array of edge ids
*******************************************************************************/
inline Omega_h::LOs get_edge_ids_on_x1(Omega_h::Mesh & aMesh)
{
    Omega_h::Read<Omega_h::I8> tMarks = Omega_h::mark_class_closure(&aMesh, Omega_h::EDGE, Omega_h::EDGE, 5 /* class id */);
    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tMarks);
    return (tLocalOrdinals);
}

/***************************************************************************//**
 * \brief Return array of edge ids on edge y=1, assuming a unit-box mesh for illustration.
 *
 *            y=1
 *        -----------
 *        |         |
 *   x=0  |         | x=1
 *        |         |
 *        |         |
 *        -----------
 *            y=0
 *
 * \param [in] aMesh mesh database
 * \return array of edge ids
*******************************************************************************/
inline Omega_h::LOs get_edge_ids_on_y1(Omega_h::Mesh & aMesh)
{
    Omega_h::Read<Omega_h::I8> tMarks = Omega_h::mark_class_closure(&aMesh, Omega_h::EDGE, Omega_h::EDGE, 7 /* class id */);
    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tMarks);
    return (tLocalOrdinals);
}

/******************************************************************************//**
 * \brief Get face IDs on a specific side of the box mesh for applying loads
 *   Specialized for 3-D applications.
 *
 * \param [in] aMesh       finite element mesh
 * \param [in] aBoundaryID boundary identifier
 * \return array of face ids
 **********************************************************************************/
inline Omega_h::LOs get_face_ids_on_boundary_3D(Omega_h::Mesh & aMesh, const std::string & aBoundaryID)
{
    const Omega_h::Int tFaceDim = 2;
    Omega_h::Read<Omega_h::I8> Marks;
    if(aBoundaryID == "x0")
        Marks = Omega_h::mark_class_closure(&aMesh, tFaceDim, tFaceDim, 12);
    else if(aBoundaryID == "x1")
        Marks = Omega_h::mark_class_closure(&aMesh, tFaceDim, tFaceDim, 14);
    else if(aBoundaryID == "y0")
        Marks = Omega_h::mark_class_closure(&aMesh, tFaceDim, tFaceDim, 10);
    else if(aBoundaryID == "y1")
        Marks = Omega_h::mark_class_closure(&aMesh, tFaceDim, tFaceDim, 16);
    else if(aBoundaryID == "z0")
        Marks = Omega_h::mark_class_closure(&aMesh, tFaceDim, tFaceDim, 4);
    else if(aBoundaryID == "z1")
        Marks = Omega_h::mark_class_closure(&aMesh, tFaceDim, tFaceDim, 22);
    else
        THROWERR("Specifed boundary ID not implemented.")

    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(Marks);
    return (tLocalOrdinals);
}

/******************************************************************************/
// This one Tpetra likes; will have to check whether this works with Magma
// Sparse and AmgX or if we need to do something to factor this out
/*! Return a box (cube) mesh.

 \param spaceDim Spatial dimensions of the mesh to be created.
 \param meshWidth Number of mesh intervals through the thickness.
 */
inline Teuchos::RCP<Omega_h::Mesh> getBoxMesh(Omega_h::Int aSpaceDim,
                                              Omega_h::Int aMeshWidth,
                                              Plato::Scalar aX_scaling = 1.0,
                                              Plato::Scalar aY_scaling = -1.0,
                                              Plato::Scalar aZ_scaling = -1.0)
/******************************************************************************/
{
    if(aY_scaling == -1.0)
    {
        aY_scaling = aX_scaling;
    }
    if(aZ_scaling == -1.0)
    {
        aZ_scaling = aY_scaling;
    }

    Omega_h::Int tNumX = 0, tNumY = 0, tNumZ = 0;
    if(aSpaceDim == 1)
    {
        tNumX = aMeshWidth;
    }
    else if(aSpaceDim == 2)
    {
        tNumX = aMeshWidth;
        tNumY = aMeshWidth;
    }
    else if(aSpaceDim == 3)
    {
        tNumX = aMeshWidth;
        tNumY = aMeshWidth;
        tNumZ = aMeshWidth;
    }

    Teuchos::RCP<Omega_h::Library> tLibOmegaH = getLibraryOmegaH();
    auto tOmegaH_mesh = Teuchos::rcp(new Omega_h::Mesh(Omega_h::build_box(tLibOmegaH->world(),
                                                                          OMEGA_H_SIMPLEX,
                                                                          aX_scaling,
                                                                          aY_scaling,
                                                                          aZ_scaling,
                                                                          tNumX,
                                                                          tNumY,
                                                                          tNumZ)));
    return (tOmegaH_mesh);
}

/******************************************************************************/
inline
void writeConnectivity(const Teuchos::RCP<Omega_h::Mesh> & aMeshOmegaH, const std::string & aName, const Omega_h::Int & aSpaceDim)
/******************************************************************************/
{
#ifndef KOKKOS_ENABLE_CUDA

    std::ofstream tOutfile(aName);

    auto tNumCells = aMeshOmegaH->nelems();
    auto tCells2nodes = aMeshOmegaH->ask_elem_verts();
    const Omega_h::Int tNumNodesPerCell = aSpaceDim + 1;
    for(Omega_h::Int tCellIndex = 0; tCellIndex < tNumCells; tCellIndex++)
    {
        for(Omega_h::Int tNodeIndex = 0; tNodeIndex < tNumNodesPerCell; tNodeIndex++)
        {
            tOutfile << tCells2nodes[tCellIndex * tNumNodesPerCell + tNodeIndex] << " ";
        }
        tOutfile << std::endl;
    }
    tOutfile.close();
#else
    (void)aMeshOmegaH;
    (void)aName;
    (void)aSpaceDim;
#endif
}

/******************************************************************************/
inline
void writeMesh(const Teuchos::RCP<Omega_h::Mesh> & aMeshOmegaH, const std::string & aName, const Omega_h::Int & aSpaceDim)
/******************************************************************************/
{
#ifndef KOKKOS_ENABLE_CUDA
    Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aName, aMeshOmegaH.get(), aSpaceDim);
    auto tTags = Omega_h::vtk::get_all_vtk_tags(aMeshOmegaH.get(), aSpaceDim);
    tWriter.write(static_cast<Omega_h::Real>(1.0), tTags);
#else
    (void)aMeshOmegaH;
    (void)aName;
    (void)aSpaceDim;
#endif
}

/******************************************************************************//**
 *
 * \brief Build 1D box mesh
 *
 * \param[in] aX x-dimension length
 * \param[in] aNx number of space in x-dimension
 *
 **********************************************************************************/
inline std::shared_ptr<Omega_h::Mesh> build_1d_box_mesh(Omega_h::Real aX, Omega_h::LO aNx)
{
    Teuchos::RCP<Omega_h::Library> tLibrary = getLibraryOmegaH();
    std::shared_ptr<Omega_h::Mesh> tMesh = std::make_shared<Omega_h::Mesh>(
            Omega_h::build_box(tLibrary->world(), Omega_h_Family::OMEGA_H_SIMPLEX, aX, 0., 0., aNx, 0, 0));
    return (tMesh);
}

/******************************************************************************//**
 *
 * \brief Build 2D box mesh
 *
 * \param[in] aX x-dimension length
 * \param[in] aY y-dimension length
 * \param[in] aNx number of space in x-dimension
 * \param[in] aNy number of space in y-dimension
 *
 **********************************************************************************/
inline std::shared_ptr<Omega_h::Mesh> build_2d_box_mesh(Omega_h::Real aX, Omega_h::Real aY, Omega_h::LO aNx, Omega_h::LO aNy)
{
    Teuchos::RCP<Omega_h::Library> tLibrary = getLibraryOmegaH();
    std::shared_ptr<Omega_h::Mesh> tMesh = std::make_shared<Omega_h::Mesh>(
            Omega_h::build_box(tLibrary->world(), Omega_h_Family::OMEGA_H_SIMPLEX, aX, aY, 0., aNx, aNy, 0) );
    return (tMesh);
}

/******************************************************************************//**
 *
 * \brief Build 3D box mesh
 *
 * \param[in] aX x-dimension length
 * \param[in] aY y-dimension length
 * \param[in] aZ z-dimension length
 * \param[in] aNx number of space in x-dimension
 * \param[in] aNy number of space in y-dimension
 * \param[in] aNz number of space in z-dimension
 *
 **********************************************************************************/
inline std::shared_ptr<Omega_h::Mesh> build_3d_box_mesh(Omega_h::Real aX,
                                                        Omega_h::Real aY,
                                                        Omega_h::Real aZ,
                                                        Omega_h::LO aNx,
                                                        Omega_h::LO aNy,
                                                        Omega_h::LO aNz)
{
    Teuchos::RCP<Omega_h::Library> tLibrary = getLibraryOmegaH();
    std::shared_ptr<Omega_h::Mesh> tMesh = std::make_shared<Omega_h::Mesh>(
            Omega_h::build_box(tLibrary->world(), Omega_h_Family::OMEGA_H_SIMPLEX, aX, aY, aZ, aNx, aNy, aNz) );
    return (tMesh);
}

/******************************************************************************//**
 *
 * \brief Get node ordinals associated with the boundary
 * \param[in] aMesh mesh data base
 *
 **********************************************************************************/
inline Omega_h::LOs get_boundary_nodes(Omega_h::Mesh & aMesh)
{
    auto tSpaceDim = aMesh.dim();

    Omega_h::Read<Omega_h::I8> tInteriorMarks = Omega_h::mark_by_class_dim(&aMesh, Omega_h::VERT, tSpaceDim);
    Omega_h::Read<Omega_h::I8> tBoundaryMarks = Omega_h::invert_marks(tInteriorMarks);
    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tBoundaryMarks);

    return tLocalOrdinals;
}

/******************************************************************************//**
 *
 * \brief Get node ordinals associated with boundary edge y=0
 * \param[in] aMesh mesh data base
 *
 **********************************************************************************/
inline Omega_h::LOs get_2D_boundary_nodes_y0(Omega_h::Mesh& aMesh)
{
    // the y=0 nodes end up on a face which has label (1,1);
    Omega_h::Read<Omega_h::I8> tMarks = Omega_h::mark_class_closure(&aMesh, Omega_h::VERT, Omega_h::EDGE, 1);
    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tMarks);
    return tLocalOrdinals;
}

/******************************************************************************//**
 *
 * \brief Get node ordinals associated with boundary edge x=0
 * \param[in] aMesh mesh data base
 *
 **********************************************************************************/
inline Omega_h::LOs get_2D_boundary_nodes_x0(Omega_h::Mesh& aMesh)
{
    // the x=0 nodes end up on an edge which has label (1,3);
    Omega_h::Read<Omega_h::I8> tMarks = Omega_h::mark_class_closure(&aMesh, Omega_h::VERT, Omega_h::EDGE, 3);
    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tMarks);
    return tLocalOrdinals;
}

/******************************************************************************//**
 *
 * \brief Get node ordinals associated with boundary edge x=1
 * \param[in] aMesh mesh data base
 *
 **********************************************************************************/
inline Omega_h::LOs get_2D_boundary_nodes_x1(Omega_h::Mesh& aMesh)
{
    // the x=1 nodes end up on an edge which has label (1,5),
    Omega_h::Read<Omega_h::I8> tMarks = Omega_h::mark_class_closure(&aMesh, Omega_h::VERT, Omega_h::EDGE, 5);
    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tMarks);
    return tLocalOrdinals;
}

/******************************************************************************//**
 *
 * \brief Get node ordinals associated with boundary edge y=1
 * \param[in] aMesh mesh data base
 *
 **********************************************************************************/
inline Omega_h::LOs get_2D_boundary_nodes_y1(Omega_h::Mesh& aMesh)
{
    // the x=1 nodes end up on an edge which has label (1,7).
    Omega_h::Read<Omega_h::I8> tMarks = Omega_h::mark_class_closure(&aMesh, Omega_h::VERT, Omega_h::EDGE, 7);
    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(tMarks);
    return tLocalOrdinals;
}

/******************************************************************************//**
 * \brief Return list of boundary nodes, specialized for 3-D applications.
 *
 * \param [in]     aMesh       finite element mesh
 * \param [in]     aBoundaryID boundary identifier
 *
 * \return list of boundary nodes
 *
 **********************************************************************************/
inline Omega_h::LOs get_boundary_nodes_3D(Omega_h::Mesh & aMesh, const std::string & aBoundaryID)
{
    const Omega_h::Int tVertexDim = 0;
    const Omega_h::Int tFaceDim = 2;
    Omega_h::Read<Omega_h::I8> Marks;
    if(aBoundaryID == "x0")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 12);
    else if(aBoundaryID == "x1")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 14);
    else if(aBoundaryID == "y0")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 10);
    else if(aBoundaryID == "y1")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 16);
    else if(aBoundaryID == "z0")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 4);
    else if(aBoundaryID == "z1")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 22);
    else
        THROWERR("Specified boundary ID is not defined.")

    Omega_h::LOs tBoundaryNodes = Omega_h::collect_marked(Marks);

    return (tBoundaryNodes);
}

/******************************************************************************//**
 * \brief Label boundarys with nodesets and sidesets, specialized for 2-D applications.
 *
 * \param [in]     aMesh       finite element mesh
 * \param [in]     aMeshSets   mesh sets to be filled
 *
 *
 **********************************************************************************/
inline void set_mesh_sets_2D(Omega_h::Mesh & aMesh, Omega_h::MeshSets & aMeshSets)
{
    auto tNodesX0 = PlatoUtestHelpers::get_2D_boundary_nodes_x0(aMesh);
    auto tNodesX1 = PlatoUtestHelpers::get_2D_boundary_nodes_x1(aMesh);
    auto tNodesY0 = PlatoUtestHelpers::get_2D_boundary_nodes_y0(aMesh);
    auto tNodesY1 = PlatoUtestHelpers::get_2D_boundary_nodes_y1(aMesh);

    aMeshSets[Omega_h::NODE_SET]["ns_X0"] = tNodesX0;
    aMeshSets[Omega_h::NODE_SET]["ns_X1"] = tNodesX1;
    aMeshSets[Omega_h::NODE_SET]["ns_Y0"] = tNodesY0;
    aMeshSets[Omega_h::NODE_SET]["ns_Y1"] = tNodesY1;

    auto tFacesX0 = PlatoUtestHelpers::get_edge_ids_on_x0(aMesh);
    auto tFacesX1 = PlatoUtestHelpers::get_edge_ids_on_x1(aMesh);
    auto tFacesY0 = PlatoUtestHelpers::get_edge_ids_on_y0(aMesh);
    auto tFacesY1 = PlatoUtestHelpers::get_edge_ids_on_y1(aMesh);

    aMeshSets[Omega_h::SIDE_SET]["ss_X0"] = tFacesX0;
    aMeshSets[Omega_h::SIDE_SET]["ss_X1"] = tFacesX1;
    aMeshSets[Omega_h::SIDE_SET]["ss_Y0"] = tFacesY0;
    aMeshSets[Omega_h::SIDE_SET]["ss_Y1"] = tFacesY1;
}

/******************************************************************************//**
 * \brief Label boundarys with nodesets and sidesets, specialized for 3-D applications.
 *
 * \param [in]     aMesh       finite element mesh
 * \param [in]     aMeshSets   mesh sets to be filled
 *
 *
 **********************************************************************************/
inline void set_mesh_sets_3D(Omega_h::Mesh & aMesh, Omega_h::MeshSets & aMeshSets)
{
    auto tNodesX0 = PlatoUtestHelpers::get_boundary_nodes_3D(aMesh, "x0");
    auto tNodesX1 = PlatoUtestHelpers::get_boundary_nodes_3D(aMesh, "x1");
    auto tNodesY0 = PlatoUtestHelpers::get_boundary_nodes_3D(aMesh, "y0");
    auto tNodesY1 = PlatoUtestHelpers::get_boundary_nodes_3D(aMesh, "y1");
    auto tNodesZ0 = PlatoUtestHelpers::get_boundary_nodes_3D(aMesh, "z0");
    auto tNodesZ1 = PlatoUtestHelpers::get_boundary_nodes_3D(aMesh, "z1");

    aMeshSets[Omega_h::NODE_SET]["ns_X0"] = tNodesX0;
    aMeshSets[Omega_h::NODE_SET]["ns_X1"] = tNodesX1;
    aMeshSets[Omega_h::NODE_SET]["ns_Y0"] = tNodesY0;
    aMeshSets[Omega_h::NODE_SET]["ns_Y1"] = tNodesY1;
    aMeshSets[Omega_h::NODE_SET]["ns_Z0"] = tNodesZ0;
    aMeshSets[Omega_h::NODE_SET]["ns_Z1"] = tNodesZ1;

    auto tFacesX0 = PlatoUtestHelpers::get_face_ids_on_boundary_3D(aMesh, "x0");
    auto tFacesX1 = PlatoUtestHelpers::get_face_ids_on_boundary_3D(aMesh, "x1");
    auto tFacesY0 = PlatoUtestHelpers::get_face_ids_on_boundary_3D(aMesh, "y0");
    auto tFacesY1 = PlatoUtestHelpers::get_face_ids_on_boundary_3D(aMesh, "y1");
    auto tFacesZ0 = PlatoUtestHelpers::get_face_ids_on_boundary_3D(aMesh, "z0");
    auto tFacesZ1 = PlatoUtestHelpers::get_face_ids_on_boundary_3D(aMesh, "z1");

    aMeshSets[Omega_h::SIDE_SET]["ss_X0"] = tFacesX0;
    aMeshSets[Omega_h::SIDE_SET]["ss_X1"] = tFacesX1;
    aMeshSets[Omega_h::SIDE_SET]["ss_Y0"] = tFacesY0;
    aMeshSets[Omega_h::SIDE_SET]["ss_Y1"] = tFacesY1;
    aMeshSets[Omega_h::SIDE_SET]["ss_Z0"] = tFacesZ0;
    aMeshSets[Omega_h::SIDE_SET]["ss_Z1"] = tFacesZ1;
}

/******************************************************************************//**
 *
 * \brief Set point load
 * \param[in] aNodeOrdinal node ordinal associated with point load
 * \param[in] aNodeOrdinals collection of node ordinals associated with the entity (point, edge or surface) were point load is applied
 * \param[in] aValues values associated with point load
 * \param[in,out] aOutput global point load
 *
 **********************************************************************************/
inline void set_point_load(const Omega_h::LO& aNodeOrdinal,
                           const Omega_h::LOs& aNodeOrdinals,
                           const Plato::ScalarMultiVector& aValues,
                           Plato::ScalarVector& aOutput)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aValues.extent(0)), LAMBDA_EXPRESSION(const Plato::OrdinalType& aIndex)
    {
        auto tOffset = aIndex * aValues.extent(1);
        auto tNumDofsPerNode = aValues.extent(0) * aValues.extent(1);
        auto tMyNodeDof = tNumDofsPerNode * aNodeOrdinals[aNodeOrdinal];
        for(Plato::OrdinalType tDim = 0; tDim < Plato::OrdinalType(aValues.extent(1)); tDim++)
        {
            auto tOutputIndex = tMyNodeDof + tOffset + tDim;
            aOutput(tOutputIndex) = aValues(aIndex, tDim);
        }
    }, "set point load");
}

/******************************************************************************//**
 *
 * \brief Set Dirichlet boundary conditions.
 *
 * \param[in] aNumDofsPerNode number of degrees of freedom per node
 * \param[in] aValue constant value associated with Dirichlet boundary conditions (only constant values are supported)
 * \param[in] aCoords coordinates associated with Dirichlet boundary conditions
 * \param[in,out] aDirichletValues values associated with Dirichlet boundary conditions
 *
 **********************************************************************************/
inline void set_dirichlet_boundary_conditions(const Plato::OrdinalType& aNumDofsPerNode,
                                              const Plato::Scalar& aValue,
                                              const Omega_h::LOs& aCoords,
                                              Plato::LocalOrdinalVector& aDirichletDofs,
                                              Plato::ScalarVector& aDirichletValues)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aCoords.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType& aIndex)
    {
        auto tOffset = aIndex*aNumDofsPerNode;
        for (Plato::OrdinalType tDof = 0; tDof < aNumDofsPerNode; tDof++)
        {
            aDirichletDofs[tOffset + tDof] = aNumDofsPerNode*aCoords[aIndex] + tDof;
            aDirichletValues[tOffset + tDof] = aValue;
        }
    }, "Dirichlet BC");
}

/******************************************************************************//**
 *
 * \brief Print ordinals' values
 * \param[in] aInput array of ordinals
 *
 **********************************************************************************/
inline void print_ordinals(const Omega_h::LOs& aInput)
{
    auto tRange = aInput.size();
    Kokkos::parallel_for("print ordinals", tRange, LAMBDA_EXPRESSION(const int & aIndex)
    {
        printf("[%d]\n", aInput[aIndex]);
    });
}

/******************************************************************************//**
 *
 * \brief Print 1D coordinates associated with node ordinals
 * \param[in] aMesh mesg data base
 * \param[in] aNodeOrdinals array of node ordinals
 *
 **********************************************************************************/
inline void print_1d_coords(const Omega_h::Mesh& aMesh, const Omega_h::LOs& aNodeOrdinals)
{
    auto tSpaceDim = aMesh.dim();
    assert(tSpaceDim == static_cast<Omega_h::Int>(1));
    auto tCoords = aMesh.coords();
    auto tNumNodes = aNodeOrdinals.size();

    printf("\nPrint 1D Coordinates (X)\n");

    // the following will only work well in serial mode on host -- this is just for basic sanity checking
    Kokkos::parallel_for("print coords", tNumNodes, LAMBDA_EXPRESSION(const int & aNodeIndex)
    {
        auto tVertexNumber = aNodeOrdinals[aNodeIndex];
        auto tEntryOffset = tVertexNumber * tSpaceDim;
        auto tX = tCoords[tEntryOffset + 0];
        printf("Node(%d)=(%f)\n", tVertexNumber, tX);
    });
}

/******************************************************************************//**
 *
 * \brief Print 2D coordinates associated with node ordinals
 * \param[in] aMesh mesg data base
 * \param[in] aNodeOrdinals array of node ordinals
 *
 **********************************************************************************/
inline void print_2d_coords(const Omega_h::Mesh& aMesh, const Omega_h::LOs& aNodeOrdinals)
{
    auto tSpaceDim = aMesh.dim();
    assert(tSpaceDim == static_cast<Omega_h::Int>(2));
    auto tCoords = aMesh.coords();
    auto tNumNodes = aNodeOrdinals.size();

    printf("\nPrint 2D Coordinates (X,Y)\n");

    // the following will only work well in serial mode on host -- this is just for basic sanity checking
    Kokkos::parallel_for("print coords", tNumNodes, LAMBDA_EXPRESSION(const int & aNodeIndex)
    {
        auto tVertexNumber = aNodeOrdinals[aNodeIndex];
        auto tEntryOffset = tVertexNumber * tSpaceDim;
        auto tX = tCoords[tEntryOffset + 0];
        auto tY = tCoords[tEntryOffset + 1];
        printf("Node(%d)=(%f,%f)\n", tVertexNumber, tX, tY);
    });
}

/******************************************************************************//**
 *
 * \param[in] aMesh mesg data base
 * \param[in] aNodeOrdinals array of node ordinals
 *
 **********************************************************************************/
inline void print_3d_coords(const Omega_h::Mesh& aMesh, const Omega_h::LOs& aNodeOrdinals)
{
    auto tSpaceDim = aMesh.dim();
    assert(tSpaceDim == static_cast<Omega_h::Int>(3));
    auto tCoords = aMesh.coords();
    auto tNumNodes = aNodeOrdinals.size();

    printf("\nPrint 3D Coordinates (X,Y,Z)\n");

    // the following will only work well in serial mode on host -- this is just for basic sanity checking
    Kokkos::parallel_for("print coords", tNumNodes, LAMBDA_EXPRESSION(const int & aNodeIndex)
    {
        auto tVertexNumber = aNodeOrdinals[aNodeIndex];
        auto tEntryOffset = tVertexNumber * tSpaceDim;
        auto tX = tCoords[tEntryOffset + 0];
        auto tY = tCoords[tEntryOffset + 1];
        auto tZ = tCoords[tEntryOffset + 2];
        printf("Node(%d)=(%f,%f,%f)\n", tVertexNumber, tX, tY, tZ);
    });
}

/******************************************************************************//**
 * \brief Set Dirichlet boundary condition values for specified degree of freedom.
 *   Specialized for 2-D applications
 *
 * \param [in] aMesh       finite element mesh
 * \param [in] aBoundaryID boundary identifier
 * \param [in] aDofValues  vector of Dirichlet boundary condition values
 * \param [in] aDofStride  degree of freedom stride
 * \param [in] aDofToSet   degree of freedom index to set
 * \param [in] aSetValue   value to set
 *
 **********************************************************************************/
inline void set_dof_value_in_vector_on_boundary_2D(Omega_h::Mesh & aMesh,
                                                   const std::string & aBoundaryID,
                                                   const Plato::ScalarVector & aDofValues,
                                                   const Plato::OrdinalType & aDofStride,
                                                   const Plato::OrdinalType & aDofToSet,
                                                   const Plato::Scalar & aSetValue)
{
    Omega_h::LOs tBoundaryNodes;
    if(aBoundaryID == "x0")
        tBoundaryNodes = PlatoUtestHelpers::get_2D_boundary_nodes_x0(aMesh);
    else if(aBoundaryID == "x1")
        tBoundaryNodes = PlatoUtestHelpers::get_2D_boundary_nodes_x1(aMesh);
    else if(aBoundaryID == "y0")
        tBoundaryNodes = PlatoUtestHelpers::get_2D_boundary_nodes_y0(aMesh);
    else if(aBoundaryID == "y1")
        tBoundaryNodes = PlatoUtestHelpers::get_2D_boundary_nodes_y1(aMesh);
    else
        THROWERR("Specifed boundary ID not implemented.")

    auto tNumBoundaryNodes = tBoundaryNodes.size();

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumBoundaryNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        Plato::OrdinalType tIndex = aDofStride * tBoundaryNodes[aIndex] + aDofToSet;
        aDofValues(tIndex) += aSetValue;
    },
                         "fill vector boundary dofs");
}

/******************************************************************************//**
 * \brief Set Dirichlet boundary condition values for specified degree of freedom.
 *   Specialized for 3-D applications.
 *
 * \param [in] aMesh       finite element mesh
 * \param [in] aBoundaryID boundary identifier
 * \param [in] aDofValues  vector of Dirichlet boundary condition values
 * \param [in] aDofStride  degree of freedom stride
 * \param [in] aDofToSet   degree of freedom index to set
 * \param [in] aSetValue   value to set
 *
 **********************************************************************************/
inline void set_dof_value_in_vector_on_boundary_3D(Omega_h::Mesh & aMesh,
                                                   const std::string & aBoundaryID,
                                                   const Plato::ScalarVector & aDofValues,
                                                   const Plato::OrdinalType & aDofStride,
                                                   const Plato::OrdinalType & aDofToSet,
                                                   const Plato::Scalar & aSetValue)
{
    const Omega_h::Int tVertexDim = 0;
    const Omega_h::Int tFaceDim = 2;
    Omega_h::Read<Omega_h::I8> Marks;
    if(aBoundaryID == "x0")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 12);
    else if(aBoundaryID == "x1")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 14);
    else if(aBoundaryID == "y0")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 10);
    else if(aBoundaryID == "y1")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 16);
    else if(aBoundaryID == "z0")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 4);
    else if(aBoundaryID == "z1")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 22);
    else
        THROWERR("Specifed boundary ID not implemented.")

    Omega_h::LOs tLocalOrdinals = Omega_h::collect_marked(Marks);
    auto tNumBoundaryNodes = tLocalOrdinals.size();

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumBoundaryNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        Plato::OrdinalType tIndex = aDofStride * tLocalOrdinals[aIndex] + aDofToSet;
        aDofValues(tIndex) += aSetValue;
    }, "fill vector boundary dofs");
}

/******************************************************************************//**
 * \brief Return list of Dirichlet degree of freedom indices, specialized for 2-D applications.
 *
 * \param [in] aMesh       finite element mesh
 * \param [in] aBoundaryID boundary identifier
 * \param [in] aDofStride  degree of freedom stride
 * \param [in] aDofToSet   degree of freedom index to set
 *
 * \return list of Dirichlet indices
 *
 **********************************************************************************/
inline Plato::LocalOrdinalVector get_dirichlet_indices_on_boundary_2D(Omega_h::Mesh & aMesh,
                                                                      const std::string & aBoundaryID,
                                                                      const Plato::OrdinalType & aDofStride,
                                                                      const Plato::OrdinalType & aDofToSet)
{
    Omega_h::LOs tBoundaryNodes;
    if(aBoundaryID == "x0")
        tBoundaryNodes = PlatoUtestHelpers::get_2D_boundary_nodes_x0(aMesh);
    else if(aBoundaryID == "x1")
        tBoundaryNodes = PlatoUtestHelpers::get_2D_boundary_nodes_x1(aMesh);
    else if(aBoundaryID == "y0")
        tBoundaryNodes = PlatoUtestHelpers::get_2D_boundary_nodes_y0(aMesh);
    else if(aBoundaryID == "y1")
        tBoundaryNodes = PlatoUtestHelpers::get_2D_boundary_nodes_y1(aMesh);
    else
        THROWERR("Specifed boundary ID not implemented.")

    Plato::LocalOrdinalVector tDofIndices;
    auto tNumBoundaryNodes = tBoundaryNodes.size();
    Kokkos::resize(tDofIndices, tNumBoundaryNodes);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumBoundaryNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        Plato::OrdinalType tIndex = aDofStride * tBoundaryNodes[aIndex] + aDofToSet;
        tDofIndices(aIndex) = tIndex;
    }, "fill dirichlet dof indices on boundary 2-D");

    return (tDofIndices);
}

/******************************************************************************//**
 * \brief Return list of Dirichlet degree of freedom indices, specialized for 3-D applications.
 *
 * \param [in]     aMesh       finite element mesh
 * \param [in]     aBoundaryID boundary identifier
 * \param [in]     aDofStride  degree of freedom stride
 * \param [in]     aDofToSet   degree of freedom index to set
 *
 * \return list of Dirichlet indices
 *
 **********************************************************************************/
inline Plato::LocalOrdinalVector get_dirichlet_indices_on_boundary_3D(Omega_h::Mesh & aMesh,
                                                                      const std::string & aBoundaryID,
                                                                      const Plato::OrdinalType & aDofStride,
                                                                      const Plato::OrdinalType & aDofToSet)
{
    const Omega_h::Int tVertexDim = 0;
    const Omega_h::Int tFaceDim = 2;
    Omega_h::Read<Omega_h::I8> Marks;
    if(aBoundaryID == "x0")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 12);
    else if(aBoundaryID == "x1")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 14);
    else if(aBoundaryID == "y0")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 10);
    else if(aBoundaryID == "y1")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 16);
    else if(aBoundaryID == "z0")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 4);
    else if(aBoundaryID == "z1")
        Marks = Omega_h::mark_class_closure(&aMesh, tVertexDim, tFaceDim, 22);
    else
        THROWERR("Specified boundary ID is not defined.")

    Omega_h::LOs tBoundaryNodes = Omega_h::collect_marked(Marks);

    Plato::LocalOrdinalVector tDofIndices;
    auto tNumBoundaryNodes = tBoundaryNodes.size();
    Kokkos::resize(tDofIndices, tNumBoundaryNodes);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumBoundaryNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aIndex)
    {
        Plato::OrdinalType tIndex = aDofStride * tBoundaryNodes[aIndex] + aDofToSet;
        tDofIndices(aIndex) = tIndex;
    }, "fill dirichlet dof indices on boundary 3-D");

    return (tDofIndices);
}

/******************************************************************************//**
 * \brief set value for this Dirichlet boundary condition index
 *
 * \param [in] aDofValues vector of Dirichlet boundary condition values
 * \param [in] aDofStride degree of freedom stride
 * \param [in] aDofToSet  degree of freedom index to set
 * \param [in] aSetValue  value to set
 *
 **********************************************************************************/
inline void set_dof_value_in_vector(const Plato::ScalarVector & aDofValues,
                                    const Plato::OrdinalType & aDofStride,
                                    const Plato::OrdinalType & aDofToSet,
                                    const Plato::Scalar & aSetValue)
{
    auto tVectorSize = aDofValues.extent(0);
    auto tRange = tVectorSize / aDofStride;

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tRange), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeIndex)
    {
        Plato::OrdinalType tIndex = aDofStride * aNodeIndex + aDofToSet;
        aDofValues(tIndex) += aSetValue;
    }, "fill specific vector entry globally");
}

inline std::vector<std::vector<Plato::Scalar>>
toFull( Teuchos::RCP<Plato::CrsMatrixType> aInMatrix )
{
    using Plato::OrdinalType;
    using Plato::Scalar;

    std::vector<std::vector<Scalar>>
        retMatrix(aInMatrix->numRows(),std::vector<Scalar>(aInMatrix->numCols(),0.0));

    auto tNumRowsPerBlock = aInMatrix->numRowsPerBlock();
    auto tNumColsPerBlock = aInMatrix->numColsPerBlock();
    auto tBlockSize = tNumRowsPerBlock*tNumColsPerBlock;

    auto tRowMap = get(aInMatrix->rowMap());
    auto tColMap = get(aInMatrix->columnIndices());
    auto tValues = get(aInMatrix->entries());

    auto tNumRows = tRowMap.extent(0)-1;
    for(OrdinalType iRowIndex=0; iRowIndex<tNumRows; iRowIndex++)
    {
        auto tFrom = tRowMap(iRowIndex);
        auto tTo   = tRowMap(iRowIndex+1);
        for(auto iColMapEntryIndex=tFrom; iColMapEntryIndex<tTo; iColMapEntryIndex++)
        {
            auto tBlockColIndex = tColMap(iColMapEntryIndex);
            for(OrdinalType iLocalRowIndex=0; iLocalRowIndex<tNumRowsPerBlock; iLocalRowIndex++)
            {
                auto tRowIndex = iRowIndex * tNumRowsPerBlock + iLocalRowIndex;
                for(OrdinalType iLocalColIndex=0; iLocalColIndex<tNumColsPerBlock; iLocalColIndex++)
                {
                    auto tColIndex = tBlockColIndex * tNumColsPerBlock + iLocalColIndex;
                    auto tSparseIndex = iColMapEntryIndex * tBlockSize + iLocalRowIndex * tNumColsPerBlock + iLocalColIndex;
                    retMatrix[tRowIndex][tColIndex] = tValues[tSparseIndex];
                }
            }
        }
    }
    return retMatrix;
}

/******************************************************************************//**
 * \brief ignore a variable and suppress compiler warnings :)
 *
 * \tparam [in] Any typename
 **********************************************************************************/
template <typename T>
void ignore_unused_variable_warning(T &&) {}

} // namespace PlatoUtestHelpers

#endif /* PLATOTESTHELPERS_HPP_ */

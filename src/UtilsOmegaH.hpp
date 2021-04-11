/*
 * UtilsOmegaH.hpp
 *
 *  Created on: Apr 5, 2021
 */

#pragma once

#include <Omega_h_vtk.hpp>
#include <Omega_h_map.hpp>
#include <Omega_h_few.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_mark.hpp>
#include <Omega_h_assoc.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_defines.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_filesystem.hpp>

#include "Variables.hpp"
#include "PlatoUtilities.hpp"
#include "ImplicitFunctors.hpp"

namespace Plato
{

namespace omega_h
{

/******************************************************************************//**
 * \tparam ViewType view type
 *
 * \fn inline void print_fad_dx_values
 *
 * \brief Copy Kokkos view into an Omega_h LOs array.
 *
 * \param [in] aInput input 1D view
**********************************************************************************/
template<typename ViewType>
inline Omega_h::LOs copy
(const ScalarVectorT<ViewType> & aInput)
{
    auto tLength = aInput.size();
    Omega_h::Write<ViewType> tWrite(tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tWrite[aOrdinal] = aInput(aOrdinal);
    }, "copy");

    return (Omega_h::LOs(tWrite));
}
// function copy

/******************************************************************************//**
 * \tparam ViewType Omega_h array type
 *
 * \fn void print
 *
 * \brief Print Omega_h array to terminal.
 *
 * \param [in] aInput Omega_h array
 * \param [in] aName  name used to identify Omega_h array
**********************************************************************************/
template<typename ViewType>
void print
(const ViewType & aInput,
 const std::string & aName)
{
    std::cout << "Start Printing Array with Name '" << aName << "'\n";
    auto tLength = aInput.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        printf("Array(%d)=%d\n",aOrdinal,aInput[aOrdinal]);
    }, "print");
    std::cout << "Finished Printing Array with Name '" << aName << "'\n";
}
// function print

/******************************************************************************//**
 * \tparam NumSpatialDims  number of spatial dimensions
 * \tparam NumNodesPerCell number of nodes per cell/element
 *
 * \fn Scalar calculate_element_size
 *
 * \brief Calculate characteristic element size
 *
 * \param [in] aCellOrdinal cell/element ordinal
 * \param [in] aCells2Nodes map from cells to node ordinal
 * \param [in] aCoords      cell/element coordinates
**********************************************************************************/
template<Plato::OrdinalType NumSpatialDims,
         Plato::OrdinalType NumNodesPerCell>
DEVICE_TYPE inline
Plato::Scalar
calculate_element_size
(const Plato::OrdinalType & aCellOrdinal,
 const Omega_h::LOs & aCells2Nodes,
 const Omega_h::Reals & aCoords)
{
    Omega_h::Few<Omega_h::Vector<NumSpatialDims>, NumNodesPerCell> tElemCoords;
    for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
    {
        const Plato::OrdinalType tVertexIndex = aCells2Nodes[aCellOrdinal*NumNodesPerCell + tNode];
        for(Plato::OrdinalType tDim = 0; tDim < NumSpatialDims; tDim++)
        {
            tElemCoords[tNode][tDim] = aCoords[tVertexIndex*NumSpatialDims + tDim];
        }
    }
    auto tSphere = Omega_h::get_inball(tElemCoords);

    return (static_cast<Plato::Scalar>(2.0) * tSphere.r);
}
// function calculate_element_size

/******************************************************************************//**
 * \fn inline std::string get_entity_name
 *
 * \brief Return entity type in string format.
 *
 * \param [in] aEntityDim Omega_h entity dimension
 * \return entity type in string format
**********************************************************************************/
inline std::string
get_entity_name
(const Omega_h::Int aEntityDim)
{
    std::unordered_map<Omega_h::Int, std::string> tMap =
        { {0, "VERTEX"}, {1, "EDGE"}, {2, "FACE"}, {3, "REGION"} };
    auto tItr = tMap.find(aEntityDim);
    if(tItr == tMap.end())
    {
        THROWERR(std::string("Entity dimension '") + std::to_string(aEntityDim) + "' is not supported. "
            + "Supported entity dimensions are: Omega_h::VERT=0, Omega_h::EDGE=1, Omega_h::FACE=2, and Omega_h::REGION=3")
    }
    return tItr->second;
}
// function get_entity_name

/******************************************************************************//**
 * \fn inline Plato::ScalarVector read_metadata_from_mesh
 *
 * \brief Read metadata from finite element mesh.
 *
 * \param [in] aMesh      finite element mesh metadata
 * \param [in] aEntityDim Omega_h entity dimension
 * \param [in] aTagName   field tag
 *
 * \return field data
**********************************************************************************/
inline Plato::ScalarVector
read_metadata_from_mesh
(const Omega_h::Mesh& aMesh,
 const Omega_h::Int aEntityDim,
 const std::string& aTagName)
{
    if(aMesh.has_tag(aEntityDim, aTagName) == false)
    {
        auto tEntityName = Plato::omega_h::get_entity_name(aEntityDim);
        THROWERR(std::string("Tag '") + aTagName + "' with entity dimension '" + tEntityName + "' is not defined in mesh.")
    }
    auto tTag = aMesh.get_tag<Omega_h::Real>(aEntityDim, aTagName);
    auto tData = tTag->array();
    if(tData.size() <= static_cast<Omega_h::LO>(0))
    {
        THROWERR(std::string("Read array with name '") + aTagName + "' is empty.")
    }
    const Plato::OrdinalType tSize = tData.size();
    Plato::ScalarVector tOutput(aTagName, tSize);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tSize), LAMBDA_EXPRESSION(const Plato::OrdinalType& tIndex)
    {
        tOutput(tIndex) = tData[tIndex];
    }, "copy read array into output array");
    return tOutput;
}
// function read_metadata_from_mesh

/******************************************************************************//**
 * \fn inline std::vector<Omega_h::filesystem::path> read_pvtu_file_paths
 *
 * \brief Return .pvtu file paths.
 *
 * \param [in] aPvdDir .pvd file path
 *
 * \return array of .pvtu file paths
**********************************************************************************/
inline std::vector<Omega_h::filesystem::path>
read_pvtu_file_paths
(const std::string & aPvdDir)
{
    std::vector<Omega_h::Real> tTimes;
    std::vector<Omega_h::filesystem::path> tPvtuPaths;
    auto const tPvdPath = Omega_h::vtk::get_pvd_path(aPvdDir);
    Omega_h::vtk::read_pvd(tPvdPath, &tTimes, &tPvtuPaths);
    if(tPvtuPaths.empty())
    {
        THROWERR("Array with .pvtu file paths is empty.")
    }
    return tPvtuPaths;
}
// function read_pvtu_file_paths

/******************************************************************************//**
 * \tparam EntitySet  entity set type (Omega_h::EntitySet)
 *
 * \fn Omega_h::LOs entity_ordinals
 *
 * \brief Return array with local entity identification numbers.
 *
 * \param [in] aMeshSets mesh entity sets
 * \param [in] aSetName  entity set name
 * \param [in] aThrow    boolean (default = true)
 * \return array with local entity identification numbers
**********************************************************************************/
template<Omega_h::SetType EntitySet>
inline Omega_h::LOs
entity_ordinals
(const Omega_h::MeshSets& aMeshSets,
 const std::string& aSetName,
 bool aThrow = true)
{
    auto& tEntitySets = aMeshSets[EntitySet];
    auto tEntitySetMapIterator = tEntitySets.find(aSetName);
    if( (tEntitySetMapIterator == tEntitySets.end()) && (aThrow) )
    {
        THROWERR(std::string("DID NOT FIND NODE SET WITH NAME '") + aSetName + "'. NODE SET '"
                 + aSetName + "' IS NOT DEFINED IN INPUT MESH FILE, I.E. INPUT EXODUS FILE");
    }
    auto tFaceLids = (tEntitySetMapIterator->second);
    return tFaceLids;
}
// function entity_ordinals

/******************************************************************************//**
 * \fn inline void is_entity_set_defined
 *
 * \brief Return true if entity set, e.g. node or side set, is defined.
 *
 * \param [in] aMeshSets list with all entity sets
 * \param [in] aSetName  entity set name
 * \return boolean
**********************************************************************************/
template<Omega_h::SetType EntitySet>
inline bool
is_entity_set_defined
(const Omega_h::MeshSets& aMeshSets,
 const std::string& aSetName)
{
    auto& tNodeSets = aMeshSets[EntitySet];
    auto tNodeSetMapItr = tNodeSets.find(aSetName);
    auto tIsNodeSetDefined = tNodeSetMapItr != tNodeSets.end() ? true : false;
    return tIsNodeSetDefined;
}
// function is_entity_set_defined

/******************************************************************************//**
 * \tparam EntitySet entity set type
 *
 * \fn inline Omega_h::LOs get_entity_ordinals
 *
 * \brief Return list of entity ordinals. If not defined, throw error to terminal.
 *
 * \param [in] aMeshSets list with all entity sets
 * \param [in] aSetName  entity set name
 * \param [in] aThrow    flag to enable throw mechanism (default = true)
 * \return list of entity ordinals for this entity set
**********************************************************************************/
template<Omega_h::SetType EntitySet>
inline Omega_h::LOs
get_entity_ordinals
(const Omega_h::MeshSets& aMeshSets,
 const std::string& aSetName,
 bool aThrow = true)
{

    if( Plato::omega_h::is_entity_set_defined<EntitySet>(aMeshSets, aSetName) )
    {
        auto tEntityFaceOrdinals = Plato::omega_h::entity_ordinals<EntitySet>(aMeshSets, aSetName);
        return tEntityFaceOrdinals;
    }
    else
    {
        THROWERR(std::string("Entity set, i.e. side or node set, with name '") + aSetName + "' is not defined.")
    }
}
// function get_entity_ordinals

/******************************************************************************//**
 * \fn inline Plato::OrdinalType get_num_entities
 *
 * \brief Return total number of entities in the mesh.
 *
 * \param [in] aEntityDim entity dimension (vertex=0, edge=1, face=2, or region=3)
 * \param [in] aMesh      computational mesh metadata
 *
 * \return total number of entities in the mesh
**********************************************************************************/
inline Plato::OrdinalType
get_num_entities
(const Omega_h::Int aEntityDim,
 const Omega_h::Mesh & aMesh)
{
    std::unordered_map<Omega_h::Int, Omega_h::LO> tMap =
        { {0, aMesh.nverts()}, {1, aMesh.nedges()}, {2, aMesh.nfaces()}, {3, aMesh.nelems()} };
    auto tItr = tMap.find(aEntityDim);
    if(tItr == tMap.end())
    {
        THROWERR(std::string("Entity with dimension id '") + aEntityDim + " is not supported. "
            + "Supported options are: Omega_h::VERT=0, Omega_h::EDGE=1, Omega_h::FACE=2, and Omega_h::REGION=3")
    }
    return tItr->second;
}
// function get_num_entities

/******************************************************************************//**
 * \tparam EntityDim entity dimension (e.g. vertex, edge, face, or region)
 * \tparam EntitySet entity set type (e.g. nodeset or sideset)
 *
 * \fn inline Omega_h::LOs find_entities_on_non_prescribed_boundary
 *
 * \brief Return list of entity ordinals on non-prescribed boundary. A prescribed
 *   boundary is defined as the boundary where user-defined Neumann and Dirichlet
 *   boundary conditions are applied.
 *
 * \param [in] aEntitySetNames list of prescribed entity set names
 * \param [in] aMesh           computational mesh metadata
 * \param [in] aMeshSets       list of mesh sets
 *
 * \return list of entity ordinals on non-prescribed boundary
**********************************************************************************/
template
<Omega_h::Int EntityDim,
 Omega_h::SetType EntitySet>
inline Omega_h::LOs
find_entities_on_non_prescribed_boundary
(const std::vector<std::string> & aEntitySetNames,
       Omega_h::Mesh            & aMesh,
       Omega_h::MeshSets        & aMeshSets)
{
    // returns all the boundary faces, excluding faces within the domain
    auto tEntitiesAreOnNonPrescribedBoundary = Omega_h::mark_by_class_dim(&aMesh, EntityDim, EntityDim);
    // loop over all the side sets to get non-prescribed boundary faces
    auto tNumEntities = Plato::omega_h::get_num_entities(EntityDim, aMesh);
    for(auto& tEntitySetName : aEntitySetNames)
    {
        // return entity ids on prescribed side set
        auto tEntitiesOnPrescribedBoundary = Plato::omega_h::get_entity_ordinals<EntitySet>(aMeshSets, tEntitySetName);
        // return boolean array (entity on prescribed side set=1, entity not on prescribed side set=0)
        auto tEntitiesAreOnPrescribedBoundary = Omega_h::mark_image(tEntitiesOnPrescribedBoundary, tNumEntities);
        // return boolean array with 1's for all entities not on prescribed side set and 0's otherwise
        auto tEntitiesAreNotOnPrescribedBoundary = Omega_h::invert_marks(tEntitiesAreOnPrescribedBoundary);
        // return boolean array (entity on the non-prescribed boundary=1, entity not on the non-prescribed boundary=0)
        tEntitiesAreOnNonPrescribedBoundary = Omega_h::land_each(tEntitiesAreOnNonPrescribedBoundary, tEntitiesAreNotOnPrescribedBoundary);
    }
    // return identification numbers of all the entities on the non-prescribed boundary
    auto tIDsOfEntitiesOnNonPrescribedBoundary = Omega_h::collect_marked(tEntitiesAreOnNonPrescribedBoundary);
    return tIDsOfEntitiesOnNonPrescribedBoundary;
}
// function find_entities_on_non_prescribed_boundary

/***************************************************************************//**
 * \tparam EntityDim Oemga_h entity dimension
 * \fn inline void read_fields
 * \brief Read field data from vtk file.
 *
 * \param [in] aMesh      mesh metadata
 * \param [in] aPath      path to vtk file
 * \param [in] aFieldTags map from field data tag to identifier
 *
 * \param [in/out] aVariables map holding simulation metadata
 ******************************************************************************/
template<Omega_h::LO EntityDim>
inline void
read_fields
(const Omega_h::Mesh& aMesh,
 const Omega_h::filesystem::path& aPath,
 const Plato::FieldTags& aFieldTags,
       Plato::Variables& aVariables)
{
    Omega_h::Mesh tReadMesh(aMesh.library());
    Omega_h::vtk::read_parallel(aPath, aMesh.library()->world(), &tReadMesh);
    auto tTags = aFieldTags.tags();
    for(auto& tTag : tTags)
    {
        auto tData = Plato::omega_h::read_metadata_from_mesh(tReadMesh, EntityDim, tTag);
        auto tFieldName = aFieldTags.id(tTag);
        aVariables.vector(tFieldName, tData);
    }
}
// function read_fields

/***************************************************************************//**
 * \brief Return face local identifiers/ordinals for each element on the requested
 * side set. Here, local is used in the context of domain decomposition.  Therefore,
 * the identifiers/ordinals are local to the subdomain.
 *
 * \param [in] aMeshSets    Omega_h side set database
 * \param [in] aSideSetName Exodus side set name
 *
 * \return face local ordinals
 *
*******************************************************************************/
inline Omega_h::LOs side_set_face_ordinals(const Omega_h::MeshSets& aMeshSets, const std::string& aSideSetName)
{
    auto& tSideSets = aMeshSets[Omega_h::SIDE_SET];
    auto tSideSetMapIterator = tSideSets.find(aSideSetName);
    if(tSideSetMapIterator == tSideSets.end())
    {
        std::ostringstream tMsg;
        tMsg << "COULD NOT FIND SIDE SET WITH NAME = '" << aSideSetName.c_str()
            << "'.  SIDE SET IS NOT DEFINED IN THE INPUT MESH FILE, I.E. EXODUS FILE.\n";
        THROWERR(tMsg.str());
    }
    auto tFaceLids = (tSideSetMapIterator->second);
    return tFaceLids;
}
// function side_set_face_ordinals

/***************************************************************************//**
 * \brief Write mesh to exodus file
 *
 * \param [in] aFilepath  exodus filepath
 * \param [in] aMesh      mesh database
 *
*******************************************************************************/
inline void write_exodus_file(const std::string& aFilepath, Omega_h::Mesh& aMesh)
{
    Omega_h::exodus::write(aFilepath, &aMesh);
}

/******************************************************************************//**
 * \brief Add output tag for element states, e.g. states defined at the elements.
 * \param [in] aMesh         mesh metadata
 * \param [in] aStateDataMap output data map
 * \param [in] aStepIndex    time step index
 **********************************************************************************/
inline void add_element_state_tags(Omega_h::Mesh& aMesh, const Plato::DataMap& aStateDataMap, Plato::OrdinalType aStepIndex)
{
    auto tDataMap = aStateDataMap.getState(aStepIndex);

    auto tNumElements = aMesh.nelems();
    {   // ScalarVectors
        //
        auto& tVars = tDataMap.scalarVectors;
        for(auto tVar=tVars.begin(); tVar!=tVars.end(); ++tVar)
        {
            auto& tElemStateName = tVar->first;
            auto& tElemStateData = tVar->second;
            if(tElemStateData.extent(0) == tNumElements)
            {
                Omega_h::Write<Omega_h::Real> tElemStateWrite(tNumElements, tElemStateName);
                Plato::copy_1Dview_to_write(tElemStateData, tElemStateWrite);
                aMesh.add_tag(aMesh.dim(), tElemStateName, /*numDataPerElement=*/1, Omega_h::Reals(tElemStateWrite));
            }
        }
    }
    {   // ScalarMultiVectors
        //
        auto& tVars = tDataMap.scalarMultiVectors;
        for(auto tVar=tVars.begin(); tVar!=tVars.end(); ++tVar)
        {
            auto& tElemStateName = tVar->first;
            auto& tElemStateData = tVar->second;
            if(tElemStateData.extent(0) == tNumElements)
            {
                auto tNumDataPerElement = tElemStateData.extent(1);
                auto tNumData = tNumElements * tNumDataPerElement;
                Omega_h::Write<Omega_h::Real> tElemStateWrite(tNumData, tElemStateName);
                Plato::copy_2Dview_to_write(tElemStateData, tElemStateWrite);
                aMesh.add_tag(aMesh.dim(), tElemStateName, tNumDataPerElement, Omega_h::Reals(tElemStateWrite));
            }
        }
    }
}
// function add_element_state_tags

/***************************************************************************//**
 * \brief Return the local identifiers/ordinals associated with this element face.
 * Here, local is used in the context of domain decomposition.  Therefore, the
 * identifiers/ordinals are local to the subdomain.  Return -100 if the element
 * ordinal is not found.  The Analyze Error Macros can be used since they rely on
 * C++ standard functions on the the host.
 *
 * \param [in] aCellOrdinal  element ordinal
 * \param [in] aFaceOrdinal  face ordinal
 * \param [in] aElem2FaceMap element-to-face map
 *
 * \return local identifiers/ordinals associated with this element face.
 *
*******************************************************************************/
template<Plato::OrdinalType SpaceDim>
DEVICE_TYPE inline Omega_h::LO get_face_ordinal
(const Plato::OrdinalType& aCellOrdinal,
 const Plato::OrdinalType& aFaceOrdinal,
 const Omega_h::LOs& aElem2FaceMap)
{
    Omega_h::LO tOut = -100;
    auto tNumFacesPerCell = SpaceDim + 1;
    for(Plato::OrdinalType tFace = 0; tFace < tNumFacesPerCell; tFace++)
    {
        if(aElem2FaceMap[aCellOrdinal*tNumFacesPerCell+tFace] == aFaceOrdinal)
        {
            return tFace;
        }
    }
    return (tOut);
}

/***************************************************************************//**
 * \brief Return local element/cell coordinates, i.e. coordinates for each node.
 *
 * \param [in] aCellOrdinal cell ordinal
 * \param [in] aCoords      mesh coordinates
 * \param [in] aCell2Verts  cell to local vertex id map
 *
 * \return node coordinates for a single element
 *
*******************************************************************************/
template<Plato::OrdinalType SpatialDim, Plato::OrdinalType NodesPerCell = SpatialDim + 1>
DEVICE_TYPE inline Omega_h::Few< Omega_h::Vector<SpatialDim>, NodesPerCell > local_element_coords
(const Plato::OrdinalType & aCellOrdinal, const Plato::NodeCoordinate<SpatialDim> & aCoords)
{
    Omega_h::Few<Omega_h::Vector<SpatialDim>, NodesPerCell> tCellCoords;
    for (Plato::OrdinalType tNode = 0; tNode < NodesPerCell; tNode++)
    {
        for (Plato::OrdinalType tDim = 0; tDim < SpatialDim; tDim++)
        {
            tCellCoords[tNode][tDim] = aCoords(aCellOrdinal, tNode, tDim);
        }
    }

    return tCellCoords;
}
// local_element_coords

/******************************************************************************//**
* \brief Normalized vector : 1-D specialization
*
* \param [in/out] aVector  Omega_H vector
*
* \return normalized vector
*
**********************************************************************************/
DEVICE_TYPE inline void normalize(Omega_h::Vector<1> & aVector) { return; }
// function normalize - 1D

/******************************************************************************//**
* \brief Normalized vector : 2-D specialization
*
* \param [in/out] aVector  Omega_H vector
*
* \return normalized vector
*
**********************************************************************************/
DEVICE_TYPE inline void normalize(Omega_h::Vector<2> & aVector)
{
    auto tMagnitude = sqrt(aVector[0]*aVector[0] + aVector[1]*aVector[1]);
    aVector[0] = aVector[0] / tMagnitude;
    aVector[1] = aVector[1] / tMagnitude;
}
// function normalize - 2D

/******************************************************************************//**
* \brief Normalized vector : 3-D specialization
*
* \param [in/out] aVector  Omega_H vector
*
* \return normalized vector
*
**********************************************************************************/
DEVICE_TYPE inline void normalize(Omega_h::Vector<3> & aVector)
{
    auto tMagnitude = sqrt( aVector[0]*aVector[0] + aVector[1]*aVector[1] + aVector[2]*aVector[2] );
    aVector[0] = aVector[0] / tMagnitude;
    aVector[1] = aVector[1] / tMagnitude;
    aVector[2] = aVector[2] / tMagnitude;
}
// function normalize - 3D

/******************************************************************************//**
* \brief Return unit normal vector : 1-D specialization
*
* \param [in] aCellOrdinal  cell ordinal, i.e. subdomain element ordinal
* \param [in] aFaceOrdinal  face ordinal, i.e. \f$ i\in\{0,n_{f}\} \f$, where \f$ n_f \f$ is the number of faces on the element.
* \param [in] aCoords       node coordinates
*
* \return unit normal vector
*
**********************************************************************************/
DEVICE_TYPE inline Omega_h::Vector<1> unit_normal_vector
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::OrdinalType & aFaceOrdinal,
 const Plato::NodeCoordinate<1> & aCoords)
{
    auto tCellPoints = Plato::omega_h::local_element_coords<1>(aCellOrdinal, aCoords);
    auto tNormalVec = Omega_h::get_side_vector(tCellPoints, aFaceOrdinal);
    Plato::omega_h::normalize(tNormalVec);
    return tNormalVec;
}
// function unit_normal_vector - 1D

/******************************************************************************//**
* \brief Return unit normal vector : 2-D specialization
*
* \param [in] aCellOrdinal  cell ordinal, i.e. subdomain element ordinal
* \param [in] aFaceOrdinal  face ordinal, i.e. \f$ i\in\{0,n_{f}\} \f$, where \f$ n_f \f$ is the number of faces on the element.
* \param [in] aCoords       node coordinates
*
* \return unit normal vector
*
**********************************************************************************/
DEVICE_TYPE inline Omega_h::Vector<2> unit_normal_vector
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::OrdinalType & aFaceOrdinal,
 const Plato::NodeCoordinate<2> & aCoords)
{
    auto tCellPoints = Plato::omega_h::local_element_coords<2>(aCellOrdinal, aCoords);
    auto tNormalVec = Omega_h::get_side_vector(tCellPoints, aFaceOrdinal);
    Plato::omega_h::normalize(tNormalVec);
    return tNormalVec;
}
// function unit_normal_vector - 2D

/******************************************************************************//**
* \brief Return unit normal vector : 3-D specialization
*
* \param [in] aCellOrdinal  cell ordinal, i.e. subdomain element ordinal
* \param [in] aFaceOrdinal  face ordinal, i.e. \f$ i\in\{0,n_{f}\} \f$, where \f$ n_f \f$ is the number of faces on the element.
* \param [in] aCoords       node coordinates
*
* \return unit normal vector
*
**********************************************************************************/
DEVICE_TYPE inline Omega_h::Vector<3> unit_normal_vector
(const Plato::OrdinalType & aCellOrdinal,
 const Plato::OrdinalType & aFaceOrdinal,
 const Plato::NodeCoordinate<3> & aCoords)
{
    auto tCellPoints = Plato::omega_h::local_element_coords<3>(aCellOrdinal, aCoords);
    auto tNormalVec = Omega_h::get_side_vector(tCellPoints, aFaceOrdinal);
    Plato::omega_h::normalize(tNormalVec);
    return tNormalVec;
}
// function unit_normal_vector - 3D

/******************************************************************************//**
* \brief Create an Omega_H write array
*
* \param [in] aName       arbitrary descriptive name
* \param [in] aEntryCount number of elements in return vector
*
* \return Omega_H write vector
*
**********************************************************************************/
template<class Type>
Omega_h::Write<Type>
create_omega_h_write_array
(std::string aName,
 Plato::OrdinalType aEntryCount)
{
  Kokkos::View<Type*, Plato::Layout, Kokkos::DefaultExecutionSpace::memory_space> view(aName, aEntryCount);
  return Omega_h::Write<Type>(view);
}
// function create_omega_h_write_array

/******************************************************************************//**
* \brief Output node field to visualization file - convenient for visualization
*
* \tparam SpaceDim       spatial dimension
* \tparam NumDofsPerNode number of degrees of freedom per node
*
* \param [in] aField      2D field array - X(Time, Dofs)
* \param [in] aFieldName  field name
* \param [in] aMesh       omega_h mesh database
* \param [in] aWriter     omega_h output/viz interface
*
**********************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 Plato::OrdinalType NumDofsPerNode>
inline void output_node_state_to_viz_file
(const Plato::ScalarMultiVector& aField,
 const std::string& aFieldName,
 Omega_h::Mesh & aMesh,
 Omega_h::vtk::Writer& aWriter)
{
    for(Plato::OrdinalType tSnapshot = 0; tSnapshot < aField.extent(0); tSnapshot++)
    {
        auto tSubView = Kokkos::subview(aField, tSnapshot, Kokkos::ALL());
        aMesh.add_tag(Omega_h::VERT, aFieldName.c_str(), NumDofsPerNode, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tSubView)));
        auto tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpaceDim);
        aWriter.write(/*time_index*/tSnapshot, /*current_time=*/(Plato::Scalar)tSnapshot, tTags);
    }
}
// function output_node_state_to_viz_file

/******************************************************************************//**
* \brief Output element field to visualization file - convenient for visualization
*
* \tparam SpaceDim      spatial dimension
*
* \param [in] aMesh     omega_h mesh database
* \param [in] aWriter   omega_h output/viz interface
* \param [in] aWriter   omega_h output/viz interface
*
**********************************************************************************/
template<Plato::OrdinalType SpaceDim>
inline void output_element_state_to_viz_file
(const Plato::OrdinalType& aNumSnapShots,
 const Plato::DataMap& aDataMap,
 Omega_h::Mesh & aMesh,
 Omega_h::vtk::Writer& aWriter)
{
    for(Plato::OrdinalType tSnapshot = 0; tSnapshot < aNumSnapShots; tSnapshot++)
    {
        Plato::omega_h::add_element_state_tags(aMesh, aDataMap, tSnapshot);
        Omega_h::TagSet tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, SpaceDim);
        aWriter.write(/*time_index*/tSnapshot, /*current_time=*/(Plato::Scalar)tSnapshot, tTags);
    }
}
// function output_node_field

}
// namespace omega_h

}
// namespace Plato

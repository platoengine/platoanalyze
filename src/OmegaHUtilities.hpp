/*
 * OmegaHUtilities.hpp
 *
 *  Created on: Mar 11, 2020
 */

#pragma once

#include <Omega_h_shape.hpp>

#include "PlatoTypes.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief 1-D specialization: Return local element/cell coordinates, i.e.
 * coordinates for each node.
 *
 * \param [in] aCellOrdinal cell ordinal
 * \param [in] aCoords      mesh coordinates
 * \param [in] aCell2Verts  cell to local vertex id map
 *
 * \return node coordinates for element with ordinal: aCellOrdinal
 *
*******************************************************************************/
inline Omega_h::Few< Omega_h::Vector<1>, 2> local_element_coords
(const Plato::OrdinalType & aCellOrdinal,
 const Omega_h::Reals & aCoords,
 const Omega_h::LOs & aCell2Verts)
{
    const Plato::OrdinalType tSpaceDim = 1;
    const Plato::OrdinalType tNodesPerCell = 2;
    Omega_h::Few<Omega_h::Vector<tSpaceDim>, tNodesPerCell> tCellPoints;

    for (Plato::OrdinalType jNode = 0; jNode < tNodesPerCell; jNode++)
    {
        const auto tVertexLocalID = aCell2Verts[aCellOrdinal * tNodesPerCell + jNode];
        for (Plato::OrdinalType tDim = 0; tDim < tSpaceDim; tDim++)
        {
            tCellPoints[jNode][tDim] = aCoords[tVertexLocalID * tSpaceDim + tDim];
        }
    }

    return tCellPoints;
}
// local_element_coords : 1-D

/***************************************************************************//**
 * \brief 2-D specialization: Return local element/cell coordinates, i.e.
 * coordinates for each node.
 *
 * \param [in] aCellOrdinal cell ordinal
 * \param [in] aCoords      mesh coordinates
 * \param [in] aCell2Verts  cell to local vertex id map
 *
 * \return node coordinates for element with ordinal: aCellOrdinal
 *
*******************************************************************************/
inline Omega_h::Few< Omega_h::Vector<2>, 3> local_element_coords
(const Plato::OrdinalType & aCellOrdinal,
 const Omega_h::Reals & aCoords,
 const Omega_h::LOs & aCell2Verts)
{
    const Plato::OrdinalType tSpaceDim = 2;
    const Plato::OrdinalType tNodesPerCell = 3;
    Omega_h::Few<Omega_h::Vector<tSpaceDim>, tNodesPerCell> tCellPoints;

    for (Plato::OrdinalType jNode = 0; jNode < tNodesPerCell; jNode++)
    {
        const auto tVertexLocalID = aCell2Verts[aCellOrdinal * tNodesPerCell + jNode];
        for (Plato::OrdinalType tDim = 0; tDim < tSpaceDim; tDim++)
        {
            tCellPoints[jNode][tDim] = aCoords[tVertexLocalID * tSpaceDim + tDim];
        }
    }

    return tCellPoints;
}
// local_element_coords : 2-D

/***************************************************************************//**
 * \brief 3-D specialization: Return local element/cell coordinates, i.e.
 * coordinates for each node.
 *
 * \param [in] aCellOrdinal cell ordinal
 * \param [in] aCoords      mesh coordinates
 * \param [in] aCell2Verts  cell to local vertex id map
 *
 * \return node coordinates for element with ordinal: aCellOrdinal
 *
*******************************************************************************/
inline Omega_h::Few< Omega_h::Vector<3>, 4> local_element_coords
(const Plato::OrdinalType & aCellOrdinal,
 const Omega_h::Reals & aCoords,
 const Omega_h::LOs & aCell2Verts)
{
    const Plato::OrdinalType tSpaceDim = 3;
    const Plato::OrdinalType tNodesPerCell = 4;
    Omega_h::Few<Omega_h::Vector<tSpaceDim>, tNodesPerCell> tCellPoints;

    for (Plato::OrdinalType jNode = 0; jNode < tNodesPerCell; jNode++)
    {
        const auto tVertexLocalID = aCell2Verts[aCellOrdinal * tNodesPerCell + jNode];
        for (Plato::OrdinalType tDim = 0; tDim < tSpaceDim; tDim++)
        {
            tCellPoints[jNode][tDim] = aCoords[tVertexLocalID * tSpaceDim + tDim];
        }
    }

    return tCellPoints;
}
// local_element_coords : 3-D

/***************************************************************************//**
 * \brief Return face local ordinals for each element on the requested side set
 *
 * \param [in] aMeshSets    Omega_h side set database
 * \param [in] aSideSetName Exodus side set name
 *
 * \return face local ordinals
 *
*******************************************************************************/
inline Omega_h::LOs get_face_local_ordinals(const Omega_h::MeshSets& aMeshSets, const std::string& aSideSetName)
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
// function get_face_local_ordinals

/******************************************************************************//**
* \brief Normalized vector : 1-D specialization
*
* \param [in/out] aVector  Omega_H vector
*
* \return normalized vector
*
**********************************************************************************/
inline void normalize(Omega_h::Vector<1> & aVector) { return; }
// function normalize - 1D

/******************************************************************************//**
* \brief Normalized vector : 2-D specialization
*
* \param [in/out] aVector  Omega_H vector
*
* \return normalized vector
*
**********************************************************************************/
inline void normalize(Omega_h::Vector<2> & aVector)
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
inline void normalize(Omega_h::Vector<3> & aVector)
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
* \param [in/out] aVector  Omega_H vector
*
* \return unit normal vector
*
**********************************************************************************/
inline Omega_h::Vector<1> unit_normal_vector
(const Omega_h::LO & aFaceOrdinal, const Omega_h::Few< Omega_h::Vector<1>, 2> & aCellPoints)
{
    auto tNormalVec = Omega_h::get_side_vector(aCellPoints, aFaceOrdinal);
    Plato::normalize(tNormalVec);
    return tNormalVec;
}
// function unit_normal_vector - 1D

/******************************************************************************//**
* \brief Return unit normal vector : 2-D specialization
*
* \param [in/out] aVector  Omega_H vector
*
* \return unit normal vector
*
**********************************************************************************/
inline Omega_h::Vector<2> unit_normal_vector
(const Omega_h::LO & aFaceOrdinal, const Omega_h::Few< Omega_h::Vector<2>, 3> & aCellPoints)
{
    auto tNormalVec = Omega_h::get_side_vector(aCellPoints, aFaceOrdinal);
    Plato::normalize(tNormalVec);
    return tNormalVec;
}
// function unit_normal_vector - 2D

/******************************************************************************//**
* \brief Return unit normal vector : 3-D specialization
*
* \param [in/out] aVector  Omega_H vector
*
* \return unit normal vector
*
**********************************************************************************/
inline Omega_h::Vector<3> unit_normal_vector
(const Omega_h::LO & aFaceOrdinal, const Omega_h::Few< Omega_h::Vector<3>, 4> & aCellPoints)
{
    auto tNormalVec = Omega_h::get_side_vector(aCellPoints, aFaceOrdinal);
    Plato::normalize(tNormalVec);
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
/******************************************************************************/
template<class T>
Omega_h::Write<T> create_omega_h_write_array(std::string aName, Plato::OrdinalType aEntryCount)
{
  Kokkos::View<T*, Kokkos::LayoutRight, Plato::MemSpace> view(aName, aEntryCount);
  return Omega_h::Write<T>(view);
}
// function create_omega_h_write_array

}
// namespace Plato

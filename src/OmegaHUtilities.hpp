/*
 * OmegaHUtilities.hpp
 *
 *  Created on: Mar 11, 2020
 */

#pragma once

#include <Omega_h_few.hpp>
#include <Omega_h_shape.hpp>

#include "PlatoTypes.hpp"
#include "alg/PlatoLambda.hpp"

namespace Plato
{

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
(const Plato::OrdinalType & aCellOrdinal,
 const Omega_h::Reals & aCoords,
 const Omega_h::LOs & aCell2Verts)
{
    Omega_h::Few<Omega_h::Vector<SpatialDim>, NodesPerCell> tCellCoords;
    for (Plato::OrdinalType jNode = 0; jNode < NodesPerCell; jNode++)
    {
        const auto tVertexLocalID = aCell2Verts[aCellOrdinal * NodesPerCell + jNode];
        for (Plato::OrdinalType tDim = 0; tDim < SpatialDim; tDim++)
        {
            tCellCoords[jNode][tDim] = aCoords[tVertexLocalID * SpatialDim + tDim];
        }
    }

    return tCellCoords;
}
// local_element_coords

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
    printf("NVEC[0] = %e, NVEC[1] = %e, NVEC[2] = %e\n", tNormalVec[0], tNormalVec[1], tNormalVec[2]);
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

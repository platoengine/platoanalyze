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

/******************************************************************************//**
* \brief Return normalized vector : 2-D specialization
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
* \brief Return normalized vector : 3-D specialization
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

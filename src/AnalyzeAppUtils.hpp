/*
 * AnalyzeAppUtils.hpp
 *
 *  Created on: Apr 11, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "Solutions.hpp"

namespace Plato
{

/******************************************************************************//**
 *
 * \brief Extract vector solution from input scalar vector.
 *
 * \param [in] aFrom scalar vector
 * \param [in] aDof    degree of freedom
 * \param [in] aStride locations in memory between beginnings of successive array elements
 * 
 * \return scalar vector
**********************************************************************************/
Plato::ScalarVector
get_vector_component
(Plato::ScalarVector aFrom,
 Plato::OrdinalType aDof,
 Plato::OrdinalType aStride);

/******************************************************************************//**
 *
 * \brief Parse parameter list inline and set value of parsed parameter
 *
 * \param [in] aParams parameter list
 * \param [in] aTarget parameter tag
 * \param [in] aValue  parameter value
**********************************************************************************/
void
parse_inline
(Teuchos::ParameterList& aParams,
 const std::string& aTarget,
 Plato::Scalar aValue);

/******************************************************************************//**
 *
 * \brief Break input string apart by 'aDelimiter'.
 *
 * \param [in] aInputString input string
 * \param [in] aDelimiter   delimiter
 * 
 * \return vector of strings: tTokens 
**********************************************************************************/
std::vector<std::string>
split
(const std::string& aInputString,
 const char aDelimiter);

/******************************************************************************//**
 *
 * \brief Return inner parameter list.
 *
 * \param [in] aParams parameter list
 * \param [in] aTokens tokens for parsing
 * 
 * \return parameter list
**********************************************************************************/
Teuchos::ParameterList&
get_inner_list
(Teuchos::ParameterList& aParams,
 std::vector<std::string>& aTokens);

/******************************************************************************//**
 *
 * \brief Set parameter value in parameter list.
 *
 * \param [in] aParams parameter list
 * \param [in] aTokens tokens for parsing
 * \param [in] aValue  parameter value
**********************************************************************************/
void
set_parameter_value
(Teuchos::ParameterList& aParams,
 std::vector<std::string> aTokens,
 Plato::Scalar aValue);

/******************************************************************************//**
 *
 * \brief Find solution supported tag 
 *
 * \param [in] aName     solution name
 * \param [in] aSolution solutions database
 * 
 * \return solution tag
**********************************************************************************/
std::string 
find_solution_tag
(const std::string & aName,
 const Plato::Solutions & aSolution);

/******************************************************************************//**
 *
 * \brief Extract state solution from Solutions database
 *
 * \param [in] aName     solution name
 * \param [in] aSolution solutions database
 * \param [in] aDof      degree of freedom
 * \param [in] aStride   locations in memory between beginnings of successive array elements
 * 
 * \return scalar vector of state solutions
**********************************************************************************/
Plato::ScalarVector
extract_solution
(const std::string        & aName,
 const Plato::Solutions   & aSolution,
 const Plato::OrdinalType & aDof,
 const Plato::OrdinalType & aStride);

/******************************************************************************//**
 * \fn read_num_time_steps_from_pvd_file
 * \brief Read number of time steps from .pvd file
 *
 * \param [in] aOutputDirectory output directory name
 * \param [in] aFindKeyword     keyword to find in line
 * 
 * \return number of time steps
**********************************************************************************/
size_t read_num_time_steps_from_pvd_file
(const std::string & aOutputDirectory,
 const std::string & aFindKeyword);
 
}
// namespace Plato

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

Plato::ScalarVector
getVectorComponent
(Plato::ScalarVector aFrom,
 Plato::OrdinalType aDof,
 Plato::OrdinalType aStride);

void
parseInline
(Teuchos::ParameterList& aParams,
 const std::string& aTarget,
 Plato::Scalar aValue);

std::vector<std::string>
split
(const std::string& aInputString,
 const char aDelimiter);

Teuchos::ParameterList&
getInnerList
(Teuchos::ParameterList& aParams,
 std::vector<std::string>& aTokens);

void
setParameterValue
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
 * \param [in] aStride   spatial dimensions
 * 
 * \return scalar vector of state solutions
**********************************************************************************/
Plato::ScalarVector
extract_solution
(const std::string        & aName,
 const Plato::Solutions   & aSolution,
 const Plato::OrdinalType & aDof,
 const Plato::OrdinalType & aStride);

}
// namespace Plato

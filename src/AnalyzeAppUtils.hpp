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
 * \brief Extract state solution from Solutions database
 *
 * \param [in] aSolution solutions database
 * \param [in] aDof      degree of freedom
 * \param [in] aStride   number of locations in memory between beginnings of successive array elements

 * \return scalar vector of state solutions
**********************************************************************************/
Plato::ScalarVector
extract_solution
(const Plato::Solutions & aSolution,
 Plato::OrdinalType aDof,
 Plato::OrdinalType aStride);

}
// namespace Plato

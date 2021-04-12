/*
 * AnalyzeAppUtils.cpp
 *
 *  Created on: Apr 11, 2021
 */

#include "AnalyzeAppUtils.hpp"

namespace Plato
{

Plato::ScalarVector
getVectorComponent
(Plato::ScalarVector aFrom,
 Plato::OrdinalType aDof,
 Plato::OrdinalType aStride)
{
    Plato::OrdinalType tNumLocalVals = aFrom.size() / aStride;
    Plato::ScalarVector tRetVal("vector component", tNumLocalVals);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumLocalVals), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tRetVal(aNodeOrdinal) = aFrom(aStride*aNodeOrdinal+aDof);
    }, "copy component from vector");
    return tRetVal;
}
// function getVectorComponent

void parseInline
(Teuchos::ParameterList& aParams,
 const std::string& aTarget,
 Plato::Scalar aValue)
{
    std::vector<std::string> tokens = split(aTarget, ':');
    Teuchos::ParameterList& innerList = getInnerList(aParams, tokens);
    Plato::setParameterValue(innerList, tokens, aValue);
}
// function parseInline

std::vector<std::string> split
(const std::string& aInputString,
 const char aDelimiter)
{
    // break aInputString apart by 'aDelimiter' below //
    // produces a vector of strings: tTokens   //
    std::vector<std::string> tTokens;
    {
        std::istringstream tStream(aInputString);
        std::string tToken;
        while(std::getline(tStream, tToken, aDelimiter))
        {
            tTokens.push_back(tToken);
        }
    }
    return tTokens;
}
// function split

Teuchos::ParameterList&
getInnerList
(Teuchos::ParameterList& aParams,
 std::vector<std::string>& aTokens)
{
    auto& token = aTokens[0];
    if(token.front() == '[' && token.back() == ']')
    {
        // listName = token with '[' and ']' removed.
        std::string listName = token.substr(1, token.size() - 2);
        aTokens.erase(aTokens.begin());
        return getInnerList(aParams.sublist(listName, /*must exist=*/true), aTokens);
    }
    else
    {
        return aParams;
    }
}
// function getInnerList

void setParameterValue
(Teuchos::ParameterList& aParams,
 std::vector<std::string> aTokens,
 Plato::Scalar aValue)
{
    // if '(int)' then
    auto& token = aTokens[0];
    auto p1 = token.find("(");
    auto p2 = token.find(")");
    if(p1 != std::string::npos && p2 != std::string::npos)
    {
        std::string vecName = token.substr(0, p1);
        auto vec = aParams.get<Teuchos::Array<Plato::Scalar>>(vecName);

        std::string strVecEntry = token.substr(p1 + 1, p2 - p1 - 1);
        int vecEntry = std::stoi(strVecEntry);
        vec[vecEntry] = aValue;

        aParams.set(vecName, vec);
    }
    else
    {
        aParams.set<Plato::Scalar>(token, aValue);
    }
}
// function setParameterValue

Plato::ScalarVector
extract_solution
(const Plato::Solutions & aSolution,
 Plato::OrdinalType aDof,
 Plato::OrdinalType aStride)
{
    if(aSolution.size() <= 1u)
    {
        THROWERR(std::string("Solutions database size is '") + std::to_string(aSolution.size()) + "'."
            + "Solution data extraction from Analyze to the Engine only supports one state solution.")
    }

    auto tTags = aSolution.tags();
    if(tTags.empty()) { THROWERR("Solutions tags array is empty.") }
    auto tState = aSolution.get(tTags[0]);
    const Plato::OrdinalType tTIME_STEP_INDEX = tState.extent(0)-1;
    if(tTIME_STEP_INDEX < 0) { THROWERR("Negative time step index. State solution is most likely empty.") }
    auto tStatesSubView = Kokkos::subview(tState, tTIME_STEP_INDEX, Kokkos::ALL());
    auto tDeviceData = Plato::getVectorComponent(tStatesSubView, aDof, aStride);
    return tDeviceData;
}
// function extract_solution

}
// namespace Plato

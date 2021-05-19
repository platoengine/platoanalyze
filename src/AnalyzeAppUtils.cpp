/*
 * AnalyzeAppUtils.cpp
 *
 *  Created on: Apr 11, 2021
 */

#include <fstream>

#include "PlatoUtilities.hpp"
#include "AnalyzeAppUtils.hpp"

namespace Plato
{

Plato::ScalarVector
get_vector_component
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
// function get_vector_component

void parse_inline
(Teuchos::ParameterList& aParams,
 const std::string& aTarget,
 Plato::Scalar aValue)
{
    std::vector<std::string> tokens = split(aTarget, ':');
    Teuchos::ParameterList& innerList = get_inner_list(aParams, tokens);
    Plato::set_parameter_value(innerList, tokens, aValue);
}
// function parse_inline

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
get_inner_list
(Teuchos::ParameterList& aParams,
 std::vector<std::string>& aTokens)
{
    auto& token = aTokens[0];
    if(token.front() == '[' && token.back() == ']')
    {
        // listName = token with '[' and ']' removed.
        std::string listName = token.substr(1, token.size() - 2);
        aTokens.erase(aTokens.begin());
        return get_inner_list(aParams.sublist(listName, /*must exist=*/true), aTokens);
    }
    else
    {
        return aParams;
    }
}
// function get_inner_list

void set_parameter_value
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
// function set_parameter_value

std::string 
find_solution_tag
(const std::string & aName,
 const Plato::Solutions & aSolution)
{
    std::unordered_map<std::string, std::string> tSupportedTags = 
        { {"solution", "state" }, {"solution x", "state"}, {"solution y", "state"}, {"solution z", "state"} };
    auto tLowerName = Plato::tolower(aName);
    auto tItr = tSupportedTags.find(tLowerName);
    if( tItr == tSupportedTags.end() )
    { THROWERR(std::string("Solution tag '") + tItr->first + "' is not a supported output key.") }
    return tItr->second;
}
// function find_solution_tag

Plato::ScalarVector
extract_solution
(const std::string        & aName,
 const Plato::Solutions   & aSolution,
 const Plato::OrdinalType & aDof,
 const Plato::OrdinalType & aStride)
 {
    if(aSolution.empty()) 
    { THROWERR("Plato::Solutions database is empty.") }
    auto tTag = Plato::find_solution_tag(aName, aSolution);
    auto tState = aSolution.get(tTag);
    const Plato::OrdinalType tTIME_STEP_INDEX = tState.extent(0)-1;
    if(tTIME_STEP_INDEX < 0) { THROWERR("Negative time step index. State solution is most likely empty.") }
    auto tStatesSubView = Kokkos::subview(tState, tTIME_STEP_INDEX, Kokkos::ALL());
    auto tDeviceData = Plato::get_vector_component(tStatesSubView, aDof, aStride);
    return tDeviceData;
}
// function extract_solution

size_t read_num_time_steps_from_pvd_file
(const std::string & aOutputDirectory,
 const std::string & aFindKeyword)
{
    size_t tNumTimeStep = 0;
    std::string tFileName = aOutputDirectory + "/steps.pvd";
    std::ifstream tFile(tFileName);
    if (tFile.is_open())
    {
        std::string tLine;
        while (std::getline(tFile, tLine))
        {
            auto tFound = tLine.find(aFindKeyword);
            if (tFound != std::string::npos)
            {
                tNumTimeStep++;
            }
        }
        tFile.close();
    }
    return tNumTimeStep;
}
// function read_num_time_steps_from_pvd_file

}
// namespace Plato

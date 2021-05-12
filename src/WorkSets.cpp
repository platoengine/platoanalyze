/*
 * WorkSets.cpp
 *
 *  Created on: Apr 5, 2021
 */

#include "WorkSets.hpp"
#include "PlatoUtilities.hpp"

namespace Plato
{

void WorkSets::set(const std::string & aName, const std::shared_ptr<Plato::MetaDataBase> & aData)
{
    auto tLowerKey = Plato::tolower(aName);
    mData[tLowerKey] = aData;
}

const std::shared_ptr<Plato::MetaDataBase> & WorkSets::get(const std::string & aName) const
{
    auto tLowerKey = Plato::tolower(aName);
    auto tItr = mData.find(tLowerKey);
    if(tItr != mData.end())
    {
        return tItr->second;
    }
    else
    {
        THROWERR(std::string("Did not find 'MetaData' with tag '") + aName + "'.")
    }
}

std::vector<std::string> WorkSets::tags() const
{       
    std::vector<std::string> tOutput;
    for(auto& tPair : mData)
    {
        tOutput.push_back(tPair.first);
    }
    return tOutput;
}

bool WorkSets::defined(const std::string & aTag) const
{
    auto tLowerKey = Plato::tolower(aTag);
    auto tItr = mData.find(tLowerKey);
    auto tFound = tItr != mData.end();
    if(tFound)
    { return true; }
    else
    { return false; }
}

}
// namespace Plato

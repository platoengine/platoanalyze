/*
 * Solutions.cpp
 *
 *  Created on: Apr 5, 2021
 */

#include "Solutions.hpp"
#include "PlatoUtilities.hpp"

namespace Plato
{

Solutions::Solutions(std::string aPhysics) :
    mPhysics(aPhysics)
{return;}

std::string Solutions::physics() const
{
    return (mPhysics);
}

Plato::OrdinalType Solutions::size() const
{
    return (mSolution.size());
}

std::vector<std::string> Solutions::tags() const
{
    std::vector<std::string> tTags;
    for(auto& tPair : mSolution)
    {
        tTags.push_back(tPair.first);
    }
    return tTags;
}

void Solutions::set(const std::string& aTag, const Plato::ScalarMultiVector& aData)
{
    auto tLowerTag = Plato::tolower(aTag);
    mSolution[tLowerTag] = aData;
}

Plato::ScalarMultiVector Solutions::get(const std::string& aTag) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tItr = mSolution.find(tLowerTag);
    if(tItr == mSolution.end())
    {
        THROWERR(std::string("Solution with tag '") + aTag + "' is not defined.")
    }
    return tItr->second;
}

void Solutions::print() const
{
    if(mSolution.empty())
    { return; }
    for(auto& tPair : mSolution)
    { Plato::print_array_2D(tPair.second, tPair.first); }
}

bool Solutions::defined(const std::string & aTag) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tItr = mSolution.find(tLowerTag);
    if(tItr == mSolution.end())
    {
        return false;
    }
    return true;
}

bool Solutions::empty() const
{
    return mSolution.empty();
}

}
// namespace Plato

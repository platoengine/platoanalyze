/*
 * Solutions.cpp
 *
 *  Created on: Apr 5, 2021
 */

#include "Solutions.hpp"
#include "PlatoUtilities.hpp"

namespace Plato
{

Solutions::Solutions
(std::string aPhysics,
 std::string aPDE,
 Plato::OrdinalType aNumSolutions
 ) :
    mPDE(aPDE),
    mPhysics(aPhysics),
    mSolution{(long unsigned int)aNumSolutions},
    mSolutionNameToNumDofsMap{(long unsigned int)aNumSolutions}
{return;}

std::string Solutions::pde() const
{
    return (mPDE);
}

std::string Solutions::physics() const
{
    return (mPhysics);
}

Plato::OrdinalType Solutions::size(Plato::OrdinalType aInd) const
{
    return (mSolution[aInd].size());
}

std::vector<std::string> Solutions::tags(Plato::OrdinalType aInd) const
{
    std::vector<std::string> tTags;
    for(auto& tPair : mSolution[aInd])
    {
        tTags.push_back(tPair.first);
    }
    return tTags;
}

void Solutions::set(const std::string& aTag, const Plato::ScalarMultiVector& aData, Plato::OrdinalType aInd)
{
    auto tLowerTag = Plato::tolower(aTag);
    mSolution[aInd][tLowerTag] = aData;
}

Plato::ScalarMultiVector Solutions::get(const std::string& aTag, Plato::OrdinalType aInd) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tItr = mSolution[aInd].find(tLowerTag);
    if(tItr == mSolution[aInd].end())
    {
        THROWERR(std::string("Solution with tag '") + aTag + "' is not defined.")
    }
    return tItr->second;
}

void Solutions::setNumDofs(const std::string& aTag, const Plato::OrdinalType& aNumDofs, Plato::OrdinalType aInd)
{
    auto tLowerTag = Plato::tolower(aTag);
    mSolutionNameToNumDofsMap[aInd][tLowerTag] = aNumDofs;
}

Plato::OrdinalType Solutions::getNumDofs(const std::string& aTag, Plato::OrdinalType aInd) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tItr = mSolutionNameToNumDofsMap[aInd].find(tLowerTag);
    if(tItr == mSolutionNameToNumDofsMap[aInd].end())
    {
        THROWERR(std::string("Solution NumDofs with tag '") + aTag + "' is not defined.")
    }
    return tItr->second;
}

Plato::OrdinalType Solutions::getNumTimeSteps(Plato::OrdinalType aInd) const
{
    if(this->empty())
    {
        THROWERR("Solution map is empty.")
    }
    auto tTags = this->tags();
    const std::string tTag = tTags[0];
    auto tItr = mSolution[aInd].find(tTag);
    auto tSolution = tItr->second;
    return tSolution.extent(0);
}

void Solutions::print(Plato::OrdinalType aInd) const
{
    if(mSolution[aInd].empty())
    { return; }
    for(auto& tPair : mSolution[aInd])
    { Plato::print_array_2D(tPair.second, tPair.first); }
}

bool Solutions::defined(const std::string & aTag, Plato::OrdinalType aInd) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tItr = mSolution[aInd].find(tLowerTag);
    if(tItr == mSolution[aInd].end())
    {
        return false;
    }
    return true;
}

bool Solutions::empty() const
{
    bool tIsEmpty(true);
    for (auto& tSolution : mSolution)
    {
        tIsEmpty &= tSolution.empty();
    }
}

}
// namespace Plato

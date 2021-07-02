/*
 * Variables.cpp
 *
 *  Created on: Apr 6, 2021
 */

#include "BLAS1.hpp"
#include "Variables.hpp"
#include "PlatoUtilities.hpp"

namespace Plato
{

Plato::Scalar Variables::scalar(const std::string& aTag) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tItr = mScalars.find(tLowerTag);
    if(tItr == mScalars.end())
    {
        THROWERR(std::string("Scalar with tag '") + aTag + "' is not defined in the variables map.")
    }
    return tItr->second;
}

void Variables::scalar(const std::string& aTag, const Plato::Scalar& aInput)
{
    auto tLowerTag = Plato::tolower(aTag);
    mScalars[tLowerTag] = aInput;
}

Plato::ScalarVector Variables::vector(const std::string& aTag) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tItr = mVectors.find(tLowerTag);
    if(tItr == mVectors.end())
    {
        THROWERR(std::string("Vector with tag '") + aTag + "' is not defined in the variables map.")
    }
    return tItr->second;
}

void Variables::vector(const std::string& aTag, const Plato::ScalarVector& aInput)
{
    auto tLowerTag = Plato::tolower(aTag);
    mVectors[tLowerTag] = aInput;
}

bool Variables::isVectorMapEmpty() const
{
    return mVectors.empty();
}

bool Variables::isScalarMapEmpty() const
{
    return mScalars.empty();
}

bool Variables::defined(const std::string & aTag) const
{
    auto tLowerTag = Plato::tolower(aTag);
    auto tScalarMapItr = mScalars.find(tLowerTag);
    auto tFoundScalarTag = tScalarMapItr != mScalars.end();
    auto tVectorMapItr = mVectors.find(tLowerTag);
    auto tFoundVectorTag = tVectorMapItr != mVectors.end();

    if(tFoundScalarTag || tFoundVectorTag)
    { return true; }
    else
    { return false; }
}

void Variables::print() const
{
    this->printScalarMap();
    this->printVectorMap();
}

void Variables::printVectorMap() const
{
    if(mVectors.empty())
    {
        return;
    }

    std::cout << "Print Vector Map\n";
    for(auto& tPair : mVectors)
    {
        std::cout << "name = " << tPair.first << ", norm = " << Plato::blas1::norm(tPair.second) << "\n" << std::flush;
    }
}

void Variables::printScalarMap() const
{
    if(mScalars.empty())
    {
        return;
    }

    std::cout << "Print Scalar Map\n";
    for(auto& tPair : mScalars)
    {
        std::cout << "name = " << tPair.first << ", value = " << tPair.second << "\n" << std::flush;
    }
}

}
// namespace Plato

namespace Plato
{

void FieldTags::set(const std::string& aTag, const std::string& aID)
{
    mFields[aTag] = aID;
}

std::vector<std::string> FieldTags::tags() const
{
    std::vector<std::string> tTags;
    for(auto& tPair : mFields)
    {
        tTags.push_back(tPair.first);
    }
    return tTags;
}

std::string FieldTags::id(const std::string& aTag) const
{
    auto tItr = mFields.find(aTag);
    if(tItr == mFields.end())
    {
        THROWERR(std::string("Field with tag '") + aTag + "' is not defined.")
    }
    return tItr->second;
}

}
// namespace Plato

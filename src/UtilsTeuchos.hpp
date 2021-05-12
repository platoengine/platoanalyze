/*
 * UtilsTeuchos.hpp
 *
 *  Created on: Apr 5, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \fn inline void is_positive_finite_number
 *
 * \brief Check if scalar number is positive, if negative, throw error.
 *
 * \param [in] aInput  scalar
 * \param [in] aTag   scalar tag
 *
 * \return parameter of type=Type
**********************************************************************************/
inline void
is_positive_finite_number
(const Plato::Scalar aInput,
 std::string aTag = "scalar")
{
    if(!std::isfinite(aInput))
    {
        THROWERR(std::string("Paramater '") + aTag + "' is set to a non-finite number")
    }

    if(aInput <= static_cast<Plato::Scalar>(0.0))
    {
        THROWERR(std::string("Expected a positive non-zero number and instead user-defined parameter '")
             + aTag + "' was set to '" + std::to_string(aInput) + "'.")
    }
}
// function is_positive_finite_number

namespace teuchos
{

/******************************************************************************//**
 * \fn void is_material_defined
 *
 * \brief Check if material is defined, if not, throw an error.
 *
 * \param [in] aMaterialName material sublist name
 * \param [in] aInputs       parameter list with input data information
**********************************************************************************/
inline void is_material_defined
(const std::string & aMaterialName,
 const Teuchos::ParameterList & aInputs)
{
    if(!aInputs.sublist("Material Models").isSublist(aMaterialName))
    {
        THROWERR(std::string("Material with tag '") + aMaterialName + "' is not defined in 'Material Models' block")
    }
}
// function is_material_defined

/******************************************************************************//**
 * \tparam Type array type
 *
 * \fn inline std::vector<Type> parse_array
 *
 * \brief Return array of type=Type parsed from input file.
 *
 * \param [in] aTag    input array tag
 * \param [in] aInputs input file metadata
 *
 * \return array of type=Type
**********************************************************************************/
template <typename Type>
inline std::vector<Type>
parse_array
(const std::string & aTag,
 const Teuchos::ParameterList & aInputs)
{
    if(!aInputs.isParameter(aTag))
    {
        std::vector<Type> tOutput;
        return tOutput;
    }
    auto tSideSets = aInputs.get< Teuchos::Array<Type> >(aTag);

    auto tLength = tSideSets.size();
    std::vector<Type> tOutput(tLength);
    for(auto & tName : tOutput)
    {
        auto tIndex = &tName - &tOutput[0];
        tOutput[tIndex] = tSideSets[tIndex];
    }
    return tOutput;
}
// function parse_array

/******************************************************************************//**
 * \tparam Type parameter type
 *
 * \fn inline Type parse_parameter
 *
 * \brief Return parameter of type=Type parsed from input file.
 *
 * \param [in] aTag    input array tag
 * \param [in] aBlock  XML sublist tag
 * \param [in] aInputs input file metadata
 *
 * \return parameter of type=Type
**********************************************************************************/
template <typename Type>
inline Type parse_parameter
(const std::string            & aTag,
 const std::string            & aBlock,
 const Teuchos::ParameterList & aInputs)
{
    if( !aInputs.isSublist(aBlock) )
    {
        THROWERR(std::string("Parameter Sublist '") + aBlock + "' within Paramater List '" + aInputs.name() + "' is not defined.")
    }
    auto tSublist = aInputs.sublist(aBlock);

    if( !tSublist.isParameter(aTag) )
    {
        THROWERR(std::string("Parameter with tag '") + aTag + "' is not defined in Parameter Sublist '" + aBlock + "'.")
    }
    auto tOutput = tSublist.get<Type>(aTag);
    return tOutput;
}
// function parse_parameter

/******************************************************************************//**
 * \tparam Type scalar type
 *
 * \fn inline Type parse_max_material_property
 *
 * \brief Return maximum material property value from the list of similar material
 *   properties defined in other material blocks.
 *
 * \param [in] aInput    input file metadata
 * \param [in] aBlock    material block
 * \param [in] aProperty material property
 * \param [in] aDomains  spatial domain metadata
 *
 * \return material property scalar value
**********************************************************************************/
template<typename Type>
inline Type parse_max_material_property
(Teuchos::ParameterList& aInputs,
 const std::string& aBlock,
 const std::string& aProperty,
 const std::vector<Plato::SpatialDomain>& aDomains)
{
    std::vector<Type> tProperties;
    for(auto& tDomain : aDomains)
    {
        auto tMaterialName = tDomain.getMaterialName();
        Plato::teuchos::is_material_defined(tMaterialName, aInputs);
        auto tMaterial = aInputs.sublist("Material Models").sublist(tMaterialName);
        if( tMaterial.isSublist(aBlock) )
        {
            auto tValue = Plato::teuchos::parse_parameter<Plato::Scalar>(aProperty, aBlock, tMaterial);
            Plato::is_positive_finite_number(tValue, aProperty);
            tProperties.push_back(tValue);
        }
    }

    Plato::Scalar tMax = std::numeric_limits<Plato::Scalar>::infinity();
    if(!tProperties.empty())
    {
        tMax = *std::max_element(tProperties.begin(), tProperties.end());
    }
    return tMax;
}
// function parse_max_material_property

}

}
// namespace Plato

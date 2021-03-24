#ifndef PLATO_PARSE_TOOLS
#define PLATO_PARSE_TOOLS

#include "AnalyzeMacros.hpp"
#include "PlatoTypes.hpp"
#include "Teuchos_ParameterList.hpp"

#include <sstream>
#include <string>

namespace Plato {

namespace ParseTools {

/**************************************************************************//**
 * \brief Get a parameter from a sublist if it exists, otherwise return the default.
 * \tparam T Type of the requested parameter.
 * \param [in] aInputParams The containing ParameterList
 * \param [in] aSubListName The name of the sublist within aInputParams
 * \param [in] aParamName The name of the desired parameter
 * \return The requested parameter value if it exists, otherwise the default
 *****************************************************************************/

template < typename T >
T getSubParam(
    Teuchos::ParameterList& aInputParams,
    const std::string aSubListName,
    const std::string aParamName,
    T aDefaultValue )
{
    if( aInputParams.isSublist(aSubListName) == true )
    {
        return aInputParams.sublist(aSubListName).get<T>(aParamName, aDefaultValue);
    }
    else
    {
        return aDefaultValue;
    }
}

/**************************************************************************//**
 * \brief Get a parameter if it exists, otherwise return the default.
 * \tparam T Type of the requested parameter.
 * \param [in] aInputParams The containing ParameterList
 * \param [in] aParamName The name of the desired parameter
 * \param [in] aDefaultValue The default value
 * \return The requested parameter value if it exists, otherwise the default
 *****************************************************************************/

template < typename T >
T getParam(
    Teuchos::ParameterList& aInputParams,
    const std::string aParamName,
    T aDefaultValue )
{
    if (aInputParams.isType<T>(aParamName))
    {
        return aInputParams.get<T>(aParamName);
    }
    else
    {
        return aDefaultValue;
    }
}

/**************************************************************************//**
 * \brief Get a parameter if it exists, otherwise throw an exception
 * \tparam T Type of the requested parameter.
 * \param [in] aInputParams The containing ParameterList
 * \param [in] aParamName The name of the desired parameter
 * \return The requested parameter value if it exists, otherwise throw
 *****************************************************************************/

template < typename T >
T getParam(
    const Teuchos::ParameterList& aInputParams,
    const std::string aParamName )
{
    if (aInputParams.isType<T>(aParamName))
    {
        return aInputParams.get<T>(aParamName);
    }
    else
    {
        std::stringstream sstream;
        sstream << "Missing required parameter " << aParamName << std::endl;
        THROWERR(sstream.str());
    }
}

/**************************************************************************//**
 * \brief Get an equation if it exists, otherwise throw an exception
 * \param [in] aInputParams The containing ParameterList
 * \param [in] equationName The name of the desired equation
 * \return The requested equation, otherwise throw
 *****************************************************************************/

std::string getEquationParam(const Teuchos::ParameterList& aInputParams,
                             const std::string equationName );

/**************************************************************************//**
 * \brief Get an equation if it exists, otherwise throw an exception
 * \param [in] aInputParams The containing ParameterList
 * \param [in] equationIndex The index of the desired equation in a Bingo File
 * \param [in] equationName The name of the desired equation
 * \return The requested equation, otherwise throw
 *****************************************************************************/

std::string getEquationParam(const Teuchos::ParameterList& aInputParams,
                             const Plato::OrdinalType equationIndex = -1,
                             const std::string equationName = std::string("Equation") );

} // namespace ParseTools

} // namespace Plato

#endif

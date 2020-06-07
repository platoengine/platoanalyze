#ifndef PLATO_PARSE_TOOLS
#define PLATO_PARSE_TOOLS

#include <Teuchos_ParameterList.hpp>

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
    else { return aDefaultValue; }
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
    Teuchos::ParameterList& aInputParams,
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

} // namespace ParseTools

} // namespace Plato

#endif

#include "CustomMaterial.hpp"

#include "ParseTools.hpp"

#include "Sacado.hpp"

namespace Plato
{

/******************************************************************************/
Plato::Scalar
CustomMaterial::GetCustomExpressionValue(
    const Teuchos::ParameterList& paramList,
          std::string equationName ) const
{
  // Only passed a string, add the default index.
  return GetCustomExpressionValue( paramList,
                                   (Plato::OrdinalType) -1, equationName );
}

/******************************************************************************/
Plato::Scalar
CustomMaterial::GetCustomExpressionValue(
    const Teuchos::ParameterList& paramList,
          Plato::OrdinalType equationIndex /* = -1 */,
          std::string equationName /* = std::string("Equation") */ ) const
{
  // Get the equation type directly from the XML.
  std::string equationType;
  if( paramList.isType<std::string>("EquationType") )
  {
    equationType = paramList.get<std::string>("EquationType");
  }
  else
  {
    equationType = "Scalar";
  }

  // Get the equation directly from the XML or a Bingo file.
  // Look for the custom equations directly in the Bingo file.
  std::string equationStr =
    ParseTools::getEquationParam( paramList, equationIndex, equationName);

  // The third argument is result sets the type.
  if( equationType == "Scalar" )
  {
    Plato::Scalar result;
    GetCustomExpressionValue( paramList, equationStr, result );

    return result;
  }
  else if( equationType == "FAD" )
  {
    Sacado::Fad::DFad<Plato::Scalar> result;
    GetCustomExpressionValue( paramList, equationStr, result );

    // Currently if a FAD is specified just return the value.
    return result.val();
  }
  else
  {
    THROWERR( "Unknown equation type specified: " + equationType );
  }
}

} // namespace Plato

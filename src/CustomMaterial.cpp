#include "CustomMaterial.hpp"

#include "AnalyzeMacros.hpp"
#include "ExpressionEvaluator.hpp"
#include "PlatoTypes.hpp"

#include <fstream>

namespace Plato
{

/******************************************************************************/
double
CustomMaterial::GetCustomExpressionValue(
    const Teuchos::ParameterList& paramList,
    const std::string name ) const
{
  // Only passed a string, add the default index.
  return GetCustomExpressionValue( paramList, -1, name );
}

/******************************************************************************/
double
CustomMaterial::GetCustomExpressionValue(
    const Teuchos::ParameterList& paramList,
    int equationIndex /* = -1 */,
    std::string equationName /* = std::string("Equation") */ ) const
{
  // Get the equation directly from the XML or a Bingo file.
  std::string equationStr;

  // Look for the custom equations directly in the XML. The equation
  // name can be customized as an argument. The default is "Equation".
  if( paramList.isType<std::string>(equationName) )
  {
    equationStr = paramList.get<std::string>(equationName);
  }
  // Look for the custom equations directly in the Bingo file.
  else if( paramList.isType<std::string>("BingoFile") )
  {
    std::string bingoFile = paramList.get<std::string>( "BingoFile" );

    // There can be multiple equations, the index can be set as an
    // argument. The initial default is -1 (unset).
    if( equationIndex < 0 )
    {
      // Look for the equation index in the XML file.
      if( paramList.isType<Plato::OrdinalType>("BingoEquation") )
      {
        equationIndex = paramList.get<Plato::OrdinalType>("BingoEquation");
      }
      // Otherwise use a default of zero which is the first equation.
      else
      {
        equationIndex = 0;
      }
    }

    // Open the Bingo file and find the equation(s). The Bingo file
    // should contain a header with "FITNESS COMPLEXITY EQUATION"
    // followed by the three, comma separated entries. The last entry
    // being the equation. There may be additional text before and
    // after the header and equations. An example:

    // FITNESS    COMPLEXITY    EQUATION
    // 0, 0, E/((1.0+v)(1.0-2.0*v))
    // 0.011125010320428712, 5, (X_1)(X_1) + (X_0)(X_0)
    // 0.6253582149736167, 3, (X_1)(X_0)
    // 1.0, 1, X_0

    // Open the Bingo file
    std::ifstream infile(bingoFile);

    if( infile.is_open() )
    {
      // Read the text file line by line.
      std::string line;

      while( std::getline(infile, line) )
      {
        // Skip empty lines.
        if( line.empty() )
        {
        }
        // Find the equation header.
        else if( line.find("FITNESS"   ) != std::string::npos &&
                 line.find("COMPLEXITY") != std::string::npos &&
                 line.find("EQUATION"  ) != std::string::npos )
        {
          line.clear();

          // Read the equation requested, default is the first.
          while( std::getline(infile, line) && equationIndex > 0)
            --equationIndex;

          if( line.empty() || equationIndex != 0 )
          {
            THROWERR( "Cannot find Bingo equation requested." );
          }

          // Find the last comma delimiter.
          size_t found = line.find_last_of( "," );

          if( found != std::string::npos )
          {
            equationStr = line.substr(found + 1);
          }
          else
          {
            THROWERR( "Malformed Bingo equation found :" + line);
          }
        }
        // Skip all other text.
        else
        {
        }
      }
    }
    else
    {
      THROWERR( "Cannot open Bingo file: " + bingoFile );
    }
  }

  if( equationStr.empty() )
  {
    THROWERR( "No custom equation found." );
  }

  // Create an expression evaluator and pass the parameter list so
  // variable values can be retrived.
  ExpressionEvaluator expEval( paramList );

  // Parse the equation. The expression tree is held internally.
  expEval.parse_expression( equationStr.c_str() );

  // If a valid equation, evaluate it,
  if( expEval.validate_expression() == 0 )
  {
    // std::cout << "________________________________" << std::endl
    //           << "expression : " << equationStr << std::endl;
    // std::cout << "________________________________" << std::endl;

    // expEval.print_expression( std::cout );

    double result = expEval.evaluate_expression();

    // std::cout << "________________________________" << std::endl
    //           << "result = " << result << std::endl;

    return result;
  }
  else
    return 0;
}

} // namespace Plato

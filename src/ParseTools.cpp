#include "ParseTools.hpp"

#include "AnalyzeMacros.hpp"

#include <fstream>

namespace Plato {

namespace ParseTools {

/**************************************************************************//**
 * \brief Get a parameter if it exists, otherwise throw an exception
 * \param [in] aInputParams The containing ParameterList
 * \param [in] equationName The name of the desired equation
 * \return The requested equation, otherwise throw
 *****************************************************************************/

std::string getEquationParam(const Teuchos::ParameterList& aInputParams,
			           std::string equationName )
{
  // Only passed a string, add the default index.
  return getEquationParam( aInputParams,
			   (Plato::OrdinalType) -1, equationName );
}

/**************************************************************************//**
 * \brief Get a parameter if it exists, otherwise throw an exception
 * \param [in] aInputParams The containing ParameterList
 * \param [in] equationIndex The index of the desired equation in a Bingo File
 * \param [in] equationName The name of the desired equation
 * \return The requested equation, otherwise throw
 *****************************************************************************/

std::string getEquationParam(const Teuchos::ParameterList& aInputParams,
			           Plato::OrdinalType equationIndex /* = -1 */,
			           std::string equationName /* = std::string("Equation") */ )
{
  // Get the equation directly from the XML or a Bingo file.
  // Look for the custom equations directly in the Bingo file.
  std::string equationStr;

  // Look for the custom equations directly in the XML. The equation
  // name can be customized as an argument. The default is "Equation".
  if( aInputParams.isType<std::string>(equationName) )
  {
    equationStr = aInputParams.get<std::string>(equationName);
  }
  // Look for the custom equations directly in the Bingo file.
  else if( aInputParams.isType<std::string>("BingoFile") )
  {
    std::string bingoFile = aInputParams.get<std::string>( "BingoFile" );

    // There can be multiple equations, the index can be set as an
    // argument. The initial default is -1 (unset).
    if( equationIndex < 0 )
    {
      // Look for the equation index in the XML file.
      if( aInputParams.isType<Plato::OrdinalType>("BingoEquation") )
      {
        equationIndex = aInputParams.get<Plato::OrdinalType>("BingoEquation");
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

  return equationStr;
}

} // namespace ParseTools

} // namespace Plato

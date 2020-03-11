#include "CustomMaterial.hpp"

#include "AnalyzeMacros.hpp"
#include "ExpressionEvaluator.hpp"
#include "PlatoTypes.hpp"

#include <fstream>

namespace Plato
{

/******************************************************************************/
double
CustomMaterial::GetCustomExpressionValue( const Teuchos::ParameterList& paramList)
{
  // Get the equation directly from the XML or a Bingo File.
  std::string equationStr;

  if( paramList.isType<std::string>("Equation") )
  {
    equationStr = paramList.get<std::string>( "Equation" );
  }
  else if( paramList.isType<std::string>("BingoFile") )
  {
    std::string bingoFile = paramList.get<std::string>( "BingoFile" );

    int bingoEquation;

    if( paramList.isType<Plato::OrdinalType>( "BingoEquation" ) )
    {
      bingoEquation = paramList.get<Plato::OrdinalType>( "BingoEquation" );
    }
    else
    {
      bingoEquation = 0;
    }

    // Open the Bingoe file and find the equation(s)
    std::ifstream infile(bingoFile);

    if( infile.is_open() )
    {
      // Read the text file line by line.
      std::string line;
      while( std::getline(infile, line) )
      {
        // Skip empty lines
        if( line.empty() )
        {
        }
        // Find the equation header
        else if( line.find("FITNESS"   ) != std::string::npos &&
                 line.find("COMPLEXITY") != std::string::npos &&
                 line.find("EQUATION"  ) != std::string::npos )
        {
          line.clear();

          // Read the equation requested, default is the first.
          while( std::getline(infile, line) && bingoEquation > 0)
            --bingoEquation;

          if( line.empty() || bingoEquation != 0 )
          {
            THROWERR( "Cannot find Bingo equation requested." );
          }

          // Find the last comma delimiter
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
        // Skip everything in bewteen
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

  expEval.parse_expression( equationStr.c_str() );

  if( expEval.validate_expression_tree() == 0 )
  {
    // std::cout << "________________________________" << std::endl
    //           << "expression : " << equationStr << std::endl;
    // std::cout << "________________________________" << std::endl;

    // expEval.print_expression_tree( std::cout );

    double result = expEval.evaluate_expression_tree();

    // std::cout << "________________________________" << std::endl
    //           << "result = " << result << std::endl;

    expEval.delete_expression_tree();

    return result;
  }
  else
    return 0;
}

} // namespace Plato

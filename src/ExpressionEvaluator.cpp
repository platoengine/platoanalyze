/*
  Original code provided by Rhyscitlema
  https://www.rhyscitlema.com/algorithms/expression-parsing-algorithm

  Modified to accept additional operations such log, pow, sqrt, abs
  etc. Also added the abilitiy to parse variables which are all
  currently defaulted to sqrt(2).
*/

#include "ExpressionEvaluator.hpp"

#include "AnalyzeMacros.hpp"
#include "PlatoTypes.hpp"

#include <cmath>
#include <sstream>
#include <vector>

namespace Plato
{

ExpressionEvaluator::ExpressionEvaluator( const Teuchos::ParameterList& paramList ) : mParamList( paramList )
{
}

ExpressionEvaluator::~ExpressionEvaluator()
{
  if( mTree )
    delete_expression_tree();
}

// ************************************************************************* //
// Print the current node's ID or its value/variable.
std::string
ExpressionEvaluator::printNodeID( const Node* node, const bool descriptor )
{
  std::stringstream os;

  if( descriptor )
  {
    switch( node->ID )
    {
      case NUMBER:             os << "number '";    break;
      case VARIABLE:           os << "variable '";  break;
      case OPEN_PARENTHESIS:   os << "open '";      break;
      case CLOSE_PARENTHESIS:  os << "close '";     break;
      case OPEN_ABS_BAR:       os << "open '";      break;
      case CLOSE_ABS_BAR:      os << "close '";     break;
      default:
        os << "'";
    }
  }

  // Print the current node
  switch( node->ID )
  {
    case POSITIVE:        os << "+ve";       break;
    case NEGATIVE:        os << "-ve";       break;

    case ADDITION:        os << "+";         break;
    case SUBTRACTION:     os << "-";         break;
    case MULTIPLICATION:  os << "*";         break;
    case DIVISION:        os << "/";         break;

    case EXPONENTIAL:     os << "exp";       break;
    case LOG:             os << "log";       break;
    case POWER:           os << "pow";       break;
    case SQRT:            os << "sqrt";      break;
    case FACTORIAL:       os << "!";         break;
    case ABS:             os << "abs";       break;

    case SIN:             os << "sin";       break;
    case COS:             os << "cos";       break;
    case TAN:             os << "tan";       break;

    case NUMBER:          os << node->number;    break;
    case VARIABLE:        os << node->variable;  break;

    case OPEN_PARENTHESIS:   os << "(";      break;
    case CLOSE_PARENTHESIS:  os << ")";      break;
    case OPEN_ABS_BAR:       os << "|";      break;
    case CLOSE_ABS_BAR:      os << "|";      break;

    default:
      os << "Error: Unknown id";
  }

  if( descriptor )
  {
    switch( node->ID )
    {
      case NUMBER:          os << "'";  break;
      case VARIABLE:        os << "'";  break;
      default:
        os << "'";
    }
  }

  return os.str();
}

// ************************************************************************* //
// Print the current node
void ExpressionEvaluator::printNode(       std::ostream &os,
                                     const Node* node,
                                     const int indent )
{
  if( node == nullptr )
    return;

  // Print right sub-tree
  printNode( os, node->right, indent+4 );

  // Print the indent spaces
  for(int i=0; i<indent; ++i)
    os << " ";

  // Print the current node
  os << printNodeID( node, false );

  // Print the indent spaces
  // for(int i=0; i<2*indent; ++i)
  //   os << " ";

  // Print the value of this node
  // os << " = " << evaluateNode( node )

  os << std::endl;

  // Print left sub-tree
  printNode( os, node->left, indent+4 );
}

// ************************************************************************* //
// Print the current tree - helper function.
void ExpressionEvaluator::print_expression_tree( std::ostream &os )
{
  printNode( os, mTree, 4 );
}

// ************************************************************************* //
// Delete the current node.
void ExpressionEvaluator::deleteNode( Node* node )
{
  if( node == nullptr )
    return;

  deleteNode( node->left  );
  deleteNode( node->right );

  delete node;
}

// ************************************************************************* //
// Delete the current tree - helper function.
void ExpressionEvaluator::delete_expression_tree()
{
  deleteNode( mTree );

  mTree = nullptr;
}

// ************************************************************************* //
// Validate the current node and it children.
ExpressionEvaluator::NodeID
ExpressionEvaluator::validateNode( const Node* node )
{
  if( node == nullptr )
    return EMPTY_NODE;

  // Validate the left side of the tree.
  NodeID left  = validateNode( node->left  );
  if( left != EMPTY_NODE )
    return left;

  // Validate the right side of the tree.
  NodeID right = validateNode( node->right );
  if( right != EMPTY_NODE )
    return right;

  std::stringstream errorMsg;
  NodeID nodeID;

  // Validate the current node.
  switch( node->ID )
  {
    // These nodes need a valid left and right side value
    case ADDITION:
    case SUBTRACTION:
    case MULTIPLICATION:
    case DIVISION:
    case POWER:
      if( node->left == nullptr )
      {
        errorMsg << "Invalid expression: No left side value for "
                  << printNodeID( node, true );
        nodeID = node->ID;
      }
      else if( node->right == nullptr )
      {
        errorMsg << "Invalid expression: No right side value for "
                  << printNodeID( node, true );
        nodeID = node->ID;
      }
      else
        nodeID = EMPTY_NODE;

      break;

    // These nodes need a valid right side value
    case POSITIVE:
    case NEGATIVE:

    case EXPONENTIAL:
    case LOG:
    case SQRT:
    case ABS:

    case SIN:
    case COS:
    case TAN:
      if( node->right == nullptr )
      {
        errorMsg << "Invalid expression: No value for "
                  << printNodeID( node, true );
        nodeID = node->ID;
      }
      else if( node->left != nullptr )
      {
        errorMsg << "Invalid expression: Found a left side value for "
                  << printNodeID( node, true );
        nodeID = node->ID;
      }
      else
        nodeID = EMPTY_NODE;

      break;

    // These nodes need a valid left side value
    case FACTORIAL:
      if( node->left == nullptr )
      {
        errorMsg << "Invalid expression: No value for "
                  << printNodeID( node, true );
        nodeID = node->ID;
      }
      else if( node->right != nullptr )
      {
        errorMsg << "Invalid expression: Found a right side value for "
                  << printNodeID( node, true );
        nodeID = node->ID;
      }
      else
        nodeID = EMPTY_NODE;

      break;

    // None of these node should ever be in the tree
    case OPEN_PARENTHESIS:
      errorMsg << "Invalid expression: Found an open parenthesis '(' without a "
                << "corresponding close parenthesis ')'";
      nodeID = node->ID;
      break;
    case CLOSE_PARENTHESIS:
      errorMsg << "Invalid expression: Found a close parenthesis ')' without a "
                << "corresponding open parenthesis '('";
      nodeID = node->ID;
      break;
    case OPEN_ABS_BAR:
      errorMsg << "Invalid expression: Found an open absolute value '|' without a "
                << "corresponding close absolute value '|'";
      nodeID = node->ID;
      break;
    case CLOSE_ABS_BAR:
      errorMsg << "Invalid expression: Found a absolute value '|' without a "
                << "corresponding open absolute value '|'";
      nodeID = node->ID;
      break;

    default:
      nodeID = EMPTY_NODE;
      break;
  }

  if( errorMsg.str().size() )
    THROWERR( errorMsg.str() );

  return nodeID;
}

// ************************************************************************* //
// Validate the tree - helper function.
int
ExpressionEvaluator::validate_expression_tree()
{
  // Return true if a non empty node is found.
  return (validateNode( mTree ) != ExpressionEvaluator::EMPTY_NODE);
}

// ************************************************************************* //
// Helper factorial function
double ExpressionEvaluator::factorial( double n )
{
  long ans = 1, m = (long) n;

  for( long i=1; i<=m; ++i )
    ans *= i;

  return (double) ans;
}

// ************************************************************************* //
// Evaluate the current node.
double ExpressionEvaluator::evaluateNode( const Node* node )
{
  if( node == nullptr )
    return 0;

  // Get the left and right side of the tree.
  double left  = evaluateNode( node->left  );
  double right = evaluateNode( node->right );

  if( node->ID == DIVISION && right == 0 )
  {
    PRINTERR("Warning: Divide by zero.");
    return 0;
  }

  switch( node->ID )
  {
    case POSITIVE:        return +right;
    case NEGATIVE:        return -right;

    case ADDITION:        return left + right;
    case SUBTRACTION:     return left - right;
    case MULTIPLICATION:  return left * right;
    case DIVISION:        return left / right;

    case EXPONENTIAL:     return exp(right);
    case LOG:             return log(right);
    case POWER:           return pow(left, right);
    case SQRT:            return sqrt(right);
    case FACTORIAL:       return factorial(left);
    case ABS:             return fabs(right);

    case SIN:             return sin(right);
    case COS:             return cos(right);
    case TAN:             return tan(right);

    case NUMBER:          return node->number;
    case VARIABLE:        return mParamList.get<Plato::Scalar>(node->variable);
    default:              return 0;
  }
}

// ************************************************************************* //
// Evaluate the tree - helper function.
double ExpressionEvaluator::evaluate_expression_tree()
{
  return evaluateNode( mTree );
}

// ************************************************************************* //
// Insert the node into the tree.
ExpressionEvaluator::Node*
ExpressionEvaluator::insert_node_item(       Node* current,
                                       const Node item,
                                       const NodeInfo info )
{
  std::stringstream errorMsg;

  // Step 4: climb up to the parent node
  if( info == NoInfo )
  {
    errorMsg << "Developer error: Can not add node, no node information.";

    current = nullptr;
  }
  else if( info == RightAssociative )
  {
    // For right-associative
    while( current && current->precedence > item.precedence )
      current = current->parent;
  }
  else if( info == LeftAssociative )
  {
    // For left-associative
    while( current && current->precedence >= item.precedence )
      current = current->parent;
  }
  else if( info == SkipClimbUp )
  {
    // For open parenthesis, open absolute value, and positive/negative.
  }

  if( info != NoInfo && current == nullptr )
  {
    errorMsg << "Developer error: Can not add node, the current node is null.";
  }

  else if( item.ID == CLOSE_PARENTHESIS )
  {
    if( current->parent != nullptr )
    {
      // Step 5.1: get the parent of the '(' node.
      Node* node = current->parent;
      // Remove the current from between the parent and current's child
      node->right = current->right;

      // Now make the current's child point to the current's parent.
      if( current->right != nullptr )
        current->right->parent = node;

      // Step 5.2: delete the '(' node.
      delete current;

      // Step 6: Set the 'current node' to be the parent node.
      current = node;
    }
    else
    {
      errorMsg << "Invalid expression: "
               << "Found a close parenthesis ')' without a "
               << "corresponding open parenthesis '('";

      current = nullptr;
    }
  }
  else if( item.ID == CLOSE_ABS_BAR )
  {
      // Change the open absolute value open bar '|' node to
      // an abs node.
      current->ID = ABS;

      // Step 5.1: get the parent of '(' node.
      Node* node = current->parent;

      // Step 6: Set the 'current node' to be the parent node.
      current = node;
  }
  else
  {
    // Step 5.1: create the new node
    Node* node = new Node;
    *node = item;

    node->right = nullptr;

    // Step 5.2: add the new node
    node->left = current->right;

    if( current->right != nullptr )
      current->right->parent = node;

    current->right = node;
    node->parent = current;

    // Step 6: Set the 'current node' to be the new node
    current = node;
  }

  if( errorMsg.str().size() )
    THROWERR( errorMsg.str() );

  return current;
}

// ************************************************************************* //
// Parse the expression.
void ExpressionEvaluator::parse_expression( const char* expression )
{
  if( expression == nullptr )
    return;

  if( mTree )
    delete_expression_tree();

  std::stringstream errorMsg;

  // Preserve the incoming expression.
  const char* expPtr = expression;

  // The absolute value can be specified via abs(X) or |X| for the
  // latter, |X| it is necessary to keep track of whether the first or
  // second instance of the bar '|' has been found.
  bool beginABS = false;

  // Initialise the tree with an empty node which will be deleted when
  // the parsing is finished.
  Node root;
  root.precedence = 0;
  root.ID         = EMPTY_NODE;
  root.parent     = nullptr;
  root.right      = nullptr;
  root.left       = nullptr;

  Node *current = &root, previous = root;

  // For tracking two parameter functions and open operands
  std::vector< int > twoParamFunctionLocations;  // Location in the expression
  std::vector< int > openAbsBarLocations;        // Location in the expression
  std::vector< int > openParenLocations;         // Location in the expression

  // Stacks for tracking functions and operands.
  std::vector< NodeID > twoParamFunctionIDs; // Two parameter functions; 'pow'
  std::vector< int >    numOpenParens;       // Number of open '(' on the stack
  std::vector< int >    numOpenAbsBars;      // Number of open '|' on the stack

  numOpenParens.push_back( 0 );  //  Number of open '(' for the root
  numOpenAbsBars.push_back( 0 ); //  Number of open '|' for the root

  // Parse the expression.
  while( true )
  {
    NodeInfo info = NoInfo;     // Set the info to the default value

    Node node;
    node.precedence = 0;
    node.ID         = EMPTY_NODE;
    node.parent     = nullptr;
    node.right      = nullptr;
    node.left       = nullptr;

    char c = *expPtr++; // get latest character of string

    // At the end of input string
    if( c == '\0' )
      break;

    // Ingnore spaces, tabs, returns, end of lines.
    else if( c == ' ' || c == '\t' || c == '\r' || c == '\n')
      continue;

    // Open parenthesis
    else if( c == '(' )
    {
      // Number of open parenthesises relative to an open absolute value bar.
      ++numOpenParens.back();

      // Location of this open parenthesis for a match close parenthesis.
      openParenLocations.push_back( strlen(expression) - strlen(expPtr) - 1 );

      // Number of open absolute value bars on this level.
      numOpenAbsBars.push_back( 0 );
      beginABS = false;

      node.ID = OPEN_PARENTHESIS;  node.precedence = 0; info = SkipClimbUp;
    }
    // Close parenthesis
    else if( c == ')' )
    {
      node.ID = CLOSE_PARENTHESIS; node.precedence = 0; info = RightAssociative;

      // When closing make sure there is a corresponding open parenthesis
      if( numOpenParens.back() == 0 )
      {
        errorMsg << "Invalid expression: "
                 << "Found a close parenthesis ')' without a "
                 << "corresponding open parenthesis '('";
        break;
      }
      // When closing make sure there is not an open absolute value bar
      else if( numOpenAbsBars.back() > 0 )
      {
        errorMsg << "Invalid expression: "
                 << "Found a close parenthesis ')' before "
                 << "a close absolute value bar '|'.";
        break;
      }
      else
      {
        // Number of open parenthesis relative to open absolute value bar.
        --numOpenParens.back();
        // No open absolute value bar.
        numOpenAbsBars.pop_back();
        // Matched an open parenthesis
        openParenLocations.pop_back();

        // Set for an open absolute value bar on previous level.
        beginABS = numOpenAbsBars.back();
      }
    }

    // Absolute value use the same delimiter. So use a flag to keep
    // track of it being the first or the second.
    else if( c == '|' )
    {
      // Open absolute value.
      if( beginABS == false )
      {
        beginABS = true;

        // Number of open absolute value bars relative to an open parenthesis.
        ++numOpenAbsBars.back();

        // Location of this open absolute value bar for a match close
        // absolute value bar.
        openAbsBarLocations.push_back( strlen(expression) - strlen(expPtr) - 1 );

        // Number of open parenthesises on this level.
        numOpenParens.push_back( 0 );

        // Create and ABS node.
        node.ID = OPEN_ABS_BAR;  node.precedence = 1;  info = SkipClimbUp;
      }
      // Close absolute value.
      else {
        beginABS = false;

        node.ID = CLOSE_ABS_BAR;  node.precedence = 1;  info = RightAssociative;

        // When closing make sure there is an open absolute value bar
        if( numOpenAbsBars.back() == 0 )
        {
          errorMsg << "Invalid expression: "
                   << "Found a close absolute value bar '|' without a "
                    << "corresponding open absolute value bar '|'";
          break;
        }
        // When closing make sure there is not an open parenthesis
        else if( numOpenParens.back() > 0 )
        {
          errorMsg << "Invalid expression: "
                   << "Found a closed absolute value bar '|' before "
                   << "a close parenthesis ')'.";
          break;
        }
        else
        {
          // Number of open absolute value bars relative to an open parenthesis.
          --numOpenAbsBars.back();
          // No open parenthesises.
          numOpenParens.pop_back();
          // Matched an open absolute value bar
          openAbsBarLocations.pop_back();
        }
      }
    }
    // Operands
    else if( c == '+' || c == '-' )
    {
      // Distinguish between a plus/mius sign vs making a result
      // positive/negative.
      if(   previous.ID == NUMBER
         || previous.ID == VARIABLE
         || previous.ID == FACTORIAL
         || previous.ID == CLOSE_PARENTHESIS
         || previous.ID == CLOSE_ABS_BAR )
      {
        node.ID = (c == '+' ? ADDITION : SUBTRACTION);
        node.precedence = 2;  info = LeftAssociative;
      }
      else {
        node.ID = (c == '+' ? POSITIVE : NEGATIVE);
        node.precedence = 5;  info = SkipClimbUp;
      }
    }

    else if( c == '*' ) {
      node.ID = MULTIPLICATION;  node.precedence = 3;  info = LeftAssociative;
    }
    else if( c == '/' ) {
      node.ID = DIVISION;        node.precedence = 3;  info = LeftAssociative;
    }
    else if( c == '^' ) {
      node.ID = POWER;           node.precedence = 4;  info = RightAssociative;
    }
    else if( c == '!' ) {
      node.ID = FACTORIAL;       node.precedence = 6;  info = LeftAssociative;
    }

    // Functions
    else if( 0 == memcmp(expPtr-1, "sin" , 3) ) {
      expPtr += 3-1;
      node.ID = SIN;  node.precedence = 7;  info = LeftAssociative;
    }
    else if( 0 == memcmp(expPtr-1, "cos" , 3) ) {
      expPtr += 3-1;
      node.ID = COS;  node.precedence = 7;  info = LeftAssociative;
    }
    else if( 0 == memcmp(expPtr-1, "tan" , 3) ) {
      expPtr += 3-1;
      node.ID = TAN;  node.precedence = 7;  info = LeftAssociative;
    }

    else if( 0 == memcmp(expPtr-1, "exp" , 3) ) {
      expPtr += 3-1;
      node.ID = EXPONENTIAL;  node.precedence = 7;  info = LeftAssociative;
    }
    else if( 0 == memcmp(expPtr-1, "log" , 3) ) {
      expPtr += 3-1;
      node.ID = LOG;  node.precedence = 7;  info = LeftAssociative;
    }
    else if( 0 == memcmp(expPtr-1, "sqrt", 3) ) {
      expPtr += 4-1;
      node.ID = SQRT;  node.precedence = 7;  info = LeftAssociative;
    }
    else if( 0 == memcmp(expPtr-1, "abs" , 3) ) {
      expPtr += 3-1;
      node.ID = ABS;  node.precedence = 7;  info = LeftAssociative;
    }

    // For two argument functions push the function on to the next ID
    // stack. But do not create a node. The node will be created when
    // the corresponding comma is found.
    else if( 0 == memcmp(expPtr-1, "pow" , 3) ) {
      expPtr += 3-1; twoParamFunctionIDs.push_back( POWER );
      twoParamFunctionLocations.push_back( strlen(expression) - strlen(expPtr) - 1 );

      continue;
    }

    // When a comma is found pop the last function from the stack and
    // make the corresponding node.
    else if( c == ',' ) {
      if( twoParamFunctionIDs.size() )
      {
        node.ID = twoParamFunctionIDs.back();  node.precedence = 5;  info = LeftAssociative;
        twoParamFunctionIDs.pop_back();
        twoParamFunctionLocations.pop_back();
      }
      else
      {
        errorMsg << "Invalid expression: "
                 << "Found a comma ',' without a corresponding operation.";
        break;
      }
    }
    // Fixed constant
    else if( 0 == memcmp(expPtr-1, "PI"  , 2) ) {
      expPtr += 2-1;
      node.ID = NUMBER;  node.precedence = 7;  info = LeftAssociative;
      node.number = M_PI;
    }

    // Number
    else if( '0' <= c && c <= '9' )
    {
      char number[128];
      int i = 0;

      while( true )
      {
        number[i++] = c;

        if( i+1 == sizeof(number) )
        {
          number[i] = '\0';
          errorMsg << "Invalid expression: "
                   << "The number '" << number << "' exceeds the "
                   << sizeof(number) << " digit limit.";
          break;
        }

        c = *expPtr;

        // Check the next character to see if it is a digit or dot.
        if( ('0' <= c && c <='9') || c == '.' )
        {
          expPtr++;
        }
        // Check for an exponenet.
        else if( c == 'e' )
        {
          expPtr++;

          // Store the character and immediately read the next character.
          number[i++] = c;

          if( i+1 == sizeof(number) )
          {
            number[i] = '\0';
            errorMsg << "Invalid expression: "
                     << "The number '" << number << "' exceeds the "
                     << sizeof(number) << " digit limit.";
            break;
          }

          c = *expPtr;

          // Check the next character to see if it is a digit or a plus
          // or minus sign.
          if( ('0' <= c && c <='9') || c == '+' || c == '-' )
            expPtr++;
          else
            break;
        }
        else
          break;
      }

      // Get the actual number from the string.
      number[i] = '\0';
      sscanf(number, "%lf", &node.number);
      node.ID = NUMBER;  node.precedence = 7;  info = LeftAssociative;
    }

    // Variable
    else if( ('a' <= c && c <= 'z') ||  // It must start with a lower
             ('A' <= c && c <= 'Z') ||  // or upper case letter
             c == '_' )                 // or an underscore.
    {
      char variable[128];
      int i = 0;

      while( true )
      {
        variable[i++] = c;

        if( i+1 == sizeof(variable) )
        {
          variable[i] = '\0';

          errorMsg << "Invalid expression: "
                   << "The variable '" << variable << "' exceeds the "
                   << sizeof(variable) << " character limit.";
          break;
        }

        c = *expPtr;

        // Variables may have letters, numbers, and/or underscores ONLY.
        if( ('a' <= c && c <= 'z') ||
            ('A' <= c && c <= 'Z') ||
            ('0' <= c && c <= '9') ||
            c == '_' )
          expPtr++;
        else
          break;
      }

      // Save the variable name.
      variable[i] = '\0';
      node.variable = variable;
      node.ID = VARIABLE;  node.precedence = 7;  info = LeftAssociative;
    }
    // Invalid character.
    else
    {
      errorMsg << "Invalid expression: "
               << "Invalid character '" << c << "'.";
      break;
    }

    // Special case for a close parenthesis followed by an open
    // parenthesis with no operand between. Which implies a multiply.
    if( node.ID == OPEN_PARENTHESIS && previous.ID == CLOSE_PARENTHESIS)
    {
      NodeInfo multInfo = LeftAssociative;

      Node multNode;
      multNode.precedence = 3;
      multNode.ID         = MULTIPLICATION;
      multNode.parent     = nullptr;
      multNode.right      = nullptr;
      multNode.left       = nullptr;

      // Add the implict multiplication node to the tree.
      current = insert_node_item( current, multNode, multInfo );

      // Prepare for the next iteration
      previous = multNode;
    }

    // Some error handling for numbers and variables to make sure
    // there is an operand in between.
    if( (node.ID    == OPEN_PARENTHESIS   || node.ID    == OPEN_ABS_BAR ||
         node.ID    == NUMBER             || node.ID    == VARIABLE) &&

        (previous.ID == CLOSE_PARENTHESIS || previous.ID == CLOSE_ABS_BAR ||
         previous.ID == FACTORIAL ||
         previous.ID == NUMBER            || previous.ID == VARIABLE) )
    {
      errorMsg << "Invalid expression: "
               << "Found a " << printNodeID( &node, true )
               << " after a " << printNodeID( &previous, true )
               << " without an operation in between.";
      break;
    }

    // Some error handling for positive and negative to make sure
    // there is a value in between.
    if( (node.ID     == POSITIVE || node.ID     == NEGATIVE ) &&
        (previous.ID == POSITIVE || previous.ID == NEGATIVE) )
    {
      node.ID     = node.ID     == POSITIVE ? ADDITION : SUBTRACTION;
      previous.ID = previous.ID == POSITIVE ? ADDITION : SUBTRACTION;

      errorMsg << "Invalid expression: Found a " << printNodeID( &node, true )
                << " after a " << printNodeID( &previous, true )
                << " without a value in between." << std::endl;
      break;
    }
    // Add the node to the tree.
    current = insert_node_item( current, node, info );

    if( current == nullptr )
    {
      errorMsg << "Error: After inserting new node "
                << "the current node is now null.";
      break;
    }

    // Prepare for the next iteration
    previous = node;
  }

  // Found an error so add the original expression and a pointer to
  // the error.
  if( !errorMsg.str().empty() )
  {
    // Report the error.
    errorMsg << std::endl;

    // Print the original expression.
    errorMsg << expression << std::endl;

    // Print a pointer to where the error was found in the expression.
    int indent = strlen(expression) - strlen(expPtr) - 1;

    for( int i=0; i<indent; ++i )
      errorMsg << " ";

    errorMsg << "^";
  }
  else
  {
    // Make sure an open parenthesis was not left open.
    while( openParenLocations.size() )
    {
      if( errorMsg.str().size() )
        errorMsg << std::endl;

      // Report the error.
      errorMsg << "Invalid expression: Found an open '(' "
               << "' without a corresponding  close ')'."
               << std::endl;

      // Print the original expression.
      errorMsg << expression << std::endl;

      // Print a pointer to where the error was found in the expression.
      while( openParenLocations.back()-- )
        errorMsg << " ";

      errorMsg << "^" << std::endl;

      openParenLocations.pop_back();
    }

    // Make sure an open absolute value bar was not left open.
    while( openAbsBarLocations.size() )
    {
      if( errorMsg.str().size() )
        errorMsg << std::endl;

      // Report the error.
      errorMsg << "Invalid expression: Found an open '|' "
               << "without a corresponding close '|'."
               << std::endl;

      // Print the original expression.
      errorMsg << expression << std::endl;

      // Print a pointer to where the error was found in the expression.
      while( openAbsBarLocations.back()-- )
        errorMsg << " ";

      errorMsg << "^" << std::endl;

      openAbsBarLocations.pop_back();
    }
  }

  // Make sure two value functions are not empty.
  while( twoParamFunctionIDs.size() )
  {
    if( errorMsg.str().size() )
      errorMsg << std::endl;

    Node tmp;
    tmp.ID = twoParamFunctionIDs.back();

    // Report the error.
    errorMsg << "Invalid expression: "
             << "Found a two parameter function '"
             << printNodeID( &tmp, true ) << "' with one or no parameters"
             << std::endl;

    // Print the original expression.
    errorMsg << expression << std::endl;

    // Print a pointer to where the error was found in the expression.
    int indent = twoParamFunctionLocations.back();

    for( int i=0; i<indent; ++i )
      errorMsg << " ";

    errorMsg << "^";

    twoParamFunctionIDs.pop_back();
    twoParamFunctionLocations.pop_back();
  }

  // Remove the initial open parenthesis '(' node as the root node.
  if( root.right )
    root.right->parent = nullptr;
  else
  {
    errorMsg << "Invalid expression: "
             << "Empty expression.";
  }

  // Error print out the meesage and delete the tree.
  if( errorMsg.str().size() )
  {
    // Delete the tree.
    deleteNode( root.right );
    root.right = nullptr;

    THROWERR( errorMsg.str() );
  }

  // Return the actual top level root node
  mTree = root.right;
}

} // namespace Plato

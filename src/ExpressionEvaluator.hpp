/*
  Original code provided by Rhyscitlema
  https://www.rhyscitlema.com/algorithms/expression-parsing-algorithm

  Modified to accept additional operations such log, pow, sqrt, abs
  etc. Also added the abilitiy to parse variables which are all
  currently defaulted to sqrt(2).
*/

#ifndef EXPRESSIONEVALUATOR_HPP
#define EXPRESSIONEVALUATOR_HPP

#include <Teuchos_ParameterList.hpp>

#include <iostream>
#include <string>

namespace Plato
{

// ************************************************************************* //
class ExpressionEvaluator
{
private:
// ************************************************************************* //
  enum NodeID
  {
    EMPTY_NODE,

    OPEN_PARENTHESIS,
    CLOSE_PARENTHESIS,

    OPEN_ABS_BAR,
    CLOSE_ABS_BAR,

    POSITIVE,
    NEGATIVE,

    ADDITION,
    SUBTRACTION,
    MULTIPLICATION,
    DIVISION,

    EXPONENTIAL,
    LOG,
    POWER,
    SQRT,
    FACTORIAL,

    ABS,

    SIN,
    COS,
    TAN,

    NUMBER,
    VARIABLE
  };

// ************************************************************************* //
  enum NodeInfo
  {
    NoInfo,
    SkipClimbUp,
    RightAssociative,
    LeftAssociative
  };

public:
// ************************************************************************* //
  typedef struct _Node {
    NodeID ID;

    int precedence;
    double number;
    std::string variable;

    struct _Node *left, *right, *parent;

  } Node;

  ExpressionEvaluator( const Teuchos::ParameterList& paramList );
  ~ExpressionEvaluator();

  void print_expression_tree( std::ostream &os );

  void     delete_expression_tree();
  double evaluate_expression_tree();
  int    validate_expression_tree();

  void parse_expression( const char* expression );

private:
  std::string printNodeID( const Node* node, const bool descriptor );
  void printNode( std::ostream &os, const Node* node, const int indent );

  void   deleteNode(         Node* node );
  double evaluateNode( const Node* node );
  NodeID validateNode( const Node* node );

  Node* insert_node_item( Node* current, const Node item, const NodeInfo info );

  double factorial( double n );

  Teuchos::ParameterList mParamList;

  Node* mTree {nullptr};
};

} // namespace Plato

#endif

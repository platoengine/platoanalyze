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

  void print_expression( std::ostream &os ) const;

  void     delete_expression();
  double evaluate_expression() const;
  int    validate_expression() const;

  void parse_expression( const char* expression );

private:
  std::string printNodeID( const Node* node, const bool descriptor ) const;
  void printNode( std::ostream &os, const Node* node, const int indent ) const;

  void   deleteNode(         Node* node ) const;
  double evaluateNode( const Node* node ) const;
  NodeID validateNode( const Node* node ) const;

  Node* insert_node_item(       Node* current,
                          const Node item,
                          const NodeInfo info ) const;

  double factorial( double n ) const;

  Teuchos::ParameterList mParamList;

  Node* mTree {nullptr};
};

} // namespace Plato

#endif

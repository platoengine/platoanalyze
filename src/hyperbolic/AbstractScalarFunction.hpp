/*
 * AbstractScalarFunction.hpp
 *
 *  Created on: Apr 6, 2021
 */

#pragma once

#include "WorkSets.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \class AbstractScalarFunction
 *
 * \brief Base pure virtual class for Plato scalar functions.
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class AbstractScalarFunction
{
private:
    using ResultT = typename EvaluationT::ResultScalarType; /*!< result FAD type */

public:
    AbstractScalarFunction(){}
    virtual ~AbstractScalarFunction(){}

    virtual std::string name() const = 0;
    virtual void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const = 0;
    virtual void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const = 0;
};
// class AbstractScalarFunction

}
// namespace Fluids

}
// namespace Plato

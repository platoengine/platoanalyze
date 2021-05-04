/*
 * AbstractVolumeIntegrand.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include "WorkSets.hpp"

namespace Plato
{

/***************************************************************************//**
 * \class AbstractVolumeIntegrand
 *
 * \tparam PhysicsT    physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \brief Abstract class used to defined interface for cell/element volume integrals.
 *
 ******************************************************************************/
template<typename PhysicsT, typename EvaluationT>
class AbstractVolumeIntegrand
{
private:
    using ResultT = typename EvaluationT::ResultScalarType; /*!< result FAD evaluation type */

public:
    virtual ~AbstractVolumeIntegrand(){}
    virtual void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarMultiVectorT<ResultT> & aResultWS) const = 0;
};
// class AbstractVolumeIntegrand

}
// namespace Plato

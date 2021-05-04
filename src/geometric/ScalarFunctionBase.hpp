#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Geometric
{

/******************************************************************************//**
 * \brief Scalar function base class
 **********************************************************************************/
class ScalarFunctionBase
{
public:
    virtual ~ScalarFunctionBase(){}

    /******************************************************************************//**
     * \brief Return function name
     * \return user defined function name
     **********************************************************************************/
    virtual std::string name() const = 0;

    /******************************************************************************//**
     * \brief Return function value
     * \param [in] aControl design variables
     * \return function value
     **********************************************************************************/
    virtual Plato::Scalar
    value(const Plato::ScalarVector & aControl) const = 0;

    /******************************************************************************//**
     * \brief Return function gradient wrt design variables
     * \param [in] aControl design variables
     * \return function gradient wrt design variables
     **********************************************************************************/
    virtual Plato::ScalarVector
    gradient_z(const Plato::ScalarVector & aControl) const = 0;

    /******************************************************************************//**
     * \brief Return function gradient wrt configurtion variables
     * \param [in] aControl design variables
     * \return function gradient wrt configurtion variables
     **********************************************************************************/
    virtual Plato::ScalarVector
    gradient_x(const Plato::ScalarVector & aControl) const = 0;

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    virtual void updateProblem(const Plato::ScalarVector & aControl) const = 0;


}; // class ScalarFunctionBase

} // namespace Geometric

} // namespace Plato

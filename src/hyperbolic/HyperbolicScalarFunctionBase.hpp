#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************//**
 * @brief Scalar function base class
 **********************************************************************************/
class ScalarFunctionBase
{
public:
    virtual ~ScalarFunctionBase(){}

    /******************************************************************************//**
     * @brief Return function name
     * @return user defined function name
     **********************************************************************************/
    virtual std::string name() const = 0;

    /******************************************************************************//**
     * @brief Return function value
     * @param [in] aState state variables
     * @param [in] aStateDot first time derivative of state variables
     * @param [in] aStateDotDot second time derivative state variables
     * @param [in] aControl design variables
     * @param [in] aTimeStep current time step
     * @return function value
     **********************************************************************************/
    virtual Plato::Scalar
    value(const Plato::ScalarMultiVector & aState,
          const Plato::ScalarMultiVector & aStateDot,
          const Plato::ScalarMultiVector & aStateDotDot,
          const Plato::ScalarVector & aControl,
          Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * @brief Return function gradient wrt design variables
     * @param [in] aState state variables
     * @param [in] aStateDot first time derivative of state variables
     * @param [in] aStateDotDot second time derivative state variables
     * @param [in] aControl design variables
     * @param [in] aTimeStep current time step
     * @return function gradient wrt design variables
     **********************************************************************************/
    virtual Plato::ScalarVector
    gradient_z(const Plato::ScalarMultiVector & aState,
               const Plato::ScalarMultiVector & aStateDot,
               const Plato::ScalarMultiVector & aStateDotDot,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * @brief Return function gradient wrt state variables
     * @param [in] aState state variables
     * @param [in] aStateDot first time derivative of state variables
     * @param [in] aStateDotDot second time derivative state variables
     * @param [in] aControl design variables
     * @param [in] aTimeStep current time step
     * @return function gradient wrt state variables
     **********************************************************************************/
    virtual Plato::ScalarVector
    gradient_u(const Plato::ScalarMultiVector & aState,
               const Plato::ScalarMultiVector & aStateDot,
               const Plato::ScalarMultiVector & aStateDotDot,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep,
               Plato::OrdinalType aStepIndex) const = 0;

    /******************************************************************************//**
     * @brief Return function gradient wrt state dot variables
     * @param [in] aState state variables
     * @param [in] aStateDot first time derivative of state variables
     * @param [in] aStateDotDot second time derivative state variables
     * @param [in] aControl design variables
     * @param [in] aTimeStep current time step
     * @return function gradient wrt state dot variables
     **********************************************************************************/
    virtual Plato::ScalarVector
    gradient_v(const Plato::ScalarMultiVector & aState,
               const Plato::ScalarMultiVector & aStateDot,
               const Plato::ScalarMultiVector & aStateDotDot,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep,
               Plato::OrdinalType aStepIndex) const = 0;

    /******************************************************************************//**
     * @brief Return function gradient wrt state dot dot variables
     * @param [in] aState state variables
     * @param [in] aStateDot first time derivative of state variables
     * @param [in] aStateDotDot second time derivative state variables
     * @param [in] aControl design variables
     * @param [in] aTimeStep current time step
     * @return function gradient wrt state dot dot variables
     **********************************************************************************/
    virtual Plato::ScalarVector
    gradient_a(const Plato::ScalarMultiVector & aState,
               const Plato::ScalarMultiVector & aStateDot,
               const Plato::ScalarMultiVector & aStateDotDot,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep,
               Plato::OrdinalType aStepIndex) const = 0;

    /******************************************************************************//**
     * @brief Return function gradient wrt configurtion variables
     * @param [in] aState state variables
     * @param [in] aStateDot first time derivative of state variables
     * @param [in] aStateDotDot second time derivative state variables
     * @param [in] aControl design variables
     * @param [in] aTimeStep current time step
     * @return function gradient wrt configurtion variables
     **********************************************************************************/
    virtual Plato::ScalarVector
    gradient_x(const Plato::ScalarMultiVector & aState,
               const Plato::ScalarMultiVector & aStateDot,
               const Plato::ScalarMultiVector & aStateDotDot,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const = 0;

}; // class ScalarFunctionBase

} // namespace Hyperbolic

} // namespace Plato

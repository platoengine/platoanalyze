#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Scalar function inc base class
 **********************************************************************************/
class ScalarFunctionIncBase
{
public:
    virtual ~ScalarFunctionIncBase(){}

    /******************************************************************************//**
     * \fn virtual std::string name() const
     * \brief Return function name
     * \return user defined function name
     **********************************************************************************/
    virtual std::string name() const = 0;

    /******************************************************************************//**
     * \fn virtual Plato::Scalar value(const Plato::ScalarMultiVector & aStates,
     *                                 const Plato::ScalarVector & aControl,
     *                                 Plato::Scalar aTimeStep = 0.0) const
     * \brief Return function value
     * \param [in] aState state variables
     * \param [in] aControl design variables
     * \param [in] aTimeStep current time step increment
     * \return function value
     **********************************************************************************/
    virtual Plato::Scalar value(const Plato::ScalarMultiVector & aStates,
                                const Plato::ScalarVector & aControl,
                                Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \fn virtual Plato::ScalarVector gradient_z(const Plato::ScalarMultiVector & aStates,
     *                                            const Plato::ScalarVector & aControl,
     *                                            Plato::Scalar aTimeStep = 0.0) const
     * \brief Return function gradient wrt design variables
     * \param [in] aState state variables
     * \param [in] aControl design variables
     * \param [in] aTimeStep current time step increment
     * \return function gradient wrt design variables
     **********************************************************************************/
    virtual Plato::ScalarVector gradient_z(const Plato::ScalarMultiVector & aStates,
                                           const Plato::ScalarVector & aControl,
                                           Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \fn virtual Plato::ScalarVector gradient_u(const Plato::ScalarMultiVector & aStates,
     *                                            const Plato::ScalarVector & aControl,
     *                                            Plato::Scalar aTimeStep,
     *                                            Plato::OrdinalType aStepIndex) const
     * \brief Return function gradient wrt state variables
     * \param [in] aState state variables
     * \param [in] aControl design variables
     * \param [in] aTimeStep current time step increment
     * \param [in] aStepIndex current time step index
     * \return function gradient wrt state variables
     **********************************************************************************/
    virtual Plato::ScalarVector gradient_u(const Plato::ScalarMultiVector & aStates,
                                           const Plato::ScalarVector & aControl,
                                           Plato::Scalar aTimeStep = 0.0,
                                           Plato::OrdinalType aStepIndex = 0) const = 0;

    /******************************************************************************//**
     * \fn virtual Plato::ScalarVector gradient_x(const Plato::ScalarMultiVector & aStates,
     *                                            const Plato::ScalarVector & aControl,
     *                                            Plato::Scalar aTimeStep = 0.0) const
     * \brief Return function gradient wrt configurtion variables
     * \param [in] aState state variables
     * \param [in] aControl design variables
     * \param [in] aTimeStep current time step increment
     * \return function gradient wrt configurtion variables
     **********************************************************************************/
    virtual Plato::ScalarVector gradient_x(const Plato::ScalarMultiVector & aStates,
                                           const Plato::ScalarVector & aControl,
                                           Plato::Scalar aTimeStep = 0.0) const = 0;
};
// class ScalarFunctionIncBase


}
// namespace Plato

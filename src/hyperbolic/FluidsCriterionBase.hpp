/*
 * FluidsCriterionBase.hpp
 *
 *  Created on: Apr 6, 2021
 */

#pragma once

#include "Variables.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \class CriterionBase
 *
 * This pure virtual class provides the template for a scalar functions of the form:
 *
 *    \f[ J = J(\phi, U^k, P^k, T^k, X) \f]
 *
 * Derived class are responsible for the evaluation of the function and its
 * corresponding derivatives with respect to control \f$\phi\f$, momentum
 * state \f$ U^k \f$, mass state \f$ P^k \f$, energy state \f$ T^k \f$, and
 * configuration \f$ X \f$ variables.
 ******************************************************************************/
class CriterionBase
{
public:
    virtual ~CriterionBase(){}

    /***************************************************************************//**
     * \fn std::string name
     * \brief Return scalar function name.
     * \return scalar function name
     ******************************************************************************/
    virtual std::string name() const = 0;

    /***************************************************************************//**
     * \fn Plato::Scalar value
     * \brief Return scalar function value.
     * \param [in] aControls control variables workset
     * \param [in] aPrimal   primal state database
     * \return scalar function value
     ******************************************************************************/
    virtual Plato::Scalar value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const = 0;

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientConfig
     * \brief Return scalar function derivative with respect to the configuration variables.
     *
     * \param [in] aControls control variables workset
     * \param [in] aPrimal   primal state database
     *
     * \return scalar function derivative with respect to the configuration variables
     ******************************************************************************/
    virtual Plato::ScalarVector gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const = 0;

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientControl
     * \brief Return scalar function derivative with respect to the control variables.
     *
     * \param [in] aControls control variables workset
     * \param [in] aPrimal   primal state database
     *
     * \return scalar function derivative with respect to the control variables
     ******************************************************************************/
    virtual Plato::ScalarVector gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const = 0;

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentPress
     * \brief Return scalar function derivative with respect to the current pressure.
     *
     * \param [in] aControls control variables workset
     * \param [in] aPrimal   primal state database
     *
     * \return scalar function derivative with respect to the current pressure
     ******************************************************************************/
    virtual Plato::ScalarVector gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const = 0;

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentTemp
     * \brief Return scalar function derivative with respect to the current temperature.
     *
     * \param [in] aControls control variables workset
     * \param [in] aPrimal   primal state database
     *
     * \return scalar function derivative with respect to the current temperature
     ******************************************************************************/
    virtual Plato::ScalarVector gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const = 0;

    /***************************************************************************//**
     * \fn Plato::ScalarVector gradientCurrentVel
     * \brief Return scalar function derivative with respect to the current velocity.
     *
     * \param [in] aControls control variables workset
     * \param [in] aPrimal   primal state database
     *
     * \return scalar function derivative with respect to the current velocity
     ******************************************************************************/
    virtual Plato::ScalarVector gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal & aPrimal) const = 0;
};
// class CriterionBase

}
// namespace Fluids

}
// namespace Plato

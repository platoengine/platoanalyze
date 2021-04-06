/*
 * LocalScalarFunctionInc.hpp
 *
 *  Created on: Mar 1, 2020
 */

#pragma once

#include "PlatoStaticsTypes.hpp"
#include "TimeData.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Abstract interface for the evaluation of path-dependent scalar functions
 * (e.g. criterion value and sensitivities).
*******************************************************************************/
class LocalScalarFunctionInc
{
public:
    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    virtual ~LocalScalarFunctionInc(){}

    /***************************************************************************//**
     * \brief Return function name
     * \return user defined function name
    *******************************************************************************/
    virtual std::string name() const = 0;

    /***************************************************************************//**
     * \brief Return scalar function value
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             current time data object
     * \return function value
    *******************************************************************************/
    virtual Plato::Scalar value(const Plato::ScalarVector & aCurrentGlobalState,
                                const Plato::ScalarVector & aPreviousGlobalState,
                                const Plato::ScalarVector & aCurrentLocalState,
                                const Plato::ScalarVector & aPreviousLocalState,
                                const Plato::ScalarVector & aControls,
                                const Plato::TimeData     & aTimeData) const = 0;

    /***************************************************************************//**
     * \brief Return workset with partial derivative with respect to design variables
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             current time data object
     * \return workset with partial derivative with respect to design variables
    *******************************************************************************/
    virtual Plato::ScalarMultiVector gradient_z(const Plato::ScalarVector & aCurrentGlobalState,
                                                const Plato::ScalarVector & aPreviousGlobalState,
                                                const Plato::ScalarVector & aCurrentLocalState,
                                                const Plato::ScalarVector & aPreviousLocalState,
                                                const Plato::ScalarVector & aControls,
                                                const Plato::TimeData     & aTimeData) const = 0;

    /***************************************************************************//**
     * \brief Return workset with partial derivative with respect to current global states
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             current time data object
     * \return workset with partial derivative with respect to current global states
    *******************************************************************************/
    virtual Plato::ScalarMultiVector gradient_u(const Plato::ScalarVector & aCurrentGlobalState,
                                                const Plato::ScalarVector & aPreviousGlobalState,
                                                const Plato::ScalarVector & aCurrentLocalState,
                                                const Plato::ScalarVector & aPreviousLocalState,
                                                const Plato::ScalarVector & aControls,
                                                const Plato::TimeData     & aTimeData) const = 0;

    /***************************************************************************//**
     * \brief Return workset with partial derivative with respect to previous global states
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             current time data object
     * \return workset with partial derivative with respect to previous global states
    *******************************************************************************/
    virtual Plato::ScalarMultiVector gradient_up(const Plato::ScalarVector & aCurrentGlobalState,
                                                 const Plato::ScalarVector & aPreviousGlobalState,
                                                 const Plato::ScalarVector & aCurrentLocalState,
                                                 const Plato::ScalarVector & aPreviousLocalState,
                                                 const Plato::ScalarVector & aControls,
                                                 const Plato::TimeData     & aTimeData) const = 0;

    /***************************************************************************//**
     * \brief Return workset partial derivative with respect to current local states
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             current time data object
     * \return workset with partial derivative with respect to current local states
    *******************************************************************************/
    virtual Plato::ScalarMultiVector gradient_c(const Plato::ScalarVector & aCurrentGlobalState,
                                                const Plato::ScalarVector & aPreviousGlobalState,
                                                const Plato::ScalarVector & aCurrentLocalState,
                                                const Plato::ScalarVector & aPreviousLocalState,
                                                const Plato::ScalarVector & aControls,
                                                const Plato::TimeData     & aTimeData) const = 0;

    /***************************************************************************//**
     * \brief Return workset with partial derivative with respect to previous local states
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             current time data object
     * \return workset with partial derivative with respect to previous local states
    *******************************************************************************/
    virtual Plato::ScalarMultiVector gradient_cp(const Plato::ScalarVector & aCurrentGlobalState,
                                                 const Plato::ScalarVector & aPreviousGlobalState,
                                                 const Plato::ScalarVector & aCurrentLocalState,
                                                 const Plato::ScalarVector & aPreviousLocalState,
                                                 const Plato::ScalarVector & aControls,
                                                 const Plato::TimeData     & aTimeData) const = 0;

    /***************************************************************************//**
     * \brief Return workset with partial derivative with respect to configurtion variables
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeData             current time data object
     * \return workset with partial derivative with respect to configurtion variables
    *******************************************************************************/
    virtual Plato::ScalarMultiVector gradient_x(const Plato::ScalarVector & aCurrentGlobalState,
                                                const Plato::ScalarVector & aPreviousGlobalState,
                                                const Plato::ScalarVector & aCurrentLocalState,
                                                const Plato::ScalarVector & aPreviousLocalState,
                                                const Plato::ScalarVector & aControls,
                                                const Plato::TimeData     & aTimeData) const = 0;

    /***************************************************************************//**
     * \brief Update physics-based parameters within a frequency of optimization iterations
     * \param [in] aGlobalStates global states for all time steps
     * \param [in] aLocalStates  local states for all time steps
     * \param [in] aControls     current controls, i.e. design variables
     * \param [in] aTimeData     current time data object
    *******************************************************************************/
    virtual void updateProblem(const Plato::ScalarMultiVector & aGlobalStates,
                               const Plato::ScalarMultiVector & aLocalStates,
                               const Plato::ScalarVector & aControls,
                               const Plato::TimeData     & aTimeData) const = 0;
};
// class LocalScalarFunctionInc

}
// namespace Plato

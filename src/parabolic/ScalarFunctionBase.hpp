#pragma once

#include "Solutions.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Parabolic
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
     * \param [in] aSolution state variables
     * \param [in] aControl design variables
     * \param [in] aTimeStep current time step
     * \return function value
     **********************************************************************************/
    virtual Plato::Scalar
    value(const Plato::Solutions    & aSolution,
          const Plato::ScalarVector & aControl,
                Plato::Scalar         aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \brief Return function gradient wrt design variables
     * \param [in] aSolution state variables
     * \param [in] aControl design variables
     * \param [in] aTimeStep current time step
     * \return function gradient wrt design variables
     **********************************************************************************/
    virtual Plato::ScalarVector
    gradient_z(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::Scalar         aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \brief Return function gradient wrt state variables
     * \param [in] aSolution state variables
     * \param [in] aControl design variables
     * \param [in] aStepIndex step index of state for which gradient is computed
     * \param [in] aTimeStep current time step
     * \return function gradient wrt state variables
     **********************************************************************************/
    virtual Plato::ScalarVector
    gradient_u(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::OrdinalType    aStepIndex,
                     Plato::Scalar         aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \brief Return function gradient wrt state dot variables
     * \param [in] aSolution state variables
     * \param [in] aControl design variables
     * \param [in] aStepIndex step index of state for which gradient is computed
     * \param [in] aTimeStep current time step
     * \return function gradient wrt state dot variables
     **********************************************************************************/
    virtual Plato::ScalarVector
    gradient_v(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::OrdinalType    aStepIndex,
                     Plato::Scalar         aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * \brief Return function gradient wrt configurtion variables
     * \param [in] aSolution state variables
     * \param [in] aControl design variables
     * \param [in] aTimeStep current time step
     * \return function gradient wrt configurtion variables
     **********************************************************************************/
    virtual Plato::ScalarVector
    gradient_x(const Plato::Solutions    & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::Scalar         aTimeStep = 0.0) const = 0;

}; // class ScalarFunctionBase

} // namespace Parabolic

} // namespace Plato

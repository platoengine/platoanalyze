/*
 * PlatoAbstractProblem.hpp
 *
 *  Created on: April 19, 2018
 */

#ifndef PLATOABSTRACTPROBLEM_HPP_
#define PLATOABSTRACTPROBLEM_HPP_

#include <Teuchos_RCPDecl.hpp>

#include "Solutions.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

struct partial
{
    enum derivative_t
    {
        CONTROL = 0, STATE = 1, CONFIGURATION = 2,
    };
};
// end struct partial

/******************************************************************************//**
 * \brief Abstract interface for a PLATO problem
**********************************************************************************/
class AbstractProblem
{
public:
    /******************************************************************************//**
     * \brief PLATO abstract problem destructor
    **********************************************************************************/
    virtual ~AbstractProblem()
    {
    }

    /******************************************************************************//**
     * \brief Output solution to visualization file.
     * \param [in] aFilename output file name
    **********************************************************************************/
    virtual void output(const std::string& aFilename) = 0;

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D container of control variables
     * \param [in] aSolution solution database
    **********************************************************************************/
    virtual void
    updateProblem(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution
    )=0;

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return 2D view of state variables
    **********************************************************************************/
    virtual Plato::Solutions
    solution(
        const Plato::ScalarVector & aControl
    )=0;

    /******************************************************************************//**
     * \brief Is criterion independent of the solution state?
     * \param [in] aName Name of criterion.
    **********************************************************************************/
    virtual bool
    criterionIsLinear(
        const std::string & aName
    ){ return false; }

    /******************************************************************************//**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return criterion function value
    **********************************************************************************/
    virtual Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    )=0;

    /******************************************************************************//**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return criterion function value
    **********************************************************************************/
    virtual Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    )=0;

    /******************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt control variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    )=0;

    /******************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt configuration variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    )=0;

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    )=0;

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt configuration variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    )=0;

    /******************************************************************************//**
     * \fn const Plato::DataMap getDataMap
     * \brief Return constant reference to Plato output database.
     * \return constant reference to Plato output database
     **********************************************************************************/
    Plato::DataMap mDataMap;
    decltype(mDataMap)& getDataMap()
    {
        return mDataMap;
    }

};
// end class AbstractProblem

}// end namespace Plato

#endif /* PLATOABSTRACTPROBLEM_HPP_ */

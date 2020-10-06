/*
 * PlatoAbstractProblem.hpp
 *
 *  Created on: April 19, 2018
 */

#ifndef PLATOABSTRACTPROBLEM_HPP_
#define PLATOABSTRACTPROBLEM_HPP_

#include <Teuchos_RCPDecl.hpp>

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
     * \brief Return 2D view of adjoint variables
     * \return 2D view of adjoint variables
    **********************************************************************************/
    virtual Plato::Adjoint getAdjoint()=0;

    /******************************************************************************//**
     * \brief Return global state variables
     * \return Plato::Solution comprised of globalstate variables
    **********************************************************************************/
    virtual Plato::Solution getGlobalSolution()=0;

    /******************************************************************************//**
     * \brief Set global state variables
     * \param [in] Plato::Solution comprised of global state variables
    **********************************************************************************/
    virtual void setGlobalSolution(const Plato::Solution & aGlobalSolution)=0;

    /******************************************************************************//**
     * \brief Return 2D view of local state variables
     * \return aLocalState 2D view of local state variables
    **********************************************************************************/
    virtual Plato::ScalarMultiVector getLocalState()
    {THROWERR("LOCAL STATES ARE NOT DEFINED FOR THIS APPLICATION");}

    /******************************************************************************//**
     * \brief Set local state variables
     * \param [in] aLocalState 2D view of local state variables
    **********************************************************************************/
    virtual void setLocalState(const Plato::ScalarMultiVector & aLocalState)
    {THROWERR("LOCAL STATES ARE NOT DEFINED FOR THIS APPLICATION");}

    /******************************************************************************//**
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    **********************************************************************************/
    virtual void
    applyConstraints(
        const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
        const Plato::ScalarVector                & aVector
    )=0;

    /******************************************************************************//**
     * \brief Apply boundary forces
     * \param [in/out] aForce 1D view of forces
    **********************************************************************************/
    virtual void
    applyBoundaryLoads(
        const Plato::ScalarVector & aForce
    )=0;

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D container of control variables
     * \param [in] aSolution Plato::Solution containing state
    **********************************************************************************/
    virtual void
    updateProblem(
        const Plato::ScalarVector & aControl,
        const Plato::Solution     & aSolution
    )=0;

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return 2D view of state variables
    **********************************************************************************/
    virtual Plato::Solution
    solution(
        const Plato::ScalarVector & aControl
    )=0;

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
     * \param [in] aSolution Plato::Solution containing state
     * \param [in] aName Name of criterion.
     * \return criterion function value
    **********************************************************************************/
    virtual Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControl,
        const Plato::Solution     & aSolution,
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
     * \param [in] aSolution Plato::Solution containing state
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const Plato::Solution     & aSolution,
        const std::string         & aName
    )=0;

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution Plato::Solution containing state
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt configuration variables
    **********************************************************************************/
    virtual Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solution     & aSolution,
        const std::string         & aName
    )=0;

    /******************************************************************************//**
     * \brief Return output database that enables import/export rights to PLATO Engine
     * \return PLATO Analyze output database
    **********************************************************************************/
    Plato::DataMap mDataMap;
    decltype(mDataMap)& getDataMap()
    {
        return mDataMap;
    }

    /******************************************************************************//**
     * \brief Return number of degrees of freedom in solution.
     * \return Number of degrees of freedom
    **********************************************************************************/
    virtual Plato::OrdinalType getNumSolutionDofs()=0;

};
// end class AbstractProblem

}// end namespace Plato

#endif /* PLATOABSTRACTPROBLEM_HPP_ */

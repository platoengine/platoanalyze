/*
 * AbstractProblem.hpp
 *
 *  Created on: Apr 10, 2021
 */

#pragma once

#include "Solutions.hpp"

namespace Plato
{

namespace Fluids
{

/******************************************************************************//**
 * \class AbstractProblem
 *
 * \brief This pure virtual class provides blueprint for any derived class.
 *   Derived classes define the main interface used to solve a Plato problem.
 *
 **********************************************************************************/
class AbstractProblem
{
public:
    virtual ~AbstractProblem() {}

    /******************************************************************************//**
     * \fn void output
     *
     * \brief Output interface to permit output of quantities of interests to a visualization file.
     *
     * \param [in] aFilePath visualization file path
     *
     **********************************************************************************/
    virtual void output(std::string aFilePath) = 0;

    /******************************************************************************//**
     * \fn const Plato::DataMap& getDataMap
     *
     * \brief Return a constant reference to the Plato output database.
     * \return constant reference to the Plato output database
     *
     **********************************************************************************/
    virtual const Plato::DataMap& getDataMap() const = 0;

    /******************************************************************************//**
     * \fn Plato::Solutions solution
     *
     * \brief Solve finite element simulation.
     * \param [in] aControl vector of design/optimization variables
     * \return Plato database with state solutions
     *
     **********************************************************************************/
    virtual Plato::Solutions solution(const Plato::ScalarVector& aControl) = 0;

    /******************************************************************************//**
     * \fn Plato::Scalar criterionValue
     *
     * \brief Evaluate criterion.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion evaluation
     *
     **********************************************************************************/
    virtual Plato::Scalar criterionValue(const Plato::ScalarVector & aControl, const std::string& aName) = 0;

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradient
     *
     * \brief Evaluate criterion gradient with respect to design/optimization variables.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion gradient with respect to design/optimization variables
     *
     **********************************************************************************/
    virtual Plato::ScalarVector criterionGradient(const Plato::ScalarVector &aControl, const std::string &aName) = 0;

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradientX
     *
     * \brief Evaluate criterion gradient with respect to configuration variables.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion gradient with respect to configuration variables
     *
     **********************************************************************************/
    virtual Plato::ScalarVector criterionGradientX(const Plato::ScalarVector &aControl, const std::string &aName) = 0;
};
// class AbstractProblem

}
// namespace Fluids

}
// namespace Plato

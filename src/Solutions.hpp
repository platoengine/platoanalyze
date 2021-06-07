/*
 * Solutions.hpp
 *
 *  Created on: Apr 5, 2021
 */

#pragma once

#include <vector>
#include <string>
#include <unordered_map>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/***************************************************************************//**
 * \struct Solutions
 *  \brief Holds POD state solutions
 ******************************************************************************/
struct Solutions
{
private:
    std::string mPDE; /*!< partial differential equation constraint */
    std::string mPhysics; /*!< physics to be analyzed/simulated */
    std::vector<std::unordered_map<std::string, Plato::ScalarMultiVector>> mSolution; /*!< map from state solution name to 2D POD array */
    std::vector<std::unordered_map<std::string, Plato::OrdinalType>> mSolutionNameToNumDofsMap; /*!< map from state solution name to number of dofs */

public:
    Plato::DataMap DataMap;

    /***************************************************************************//**
     * \fn Solutions
     *
     * \brief Constructor.
     * \param [in] aPhysics physics to be analyzed/simulated
     * \param [in] aPDE     partial differential equation constraint type
     ******************************************************************************/
    explicit Solutions(std::string aPhysics = "undefined", std::string aPDE = "undefined", Plato::OrdinalType aNumSolutions=1);

    /***************************************************************************//**
     * \fn std::string pde
     *
     * \brief Return partial differential equation (pde) constraint type.
     * \return pde (string)
     ******************************************************************************/
    std::string pde() const;

    /***************************************************************************//**
     * \fn std::string physics
     *
     * \brief Return analyzed/simulated physics.
     * \return physics (string)
     ******************************************************************************/
    std::string physics() const;

    /***************************************************************************//**
     * \fn Plato::OrdinalType size
     *
     * \brief Return number of elements in solution map.
     * \return number of elements in solution map (integer)
     ******************************************************************************/
    Plato::OrdinalType size(Plato::OrdinalType aInd=0) const;

    /***************************************************************************//**
     * \fn std::vector<std::string> tags
     *
     * \brief Return list with state solution tags.
     * \return list with state solution tags
     ******************************************************************************/
    std::vector<std::string> tags(Plato::OrdinalType aInd=0) const;

    /***************************************************************************//**
     * \fn void set
     *
     * \brief Set value of an element in the solution map.
     * \param aInd  data index
     * \param aTag  data tag
     * \param aData 2D POD array
     ******************************************************************************/
    void set(const std::string& aTag, const Plato::ScalarMultiVector& aData, Plato::OrdinalType aInd=0);

    /***************************************************************************//**
     * \fn Plato::ScalarMultiVector get
     *
     * \brief Return 2D POD array.
     * \param aInd  data index
     * \param aTag data tag
     ******************************************************************************/
    Plato::ScalarMultiVector get(const std::string& aTag, Plato::OrdinalType aInd=0) const;

    /***************************************************************************//**
     * \fn void set number of degrees of freedom (dofs) per node in map
     *
     * \brief Set value of an element in the solution-to-numdofs map.
     * \param aTag  data tag
     * \param aNumDofs number of dofs
     ******************************************************************************/
    void setNumDofs(const std::string& aTag, const Plato::OrdinalType& aNumDofs, Plato::OrdinalType aInd=0);

    /***************************************************************************//**
     * \fn Plato::OrdinalType get the number of degrees of freedom (dofs)
     *
     * \brief Return the number of dofs
     * \param aTag data tag
     ******************************************************************************/
    Plato::OrdinalType getNumDofs(const std::string& aTag, Plato::OrdinalType aInd=0) const;

    /***************************************************************************//**
     * \fn Plato::OrdinalType get the number of time steps
     *
     * \brief Return the number of time steps
     * \param aTag data tag
     ******************************************************************************/
    Plato::OrdinalType getNumTimeSteps(Plato::OrdinalType aInd=0) const;

    /***************************************************************************//**
     * \fn void print
     *
     * \brief Print solutions metadata.
     ******************************************************************************/
    void print(Plato::OrdinalType aInd=0) const;

    /***************************************************************************//**
     * \fn bool defined
     *
     * \brief Check if solution with input tag is defined in the database
     * \param [in] aTag solution tag/identifier
     * \return boolean (true = is defined; false = not defined) 
     ******************************************************************************/
    bool defined(const std::string & aTag, Plato::OrdinalType aInd=0) const;

    /***************************************************************************//**
     * \fn bool empty
     *
     * \brief Check if the solution database is empty
     * \return boolean (true = is empty; false = is not empty) 
     ******************************************************************************/
    bool empty() const;
};
// struct Solutions

}
// namespace Plato

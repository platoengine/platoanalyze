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
    std::unordered_map<std::string, Plato::ScalarMultiVector> mSolution; /*!< map from state solution name to 2D POD array */

public:
    /***************************************************************************//**
     * \fn Solutions
     *
     * \brief Constructor.
     * \param [in] aPhysics physics to be analyzed/simulated
     * \param [in] aPDE     partial differential equation constraint type
     ******************************************************************************/
    explicit Solutions(std::string aPhysics = "undefined", std::string aPDE = "undefined");

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
    Plato::OrdinalType size() const;

    /***************************************************************************//**
     * \fn std::vector<std::string> tags
     *
     * \brief Return list with state solution tags.
     * \return list with state solution tags
     ******************************************************************************/
    std::vector<std::string> tags() const;

    /***************************************************************************//**
     * \fn void set
     *
     * \brief Set value of an element in the solution map.
     * \param aTag  data tag
     * \param aData 2D POD array
     ******************************************************************************/
    void set(const std::string& aTag, const Plato::ScalarMultiVector& aData);

    /***************************************************************************//**
     * \fn Plato::ScalarMultiVector get
     *
     * \brief Return 2D POD array.
     * \param aTag data tag
     ******************************************************************************/
    Plato::ScalarMultiVector get(const std::string& aTag) const;

    /***************************************************************************//**
     * \fn void print
     *
     * \brief Print solutions metadata.
     ******************************************************************************/
    void print() const;

    /***************************************************************************//**
     * \fn bool defined
     *
     * \brief Check if solution with input tag is defined in the database
     * \param [in] aTag solution tag/identifier
     * \return boolean (true = is defined; false = not defined) 
     ******************************************************************************/
    bool defined(const std::string & aTag) const;

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

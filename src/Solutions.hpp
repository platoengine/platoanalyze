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
    std::string mPhysics; /*!< physics to be analyzed/simulated */
    std::unordered_map<std::string, Plato::ScalarMultiVector> mSolution; /*!< map from state solution name to 2D POD array */

public:
    /***************************************************************************//**
     * \fn Solutions
     *
     * \brief Constructor.
     * \param [in] aPhysics physics to be analyzed/simulated
     ******************************************************************************/
    explicit Solutions(const std::string & aPhysics);

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
};
// struct Solutions

}
// namespace Plato

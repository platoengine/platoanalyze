/*
 * Variables.hpp
 *
 *  Created on: Apr 6, 2021
 */

#pragma once

#include <unordered_map>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/***************************************************************************//**
 * \struct Variables
 *
 * \brief Holds quantity of interests pertaining to the simulation.
 ******************************************************************************/
struct Variables
{
private:
    std::unordered_map<std::string, Plato::Scalar> mScalars; /*!< map to scalar quantities of interest */
    std::unordered_map<std::string, Plato::ScalarVector> mVectors; /*!< map to vector quantities of interest */

public:
    /***************************************************************************//**
     * \fn void scalar
     * \brief Return scalar value associated with this tag.
     * \param [in] aTag element tag/key
     ******************************************************************************/
    Plato::Scalar scalar(const std::string& aTag) const;

    /***************************************************************************//**
     * \fn void scalar
     * \brief Set (element,key) pair in scalar value map.
     * \param [in] aTag   element tag/key
     * \param [in] aInput element value
     ******************************************************************************/
    void scalar(const std::string& aTag, const Plato::Scalar& aInput);

    /***************************************************************************//**
     * \fn void vector
     * \brief Return scalar vector associated with this tag/key.
     * \param [in] aTag element tag/key
     ******************************************************************************/
    Plato::ScalarVector vector(const std::string& aTag) const;

    /***************************************************************************//**
     * \fn void vector
     * \brief Set (element,key) pair in vector value map.
     * \param [in] aTag   element tag/key
     * \param [in] aInput element value
     ******************************************************************************/
    void vector(const std::string& aTag, const Plato::ScalarVector& aInput);

    /***************************************************************************//**
     * \fn bool isVectorMapEmpty
     * \brief Returns true if vector map is empty; false, if not empty.
     * \return boolean (true or false)
     ******************************************************************************/
    bool isVectorMapEmpty() const;

    /***************************************************************************//**
     * \fn bool isScalarMapEmpty
     * \brief Returns true if scalar map is empty; false, if not empty.
     * \return boolean (true or false)
     ******************************************************************************/
    bool isScalarMapEmpty() const;

    /***************************************************************************//**
     * \fn bool defined
     * \brief Returns true if element with tak/key is defined in a map.
     * \return boolean (true or false)
     ******************************************************************************/
    bool defined(const std::string & aTag) const;

    /***************************************************************************//**
     * \fn void print
     * \brief Print metadata stored in member containers.
     ******************************************************************************/
    void print() const;

private:
    /***************************************************************************//**
     * \fn void printVectorMap
     * \brief Print metadata stored in vector map.
     ******************************************************************************/
    void printVectorMap() const;

    /***************************************************************************//**
     * \fn void printVectorMap
     * \brief Print metadata stored in scalar map.
     ******************************************************************************/
    void printScalarMap() const;
};
// struct Variables

typedef Variables Dual;   /*!< variant name used to identify quantities associated with the dual optimization problem */
typedef Variables Primal; /*!< variant name used to identify quantities associated with the primal optimization problem */

}
// namespace Plato

namespace Plato
{

/***************************************************************************//**
 * \struct FieldTags
 *
 * \brief Holds string identifiers linked to field data tags. Field data is defined
 *   as the field data saved on the finite element mesh metadata structure.
 ******************************************************************************/
struct FieldTags
{
private:
    std::unordered_map<std::string, std::string> mFields; /*!< map from field data tag to field data identifier */

public:
    /***************************************************************************//**
     * \fn void set
     * \brief Set field data tag and associated identifier.
     * \param [in] aTag field data tag
     * \param [in] aID  field data identifier
     ******************************************************************************/
    void set(const std::string& aTag, const std::string& aID);

    /***************************************************************************//**
     * \fn void tags
     * \brief Return list of field data tags.
     * \return list of field data tags
     ******************************************************************************/
    std::vector<std::string> tags() const;

    /***************************************************************************//**
     * \fn void name
     * \brief Return field data identifier.
     * \param [in] aTag field data tag
     * \return field data identifier
     ******************************************************************************/
    std::string id(const std::string& aTag) const;
};
// struct FieldTags

}
// namespace Plato

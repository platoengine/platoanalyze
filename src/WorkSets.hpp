/*
 * WorkSets.hpp
 *
 *  Created on: Apr 5, 2021
 */

#pragma once

#include <unordered_map>

#include "MetaData.hpp"
#include "ImplicitFunctors.hpp"

namespace Plato
{

/***************************************************************************//**
 * \struct WorkSets
 * \brief Map with Plato metadata worksets.
 ******************************************************************************/
struct WorkSets
{
private:
    std::unordered_map<std::string, std::shared_ptr<Plato::MetaDataBase>> mData; /*!< map from tag to metadata shared pointer */

public:
    WorkSets(){}

    /***************************************************************************//**
     * \fn void set
     * \brief Set element metadata at input key location.
     * \param aName metadata tag (i.e. key)
     * \param aData metadata shared pointer
     ******************************************************************************/
    void set(const std::string & aName, const std::shared_ptr<Plato::MetaDataBase> & aData);

    /***************************************************************************//**
     * \fn const std::shared_ptr<Plato::MetaDataBase> & get
     * \brief Return const reference to metadata shared pointer at input key location.
     * \param aName metadata tag (i.e. key)
     * \return metadata shared pointer
     ******************************************************************************/
    const std::shared_ptr<Plato::MetaDataBase> & get(const std::string & aName) const;

    /***************************************************************************//**
     * \fn std::vector<std::string> tags
     * \brief Return list of keys/tags in the metadata map.
     * \return list of keys/tags in the metadata map
     ******************************************************************************/
    std::vector<std::string> tags() const;

    /***************************************************************************//**
     * \fn bool defined
     * \brief Return true is key/metadata pair is defined in the metadata map, else
     *   return false.
     * \return boolean (true or false)
     ******************************************************************************/
    bool defined(const std::string & aTag) const;
};
// struct WorkSets

/***************************************************************************//**
 * \tparam PhysicsT physics type, e.g. fluid, mechancis, thermal, etc.
 *
 * \struct LocalOrdinalMaps
 *
 * \brief Collection of ordinal id maps for scalar, vector, and control fields.
 ******************************************************************************/
template <typename PhysicsT>
struct LocalOrdinalMaps
{
    Plato::NodeCoordinate<PhysicsT::SimplexT::mNumSpatialDims> mNodeCoordinate; /*!< list of node coordinates */
    Plato::VectorEntryOrdinal<PhysicsT::SimplexT::mNumSpatialDims, 1 /*scalar dofs per node*/>                 mScalarFieldOrdinalsMap; /*!< element to scalar field degree of freedom map */
    Plato::VectorEntryOrdinal<PhysicsT::SimplexT::mNumSpatialDims, PhysicsT::SimplexT::mNumSpatialDims>        mVectorFieldOrdinalsMap; /*!< element to vector field degree of freedom map */
    Plato::VectorEntryOrdinal<PhysicsT::SimplexT::mNumSpatialDims, PhysicsT::SimplexT::mNumControlDofsPerNode> mControlOrdinalsMap; /*!< element to control field degree of freedom map */

    /***************************************************************************//**
     * \fn LocalOrdinalMaps
     *
     * \brief Constructor
     * \param [in] aMesh mesh metadata
     ******************************************************************************/
    LocalOrdinalMaps(Omega_h::Mesh & aMesh) :
        mNodeCoordinate(&aMesh),
        mScalarFieldOrdinalsMap(&aMesh),
        mVectorFieldOrdinalsMap(&aMesh),
        mControlOrdinalsMap(&aMesh)
    { return; }
};
// struct LocalOrdinalMaps

}
// namespace Plato

/*
 * MetaData.hpp
 *
 *  Created on: Apr 5, 2021
 */

#pragma once

#include <memory>

namespace Plato
{

/***************************************************************************//**
 *  \class MetaDataBase
 *  \brief Plato metadata pure virtual base class.
 ******************************************************************************/
class MetaDataBase
{
public:
    virtual ~MetaDataBase() = 0;
};
inline MetaDataBase::~MetaDataBase(){}
// class MetaDataBase

/***************************************************************************//**
 * \tparam Type metadata type
 * \class MetaData
 * \brief Plato metadata derived class.
 ******************************************************************************/
template<class Type>
class MetaData : public MetaDataBase
{
public:
    /***************************************************************************//**
     * \brief Constructor
     * \param aData metadata
     ******************************************************************************/
    explicit MetaData(const Type &aData) : mData(aData) {}
    MetaData() {}
    Type mData; /*!< metadata */
};
// class MetaData

/***************************************************************************//**
 * \tparam Type metadata type
 *
 * \fn inline Type metadata
 *
 * \brief Perform dynamic cast from MetaDataBase to Type data.
 *
 * \param aInput shared pointer of Plato metadata
 * \return Type data
 ******************************************************************************/
template<class Type>
inline Type metadata(const std::shared_ptr<Plato::MetaDataBase> & aInput)
{
    return (dynamic_cast<Plato::MetaData<Type>&>(aInput.operator*()).mData);
}
// function metadata

}
// namespace Plato

#pragma once

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Base class for spatial domain
 */
class SpatialDomain
/******************************************************************************/
{
public:
    Omega_h::Mesh     & Mesh;     /*!< mesh database */
    Omega_h::MeshSets & MeshSets; /*!< element, side, node sets database */

private:
    std::string         mElementBlockName;  /*!< element block name */
    std::string         mMaterialModelName; /*!< material model name */
    std::string         mSpatialDomainName; /*!< element block name */
    Omega_h::LOs        mElemLids;          /*!< element block local identification numbers */

public:
    /******************************************************************************//**
     * \fn getMaterialName
     * \brief Return material model name.
     * \return material model name
     **********************************************************************************/
    decltype(mMaterialModelName) getMaterialName() const
    {
        return mMaterialModelName;
    }

    /******************************************************************************//**
     * \fn getDomainName
     * \brief Return domain name.
     * \return domain name
     **********************************************************************************/
    decltype(mSpatialDomainName) getDomainName() const
    {
        return mSpatialDomainName;
    }

    /******************************************************************************//**
     * \fn numCells
     * \brief Return the number of cells.
     * \return number of cells
     **********************************************************************************/
    Plato::OrdinalType numCells() const
    {
        return mElemLids.size();
    }

    /******************************************************************************//**
     * \fn cellOrdinals
     * \brief Return the list of cell ordinals.
     * \return list of cell ordinals
     **********************************************************************************/
    Omega_h::LOs cellOrdinals() const
    {
        return mElemLids;
    }

    /******************************************************************************//**
     * \fn cellOrdinals
     * \brief Set cell ordinals for this element set (block).
     * \param [in] aName element set (block) name
     **********************************************************************************/
    void cellOrdinals(const std::string & aName)
    {
        auto &tElemSets = MeshSets[Omega_h::ELEM_SET];
        auto tElemSetsI = tElemSets.find(aName);
        if (tElemSetsI == tElemSets.end())
        {
            std::ostringstream tMsg;
            tMsg << "Could not find element set (block) with name = '" << aName << "'.";
            THROWERR(tMsg.str())
        }
        mElementBlockName = aName;
        mElemLids = (tElemSetsI->second);
    }

    /******************************************************************************//**
     * \fn cellOrdinals
     * \brief Set cell ordinals for this element set (block)
     * \param [in] aOrdinals list of cell ordinals
     **********************************************************************************/
    void cellOrdinals(const std::string & aName, const Omega_h::LOs & aOrdinals)
    {
        mElemLids = aOrdinals;
        mElementBlockName = aName;
    }

    /******************************************************************************//**
     * \brief Constructor for Plato::SpatialModel base class
     * \param [in] aMesh Default mesh
     * \param [in] aMeshSets mesh sets (nodesets, sidesets)
     * \param [in] aInputParams Spatial model definition
     **********************************************************************************/
    SpatialDomain
    (      Omega_h::Mesh     & aMesh,
           Omega_h::MeshSets & aMeshSets,
     const std::string       & aName) :
        Mesh(aMesh),
        MeshSets(aMeshSets),
        mSpatialDomainName(aName)
    {}

    /******************************************************************************//**
     * \brief Constructor for Plato::SpatialModel base class
     * \param [in] aMesh        Default mesh
     * \param [in] aMeshSets    mesh sets (nodesets, sidesets)
     * \param [in] aInputParams Spatial model definition
     * \param [in] aName        Spatial model name
     **********************************************************************************/
    SpatialDomain
    (      Omega_h::Mesh          & aMesh,
           Omega_h::MeshSets      & aMeshSets,
     const Teuchos::ParameterList & aInputParams,
     const std::string            & aName) :
        Mesh(aMesh),
        MeshSets(aMeshSets),
        mSpatialDomainName(aName)
    {
        if (aInputParams.isType<std::string>("Element Block"))
        {
            auto tElemBlockName = aInputParams.get<std::string>("Element Block");
            this->cellOrdinals(tElemBlockName);
        }
        else
        {
            THROWERR("Parsing new Domain. Required keyword 'Element Block' not found");
        }

        if (aInputParams.isType<std::string>("Material Model"))
        {
            mMaterialModelName = aInputParams.get<std::string>("Material Model");
        }
        else
        {
            THROWERR("Parsing new Domain. Required keyword 'Material Model' not found");
        }
    }
};
// class SpatialDomain

/******************************************************************************/
/*!
 \brief Spatial models contain the mesh, meshsets, domains, etc that define
 a discretized geometry.
 */
class SpatialModel
/******************************************************************************/
{
public:
    Omega_h::Mesh     & Mesh;     /*!< mesh database */
    Omega_h::MeshSets & MeshSets; /*!< element, side, node sets database */

    std::vector<Plato::SpatialDomain> Domains; /*!< list of spatial domains, i.e. element blocks */

    /******************************************************************************//**
     * \brief Constructor for Plato::SpatialModel base class
     * \param [in] aMesh Default mesh
     * \param [in] aMeshSets mesh sets (nodesets, sidesets)
     * \param [in] aInputParams Spatial model definition
     **********************************************************************************/
    SpatialModel(Omega_h::Mesh &aMesh,
                 Omega_h::MeshSets &aMeshSets,
                 const Teuchos::ParameterList &aInputParams) :
        Mesh(aMesh),
        MeshSets(aMeshSets)
    {
        if (aInputParams.isSublist("Spatial Model"))
        {
            auto tModelParams = aInputParams.sublist("Spatial Model");
            if (!tModelParams.isSublist("Domains"))
            {
                THROWERR("Parsing 'Spatial Model'. Required 'Domains' list not found");
            }

            auto tDomainsParams = tModelParams.sublist("Domains");
            for (auto tIndex = tDomainsParams.begin(); tIndex != tDomainsParams.end(); ++tIndex)
            {
                const auto &tEntry = tDomainsParams.entry(tIndex);
                const auto &tMyName = tDomainsParams.name(tIndex);

                if (!tEntry.isList())
                {
                    THROWERR("Parameter in Domains list not valid.  Expect lists only.");
                }

                Teuchos::ParameterList &tDomainParams = tDomainsParams.sublist(tMyName);
                Domains.push_back( { aMesh, aMeshSets, tDomainParams, tMyName });
            }
        }
        else
        {
            THROWERR("Parsing 'Plato Problem'. Required 'Spatial Model' list not found");
        }
    }
};
// class SpatialModel

} // namespace Plato

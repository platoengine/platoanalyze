#pragma once

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

#include "PlatoMask.hpp"
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

    Plato::LocalOrdinalVector mTotalElemLids;   /*!< List of all elements in this domain */
    Plato::LocalOrdinalVector mMaskedElemLids;  /*!< List of elements after a mask is applied */

public:
    /******************************************************************************//**
     * \fn getDomainName
     * \brief Return domain name.
     * \return domain name
     **********************************************************************************/
    decltype(mSpatialDomainName) 
    getDomainName() const
    {
        return mSpatialDomainName;
    }

    /******************************************************************************//**
     * \fn getMaterialName
     * \brief Return material model name.
     * \return material model name
     **********************************************************************************/
    decltype(mMaterialModelName) 
    getMaterialName() const
    {
        return mMaterialModelName;
    }

    /******************************************************************************//**
     * \fn setMaterialName
     * \brief Set material model name.
     * \param [in] aName material model name
     **********************************************************************************/
    void setMaterialName(const std::string & aName)
    {
        mMaterialModelName = aName;
    }

    /******************************************************************************//**
     * \fn getElementBlockName
     * \brief Return element block name.
     * \return element block name
     **********************************************************************************/
    decltype(mElementBlockName) 
    getElementBlockName() const
    {
        return mElementBlockName;
    }

    /******************************************************************************//**
     * \fn setElementBlockName
     * \brief Set element block name.
     * \param [in] aName element block name
     **********************************************************************************/
    void setElementBlockName
    (const std::string & aName)
    {
        mElementBlockName = aName;
    }

    /******************************************************************************//**
     * \fn numCells
     * \brief Return the number of cells.
     * \return number of cells
     **********************************************************************************/
    Plato::OrdinalType 
    numCells() const
    {
        return mMaskedElemLids.extent(0);
    }

    /******************************************************************************//**
     * \brief get cell ordinal list
     * Note: A const reference is returned to prevent the ref count from being modified.  
    **********************************************************************************/
    const Plato::LocalOrdinalVector &
    cellOrdinals() const
    {
        return mMaskedElemLids;
    }

    /******************************************************************************//**
     * \fn cellOrdinals
     * \brief Set cell ordinals for this element block.
     * \param [in] aName element block name
     **********************************************************************************/
    void cellOrdinals(const std::string & aName)
    {
        this->setMaskLocalElemIDs(aName);
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
        this->initialize(aInputParams);
    }

    /******************************************************************************//**
     * \brief Apply mask to this Domain
     *        This function removes elements that have a mask value of zero in \p aMask.
     *        Subsequent calls to numCells() and cellOrdinals() refer to the reduced list.
     *        Call removeMask() to remove the mask, or applyMask(...) to apply a different
     *        mask.
     * \param [in] aMask Plato Mask specifying active/inactive nodes and elements
    **********************************************************************************/
    template<Plato::OrdinalType mSpatialDim>
    void applyMask
    (std::shared_ptr<Plato::Mask<mSpatialDim>> aMask)
    {
        using OrdinalT = Plato::OrdinalType;

        auto tMask = aMask->cellMask();
        auto tTotalElemLids = mTotalElemLids;
        auto tNumEntries = tTotalElemLids.extent(0);

        // how many non-zeros in the mask?
        Plato::OrdinalType tSum(0);
        Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,tNumEntries), 
        LAMBDA_EXPRESSION(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType & aUpdate)
        {
            auto tElemOrdinal = tTotalElemLids(aOrdinal);
            aUpdate += tMask(tElemOrdinal); 
        }, tSum);
        Kokkos::resize(mMaskedElemLids, tSum);

        auto tMaskedElemLids = mMaskedElemLids;

        // create a list of elements with non-zero mask values
        OrdinalT tOffset(0);
        Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalT>(0,tNumEntries),
        KOKKOS_LAMBDA (const OrdinalT& aOrdinal, OrdinalT& aUpdate, const bool& tIsFinal)
        {
            auto tElemOrdinal = tTotalElemLids(aOrdinal);
            const OrdinalT tVal = tMask(tElemOrdinal);
            if( tIsFinal && tVal ) { tMaskedElemLids(aUpdate) = tElemOrdinal; }
            aUpdate += tVal;
        }, tOffset);
    }
        
    /******************************************************************************//**
     * \brief Remove applied mask.
     *        This function resets the element list in this domain to the original definition.
     *        If no mask has been applied, this function has no effect.
    **********************************************************************************/
    void
    removeMask()
    {
        Kokkos::deep_copy(mMaskedElemLids, mTotalElemLids);
    }
    
    Omega_h::LOs 
    getLocalElemIDs
    (const std::string& aBlockName) const
    {
        auto& tElemSets = MeshSets[Omega_h::ELEM_SET];
        auto tElemSetsI = tElemSets.find(aBlockName);
        if(tElemSetsI == tElemSets.end())
        {
            std::ostringstream tMsg;
            tMsg << "Could not find element set (block) with name = '" << aBlockName;
            THROWERR(tMsg.str())
        }

        return (tElemSetsI->second);
    }

    void setMaskLocalElemIDs
    (const std::string& aBlockName)
    {
        auto tElemLids = this->getLocalElemIDs(aBlockName);
        auto tNumElems = tElemLids.size();
        mTotalElemLids = Plato::LocalOrdinalVector("element list", tNumElems);
        mMaskedElemLids = Plato::LocalOrdinalVector("masked element list", tNumElems);

        auto tTotalElemLids = mTotalElemLids;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumElems), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tTotalElemLids(aCellOrdinal) = tElemLids[aCellOrdinal];
        }, "get element ids");
        Kokkos::deep_copy(mMaskedElemLids, mTotalElemLids);
    }

    void 
    initialize
    (const Teuchos::ParameterList & aInputParams)
    {
        if(aInputParams.isType<std::string>("Element Block"))
        {
            mElementBlockName = aInputParams.get<std::string>("Element Block");
            this->cellOrdinals(mElementBlockName);
        }
        else
        {
            THROWERR("Parsing new Domain. Required keyword 'Element Block' not found");
        }

        if(aInputParams.isType<std::string>("Material Model"))
        {
            mMaterialModelName = aInputParams.get<std::string>("Material Model");
        }
        else
        {
            THROWERR("Parsing new Domain. Required keyword 'Material Model' not found");
        }

        this->setMaskLocalElemIDs(mElementBlockName);
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
     * \param [in] aMesh     Default mesh
     * \param [in] aMeshSets mesh sets (nodesets, sidesets)
     **********************************************************************************/
    SpatialModel
    (Omega_h::Mesh     & aMesh,
     Omega_h::MeshSets & aMeshSets) :
         Mesh(aMesh),
         MeshSets(aMeshSets)
    {}

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

    template <Plato::OrdinalType mSpatialDim>
    void applyMask
    (std::shared_ptr<Plato::Mask<mSpatialDim>> aMask)
    {
        for( auto& tDomain : Domains )
        {
            tDomain.applyMask(aMask);
        }
    }

    /******************************************************************************//**
     * \brief Append spatial domain to spatial model.
     * \param [in] aDomain Spatial domain
     **********************************************************************************/
    void append
    (Plato::SpatialDomain & aDomain)
    {
        Domains.push_back(aDomain);
    }
};
// class SpatialModel

} // namespace Plato

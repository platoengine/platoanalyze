#pragma once

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato {

    /******************************************************************************/
    /*!
      \brief Base class for spatial domain
    */
    class SpatialDomain
    /******************************************************************************/
    {
      public:
        Omega_h::Mesh     & Mesh;
        Omega_h::MeshSets & MeshSets;

      private:
        std::string         mElementBlockName;
        std::string         mMaterialModelName;
        std::string         mSpatialDomainName;
        Omega_h::LOs        mElemLids;

      public:
        decltype(mMaterialModelName) getMaterialName() const {return mMaterialModelName;}
        decltype(mSpatialDomainName) getDomainName()   const {return mSpatialDomainName;}

        Plato::OrdinalType
        numCells() const
        {
            return mElemLids.size();
        }

        Omega_h::LOs
        cellOrdinals() const
        {
            return mElemLids;
        }


        /******************************************************************************//**
         * \brief Constructor for Plato::SpatialModel base class
         * \param [in] aMesh Default mesh
         * \param [in] aMeshSets mesh sets (nodesets, sidesets)
         * \param [in] aInputParams Spatial model definition
        **********************************************************************************/
        SpatialDomain(
                  Omega_h::Mesh          & aMesh,
                  Omega_h::MeshSets      & aMeshSets,
            const Teuchos::ParameterList & aInputParams,
                  std::string              aName
        ) :
            Mesh(aMesh),
            MeshSets(aMeshSets),
            mSpatialDomainName(aName)
        {
            if(aInputParams.isType<std::string>("Element Block"))
            {
                mElementBlockName = aInputParams.get<std::string>("Element Block");
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

            auto& tElemSets = MeshSets[Omega_h::ELEM_SET];
            auto tElemSetsI = tElemSets.find(mElementBlockName);
            if(tElemSetsI == tElemSets.end())
            {
                std::ostringstream tMsg;
                tMsg << "Could not find element set (block) with name = '" << mElementBlockName;
                THROWERR(tMsg.str())
            }
            mElemLids = (tElemSetsI->second);

        }
    };

    /******************************************************************************/
    /*!
      \brief Spatial models contain the mesh, meshsets, domains, etc that define
      a discretized geometry.
    */
    class SpatialModel
    /******************************************************************************/
    {
      public:
        Omega_h::Mesh     & Mesh;
        Omega_h::MeshSets & MeshSets;

        std::vector<Plato::SpatialDomain> Domains;
       
        /******************************************************************************//**
         * \brief Constructor for Plato::SpatialModel base class
         * \param [in] aMesh Default mesh
         * \param [in] aMeshSets mesh sets (nodesets, sidesets)
         * \param [in] aInputParams Spatial model definition
        **********************************************************************************/
        SpatialModel(
                  Omega_h::Mesh          & aMesh,
                  Omega_h::MeshSets      & aMeshSets,
            const Teuchos::ParameterList & aInputParams
        ) :
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
                for(auto tIndex = tDomainsParams.begin(); tIndex != tDomainsParams.end(); ++tIndex)
                {
                    const auto & tEntry  = tDomainsParams.entry(tIndex);
                    const auto & tMyName = tDomainsParams.name(tIndex);

                    if (!tEntry.isList())
                    {
                        THROWERR("Parameter in Domains list not valid.  Expect lists only.");
                    }

                    Teuchos::ParameterList& tDomainParams = tDomainsParams.sublist(tMyName);
                    Domains.push_back({aMesh, aMeshSets, tDomainParams, tMyName});
                }
            }
            else
            {
                THROWERR("Parsing 'Plato Problem'. Required 'Spatial Model' list not found");
            }
        }
    };
} // namespace Plato

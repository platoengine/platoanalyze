#ifndef ESSENTIAL_BC_HPP
#define ESSENTIAL_BC_HPP

#include <sstream>

#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

#include "PlatoMathExpr.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Class for essential boundary conditions.
 */
template<typename SimplexPhysicsType>
class EssentialBC
/******************************************************************************/
{
public:

    EssentialBC<SimplexPhysicsType>(
        const std::string            & aName,
              Teuchos::ParameterList & aParam,
              Plato::Scalar            aScaleFactor = 1.0
    ) :
        mName(aName),
        mNodeSetName(aParam.get < std::string > ("Sides")),
        mDof(aParam.get<Plato::OrdinalType>("Index", 0)),
        mMathExpr(nullptr),
        mScaleFactor(aScaleFactor)
    {
        if (aParam.isType<Plato::Scalar>("Value") && aParam.isType<std::string>("Function") )
        {
            THROWERR("Specify either 'Value' or 'Function' in Boundary Condition definition");
        } else
        if (!aParam.isType<Plato::Scalar>("Value") && !aParam.isType<std::string>("Function") )
        {
            THROWERR("Specify either 'Value' or 'Function' in Boundary Condition definition");
        } else
        if (aParam.isType<Plato::Scalar>("Value"))
        {
            mValue = aParam.get<Plato::Scalar>("Value");
        } else
        if (aParam.isType<std::string>("Function"))
        {
            auto tValueExpr = aParam.get<std::string>("Function");
            mMathExpr = std::make_shared<Plato::MathExpr>(tValueExpr);
        }
    }

    ~EssentialBC()
    {
    }

    /*!
     \brief Get the ordinals/values of the constrained nodeset.
     \param aMeshSets Omega_h mesh sets that contains the constrained nodeset.
     \param bcDofs Ordinal list to which the constrained dofs will be added.
     \param bcValues Value list to which the constrained value will be added.
     \param offset Starting location in bcDofs/bcValues where constrained dofs/values will be added.
     */
    void
    get(
        const Omega_h::MeshSets  & aMeshSets,
              LocalOrdinalVector & aBcDofs,
              ScalarVector       & aBcValues,
              OrdinalType          aOffset,
              Plato::Scalar        aTime=0);

    // ! Get number of nodes is the constrained nodeset.
    OrdinalType get_length(const Omega_h::MeshSets& aMeshSets);

    // ! Get nodeset name
    std::string const& get_ns_name() const
    {
        return mNodeSetName;
    }

    // ! Get index of constrained dof (i.e., if X dof is to be constrained, get_dof() returns 0).
    Plato::OrdinalType get_dof() const
    {
        return mDof;
    }

    // ! Get the value to which the dofs will be constrained.
    Plato::Scalar get_value(Plato::Scalar aTime) const
    {
        if (mMathExpr == nullptr)
        {
            return mValue / mScaleFactor;
        }
        else
        {
            return mMathExpr->value(aTime) / mScaleFactor;
        }
    }

protected:
    const std::string mName;
    const std::string mNodeSetName;
    const Plato::OrdinalType mDof;
    Plato::Scalar mValue;
    std::shared_ptr<Plato::MathExpr> mMathExpr;
    Plato::Scalar mScaleFactor;

};

/******************************************************************************/
/*!
 \brief Owner class that contains a vector of EssentialBC objects.
 */
template<typename SimplexPhysicsType>
class EssentialBCs
/******************************************************************************/
{
private:

    const Omega_h::MeshSets  & mMeshSets;

    std::vector<std::shared_ptr<EssentialBC<SimplexPhysicsType>>>mBCs;

public :

    /*!
     \brief Constructor that parses and creates a vector of EssentialBC objects
     based on the ParameterList.
     */
    EssentialBCs(Teuchos::ParameterList &aParams, const Omega_h::MeshSets & aMeshSets);

    /*!
     \brief Constructor that parses and creates a vector of EssentialBC objects with scale factors
     */
    EssentialBCs(Teuchos::ParameterList &aParams, const Omega_h::MeshSets & aMeshSets, 
                 const std::map<Plato::OrdinalType, Plato::Scalar> & aDofOffsetToScaleFactor);

    /*!
     \brief Get ordinals and values for constraints.
     \param [in/out] aBcDofs   Ordinals of all constrained dofs.
     \param [in/out] aBcValues Values of all constrained dofs.
     \param [in]     aTime     Current time (default=0.0).
     */
    void
    get(
        LocalOrdinalVector & aBcDofs,
        ScalarVector       & aBcValues,
        Plato::Scalar        aTime=0.0);

    bool empty() const
    {
        return mBCs.empty();
    }
};

/****************************************************************************/
template<typename SimplexPhysicsType>
OrdinalType EssentialBC<SimplexPhysicsType>::get_length(const Omega_h::MeshSets& aMeshSets)
/****************************************************************************/
{
    auto& tNodeSets = aMeshSets[Omega_h::NODE_SET];
    auto tNodeSetsIter = tNodeSets.find(this->mNodeSetName);
    if(tNodeSetsIter == tNodeSets.end())
    {
        std::ostringstream tMsg;
        tMsg << "COULD NOT FIND NODE SET WITH NAME = '" << mNodeSetName.c_str()
                << "'.  NODE SET IS NOT DEFINED IN INPUT MESH FILE, I.E. EXODUS FILE.\n";
        THROWERR(tMsg.str())
    }
    auto tNodeLids = (tNodeSetsIter->second);
    auto tNumberConstrainedNodes = tNodeLids.size();

    if (tNumberConstrainedNodes == static_cast<Plato::OrdinalType>(0))
    {
        const std::string tErrorMessage = std::string("The set '") +
              mNodeSetName + "' specified in Essential Boundary Conditions contains 0 nodes.";
        WARNING(tErrorMessage)
    }

    return tNumberConstrainedNodes;
}

/****************************************************************************/
template<typename SimplexPhysicsType>
void
EssentialBC<SimplexPhysicsType>::get(
    const Omega_h::MeshSets  & aMeshSets,
          LocalOrdinalVector & aBcDofs,
          ScalarVector       & aBcValues,
          OrdinalType          aOffset,
          Plato::Scalar        aTime)
/****************************************************************************/
{
    // parse constrained nodesets
    auto& tNodeSets = aMeshSets[Omega_h::NODE_SET];
    auto tNodeSetsIter = tNodeSets.find(this->mNodeSetName);
    if(tNodeSetsIter == tNodeSets.end())
    {
        std::ostringstream tMsg;
        tMsg << "Could not find Node Set with name = '" << mNodeSetName.c_str()
                << "'. Node Set is not defined in input geometry/mesh file.\n";
        THROWERR(tMsg.str())
    }
    auto tNodeLids = (tNodeSetsIter->second);
    auto tNumberConstrainedNodes = tNodeLids.size();

    constexpr Plato::OrdinalType tDofsPerNode = SimplexPhysicsType::mNumDofsPerNode;

    auto tValue = this->get_value(aTime);

    auto tLdofs = aBcDofs;
    auto tLvals = aBcValues;
    auto tLdof = this->get_dof();
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberConstrainedNodes), LAMBDA_EXPRESSION(Plato::OrdinalType aNodeOrdinal)
    {
        tLdofs(aOffset+aNodeOrdinal) = tDofsPerNode*tNodeLids[aNodeOrdinal]+tLdof;
        tLvals(aOffset+aNodeOrdinal) = tValue;
    }, "Dirichlet BC");
}

/****************************************************************************/
template<typename SimplexPhysicsType>
EssentialBCs<SimplexPhysicsType>::EssentialBCs(
          Teuchos::ParameterList & aParams,
    const Omega_h::MeshSets      & aMeshSets
) :
    mMeshSets(aMeshSets),
    mBCs()
/****************************************************************************/
{
    for(Teuchos::ParameterList::ConstIterator tIndex = aParams.begin(); tIndex != aParams.end(); ++tIndex)
    {
        const Teuchos::ParameterEntry & tEntry = aParams.entry(tIndex);
        const std::string & tMyName = aParams.name(tIndex);

        TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error, " Parameter in Boundary Conditions block not valid.  Expect lists only.");

        Teuchos::ParameterList& tSublist = aParams.sublist(tMyName);
        const std::string tType = tSublist.get < std::string > ("Type");
        std::shared_ptr<EssentialBC<SimplexPhysicsType>> tMyBC;
        if("Zero Value" == tType)
        {
            const std::string tValueDocument = "solution component set to zero.";
            tSublist.set("Value", static_cast<Plato::Scalar>(0.0), tValueDocument);
            tMyBC.reset(new EssentialBC<SimplexPhysicsType>(tMyName, tSublist));
        }
        else if(tType == "Fixed Value" || tType == "Time Dependent")
            tMyBC.reset(new EssentialBC<SimplexPhysicsType>(tMyName, tSublist));
        else
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, " Boundary Condition type invalid: Not 'Zero Value' or 'Fixed Value'.");
        mBCs.push_back(tMyBC);
    }
}

/****************************************************************************/
template<typename SimplexPhysicsType>
EssentialBCs<SimplexPhysicsType>::EssentialBCs(
          Teuchos::ParameterList & aParams,
    const Omega_h::MeshSets      & aMeshSets,
    const std::map<Plato::OrdinalType, Plato::Scalar> & aDofOffsetToScaleFactor
) :
    mMeshSets(aMeshSets),
    mBCs()
/****************************************************************************/
{
    for(Teuchos::ParameterList::ConstIterator tIndex = aParams.begin(); tIndex != aParams.end(); ++tIndex)
    {
        const Teuchos::ParameterEntry & tEntry = aParams.entry(tIndex);
        const std::string & tMyName = aParams.name(tIndex);

        TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error, " Parameter in Boundary Conditions block not valid.  Expect lists only.");

        Teuchos::ParameterList& tSublist = aParams.sublist(tMyName);
        const std::string tType = tSublist.get <std::string> ("Type");
        const Plato::OrdinalType tDofIndex = tSublist.get<Plato::OrdinalType>("Index", 0);
        Plato::Scalar tScaleFactor(1.0);
        auto tSearch = aDofOffsetToScaleFactor.find(tDofIndex);
        if(tSearch != aDofOffsetToScaleFactor.end())
            tScaleFactor = tSearch->second;

        std::shared_ptr<EssentialBC<SimplexPhysicsType>> tMyBC;
        if("Zero Value" == tType)
        {
            const std::string tValueDocument = "solution component set to zero.";
            tSublist.set("Value", static_cast<Plato::Scalar>(0.0), tValueDocument);
            tMyBC.reset(new EssentialBC<SimplexPhysicsType>(tMyName, tSublist, tScaleFactor));
        }
        else if(tType == "Fixed Value" || tType == "Time Dependent")
            tMyBC.reset(new EssentialBC<SimplexPhysicsType>(tMyName, tSublist, tScaleFactor));
        else
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, " Boundary Condition type invalid: Not 'Zero Value' or 'Fixed Value'.");
        mBCs.push_back(tMyBC);
    }
}

/****************************************************************************/
template<typename SimplexPhysicsType>
void
EssentialBCs<SimplexPhysicsType>::get
(Plato::LocalOrdinalVector & aBcDofs,
 Plato::ScalarVector       & aBcValues,
 Plato::Scalar               aTime)
/****************************************************************************/
{
    Plato::OrdinalType tNumConstrainedDofs(0);
    for(std::shared_ptr<EssentialBC<SimplexPhysicsType>> &tBC : mBCs)
    {
        tNumConstrainedDofs += tBC->get_length(mMeshSets);
    }

    Kokkos::resize(aBcDofs, tNumConstrainedDofs);
    Kokkos::resize(aBcValues, tNumConstrainedDofs);

    Plato::OrdinalType tOffset(0);
    for(std::shared_ptr<EssentialBC<SimplexPhysicsType>> &tBC : mBCs)
    {
        tBC->get(mMeshSets, aBcDofs, aBcValues, tOffset, aTime);
        tOffset += tBC->get_length(mMeshSets);
    }
}

} // namespace Plato

#endif


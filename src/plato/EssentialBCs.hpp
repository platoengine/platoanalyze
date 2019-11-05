#ifndef ESSENTIAL_BC_HPP
#define ESSENTIAL_BC_HPP

#include <sstream>

#include <Omega_h_assoc.hpp>

#include "plato/PlatoStaticsTypes.hpp"

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

    EssentialBC<SimplexPhysicsType>(const std::string & aName, Teuchos::ParameterList & aParam) :
            name(aName),
            ns_name(aParam.get < std::string > ("Sides")),
            dof(aParam.get<Plato::OrdinalType>("Index", 0)),
            value(aParam.get<Plato::Scalar>("Value"))
    {
    }

    ~EssentialBC()
    {
    }

    /*!
     \brief Get the ordinals/values of the constrained nodeset.
     @param aMeshSets Omega_h mesh sets that contains the constrained nodeset.
     @param bcDofs Ordinal list to which the constrained dofs will be added.
     @param bcValues Value list to which the constrained value will be added.
     @param offset Starting location in bcDofs/bcValues where constrained dofs/values will be added.
     */
    void get(const Omega_h::MeshSets& aMeshSets, LocalOrdinalVector& bcDofs, ScalarVector& bcValues, OrdinalType offset);

    // ! Get number of nodes is the constrained nodeset.
    OrdinalType get_length(const Omega_h::MeshSets& aMeshSets);

    // ! Get nodeset name
    std::string const& get_ns_name() const
    {
        return ns_name;
    }

    // ! Get index of constrained dof (i.e., if X dof is to be constrained, get_dof() returns 0).
    Plato::OrdinalType get_dof() const
    {
        return dof;
    }

    // ! Get the value to which the dofs will be constrained.
    Scalar get_value() const
    {
        return value;
    }

protected:
    const std::string name;
    const std::string ns_name;
    const Plato::OrdinalType dof;
    const Scalar value;

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
    std::vector<std::shared_ptr<EssentialBC<SimplexPhysicsType>>>BCs;
public :

    /*!
     \brief Constructor that parses and creates a vector of EssentialBC objects
     based on the ParameterList.
     */
    EssentialBCs(Teuchos::ParameterList &aParams);

    /*!
     \brief Get ordinals and values for constraints.
     @param mesh Omega_h mesh that contains the constrained nodesets.
     @param bcDofs Ordinals of all constrained dofs.
     @param bcValues Values of all constrained dofs.
     */
    void get( const Omega_h::MeshSets& aMeshSets,
            LocalOrdinalVector& bcDofs,
            ScalarVector& bcValues);
};

/****************************************************************************/
template<typename SimplexPhysicsType>
OrdinalType EssentialBC<SimplexPhysicsType>::get_length(const Omega_h::MeshSets& aMeshSets)
/****************************************************************************/
{
    auto& tNodeSets = aMeshSets[Omega_h::NODE_SET];
    auto tNodeSetsIter = tNodeSets.find(this->ns_name);
    if(tNodeSetsIter == tNodeSets.end())
    {
        std::ostringstream tMsg;
        tMsg << "Could not find Node Set with name = '" << ns_name.c_str()
                << "'. Node Set is not defined in input geometry/mesh file.\n";
        THROWERR(tMsg.str())
    }
    auto tNodeLids = (tNodeSetsIter->second);
    auto tNumberConstrainedNodes = tNodeLids.size();

    return tNumberConstrainedNodes;
}

/****************************************************************************/
template<typename SimplexPhysicsType>
void EssentialBC<SimplexPhysicsType>::get(const Omega_h::MeshSets& aMeshSets,
                                          LocalOrdinalVector& bcDofs,
                                          ScalarVector& bcValues,
                                          OrdinalType offset)
/****************************************************************************/
{
    // parse constrained nodesets
    auto& tNodeSets = aMeshSets[Omega_h::NODE_SET];
    auto tNodeSetsIter = tNodeSets.find(this->ns_name);
    if(tNodeSetsIter == tNodeSets.end())
    {
        std::ostringstream tMsg;
        tMsg << "Could not find Node Set with name = '" << ns_name.c_str()
                << "'. Node Set is not defined in input geometry/mesh file.\n";
        THROWERR(tMsg.str())
    }
    auto tNodeLids = (tNodeSetsIter->second);
    auto tNumberConstrainedNodes = tNodeLids.size();

    constexpr Plato::OrdinalType dofsPerNode = SimplexPhysicsType::mNumDofsPerNode;

    auto tValue = this->value;
    auto tLdofs = bcDofs;
    auto tLvals = bcValues;
    auto tLdof = this->dof;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberConstrainedNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        tLdofs(offset+nodeOrdinal) = dofsPerNode*tNodeLids[nodeOrdinal]+tLdof;
        tLvals(offset+nodeOrdinal) = tValue;
    }, "Dirichlet BC");
}

/****************************************************************************/
template<typename SimplexPhysicsType>
EssentialBCs<SimplexPhysicsType>::EssentialBCs(Teuchos::ParameterList & aParams) :
        BCs()
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
        else if("Fixed Value" == tType)
            tMyBC.reset(new EssentialBC<SimplexPhysicsType>(tMyName, tSublist));
        else
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, " Boundary Condition type invalid: Not 'Zero Value' or 'Fixed Value'.");
        BCs.push_back(tMyBC);
    }
}

/****************************************************************************/
template<typename SimplexPhysicsType>
void EssentialBCs<SimplexPhysicsType>::get(const Omega_h::MeshSets& aMeshSets, LocalOrdinalVector& bcDofs, ScalarVector& bcValues)
/****************************************************************************/
{
    OrdinalType numConstrainedDofs(0);
    for(std::shared_ptr<EssentialBC<SimplexPhysicsType>> &bc : BCs)
        numConstrainedDofs += bc->get_length(aMeshSets);

    Kokkos::resize(bcDofs, numConstrainedDofs);
    Kokkos::resize(bcValues, numConstrainedDofs);

    OrdinalType offset(0);
    for(std::shared_ptr<EssentialBC<SimplexPhysicsType>> &bc : BCs)
    {
        bc->get(aMeshSets, bcDofs, bcValues, offset);
        offset += bc->get_length(aMeshSets);
    }
}

} // namespace Plato

#endif


/*
 * TieMultipointConstraint.hpp
 *
 *  Created on: May 26, 2020
 */

#include "TieMultipointConstraint.hpp"

namespace Plato
{

/****************************************************************************/
Plato::TieMultipointConstraint::
TieMultipointConstraint(const Omega_h::MeshSets & aMeshSets,
                        const std::string & aName, 
                        Teuchos::ParameterList & aParam) :
                        Plato::MultipointConstraint(aName)
/****************************************************************************/
{
    // parse RHS value
    mValue = aParam.get<Plato::Scalar>("Value");

    // parse child and parent node sets
    auto& tNodeSets = aMeshSets[Omega_h::NODE_SET];
    std::string tChildNodeSet = aParam.get<std::string>("Child");
    std::string tParentNodeSet = aParam.get<std::string>("Parent");
    
    // parse child nodes
    auto tChildNodeSetsIter = tNodeSets.find(tChildNodeSet);
    if(tChildNodeSetsIter == tNodeSets.end())
    {
        std::ostringstream tMsg;
        tMsg << "Could not find Node Set with name = '" << tChildNodeSet.c_str()
                << "'. Node Set is not defined in input geometry/mesh file.\n";
        THROWERR(tMsg.str())
    }
    Omega_h::LOs tChildNodeLids = (tChildNodeSetsIter->second);
    auto tNumberChildNodes = tChildNodeLids.size();
    
    // parse parent nodes
    auto tParentNodeSetsIter = tNodeSets.find(tParentNodeSet);
    if(tParentNodeSetsIter == tNodeSets.end())
    {
        std::ostringstream tMsg;
        tMsg << "Could not find Node Set with name = '" << tParentNodeSet.c_str()
                << "'. Node Set is not defined in input geometry/mesh file.\n";
        THROWERR(tMsg.str())
    }
    Omega_h::LOs tParentNodeLids = (tParentNodeSetsIter->second);
    auto tNumberParentNodes = tParentNodeLids.size();

    // Check that the number of child and parent nodes match
    if (tNumberChildNodes != tNumberParentNodes)
    {
        std::ostringstream tMsg;
        tMsg << "CHILD AND PARENT NODESETS FOR TIE CONSTRAINT NOT OF EQUAL LENGTH. \n";
        THROWERR(tMsg.str())
    }

    // Fill in child and parent nodes
    Kokkos::resize(mChildNodes, tNumberChildNodes);
    Kokkos::resize(mParentNodes, tNumberParentNodes);

    this->updateNodesets(tNumberChildNodes, tChildNodeLids, tParentNodeLids);
}

/****************************************************************************/
void Plato::TieMultipointConstraint::
get(LocalOrdinalVector & aMpcChildNodes,
    LocalOrdinalVector & aMpcParentNodes,
    Plato::CrsMatrixType::RowMapVector & aMpcRowMap,
    Plato::CrsMatrixType::OrdinalVector & aMpcColumnIndices,
    Plato::CrsMatrixType::ScalarVector & aMpcEntries,
    ScalarVector & aMpcValues,
    OrdinalType aOffsetChild,
    OrdinalType aOffsetParent,
    OrdinalType aOffsetNnz)
/****************************************************************************/
{
    auto tValue = mValue;
    auto tNumberChildNodes = mChildNodes.size();

    // Fill in constraint info
    auto tMpcChildNodes = aMpcChildNodes;
    auto tMpcParentNodes = aMpcParentNodes;
    auto tRowMap = aMpcRowMap;
    auto tColumnIndices = aMpcColumnIndices;
    auto tEntries = aMpcEntries;
    auto tValues = aMpcValues;

    auto tChildNodes = mChildNodes;
    auto tParentNodes = mParentNodes;

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        tMpcChildNodes(aOffsetChild + nodeOrdinal) = tChildNodes(nodeOrdinal); // child node ID
        tMpcParentNodes(aOffsetParent + nodeOrdinal) = tParentNodes(nodeOrdinal); // parent node ID

        tRowMap(aOffsetChild + nodeOrdinal) = aOffsetChild + nodeOrdinal; // row map
        tRowMap(aOffsetChild + nodeOrdinal + 1) = aOffsetChild + nodeOrdinal + 1; // row map

        tColumnIndices(aOffsetNnz + nodeOrdinal) = aOffsetParent + nodeOrdinal; // column indices (local parent node ID)
        tEntries(aOffsetNnz + nodeOrdinal) = 1.0; // entries (constraint coefficients)

        tValues(aOffsetChild + nodeOrdinal) = tValue; // constraint RHS

    }, "Tie constraint data");
}

/****************************************************************************/
void Plato::TieMultipointConstraint::
updateLengths(OrdinalType& lengthChild,
              OrdinalType& lengthParent,
              OrdinalType& lengthNnz)
/****************************************************************************/
{
    auto tNumberChildNodes = mChildNodes.size();
    auto tNumberParentNodes = mParentNodes.size();

    lengthChild += tNumberChildNodes;
    lengthParent += tNumberParentNodes;
    lengthNnz += tNumberChildNodes;
}

/****************************************************************************/
void Plato::TieMultipointConstraint::
updateNodesets(const OrdinalType& tNumberChildNodes,
               const Omega_h::LOs& tChildNodeLids,
               const Omega_h::LOs& tParentNodeLids)
/****************************************************************************/
{
    auto tChildNodes = mChildNodes;
    auto tParentNodes = mParentNodes;

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        tChildNodes(nodeOrdinal) = tChildNodeLids[nodeOrdinal]; // child node ID
        tParentNodes(nodeOrdinal) = tParentNodeLids[nodeOrdinal]; // parent node ID
    }, "Tie constraint data");
}

}
// namespace Plato

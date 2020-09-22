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
TieMultipointConstraint(const std::string & aName, Teuchos::ParameterList & aParam) :
    Plato::MultipointConstraint(aName)
/****************************************************************************/
{
    mChildNodeSet = aParam.get<std::string>("Child");
    mParentNodeSet = aParam.get<std::string>("Parent");

    mValue = aParam.get<Plato::Scalar>("Value");
}

/****************************************************************************/
void Plato::TieMultipointConstraint::
get(const Omega_h::Mesh& aMesh,
    const Omega_h::MeshSets& aMeshSets,
    LocalOrdinalVector & mpcChildNodes,
    LocalOrdinalVector & mpcParentNodes,
    Plato::CrsMatrixType::RowMapVector & mpcRowMap,
    Plato::CrsMatrixType::OrdinalVector & mpcColumnIndices,
    Plato::CrsMatrixType::ScalarVector & mpcEntries,
    ScalarVector & mpcValues,
    OrdinalType offsetChild,
    OrdinalType offsetParent,
    OrdinalType offsetNnz)
/****************************************************************************/
{
    auto& tNodeSets = aMeshSets[Omega_h::NODE_SET];
    auto tValue = this->mValue;
    
    // parse child nodes
    auto tChildNodeSetsIter = tNodeSets.find(this->mChildNodeSet);
    auto tChildNodeLids = (tChildNodeSetsIter->second);
    auto tNumberChildNodes = tChildNodeLids.size();
    
    // parse parent nodes
    auto tParentNodeSetsIter = tNodeSets.find(this->mParentNodeSet);
    auto tParentNodeLids = (tParentNodeSetsIter->second);
    auto tNumberParentNodes = tParentNodeLids.size();

    // Check that the number of child and parent nodes match
    if (tNumberChildNodes != tNumberParentNodes)
    {
        std::ostringstream tMsg;
        tMsg << "CHILD AND PARENT NODESETS FOR TIE CONSTRAINT NOT OF EQUAL LENGTH. \n";
        THROWERR(tMsg.str())
    }

    // Fill in constraint info
    auto tChildNodes = mpcChildNodes;
    auto tParentNodes = mpcParentNodes;
    auto tRowMap = mpcRowMap;
    auto tColumnIndices = mpcColumnIndices;
    auto tEntries = mpcEntries;
    auto tValues = mpcValues;

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        tChildNodes(offsetChild+nodeOrdinal) = tChildNodeLids[nodeOrdinal]; // child node ID
        tParentNodes(offsetParent+nodeOrdinal) = tParentNodeLids[nodeOrdinal]; // parent node ID

        tRowMap(offsetChild+nodeOrdinal) = offsetChild + nodeOrdinal; // row map
        tColumnIndices(offsetNnz+nodeOrdinal) = offsetParent + nodeOrdinal; // column indices (local parent node ID)
        tEntries(offsetNnz+nodeOrdinal) = 1.0; // entries (constraint coefficients)

        tValues(offsetChild+nodeOrdinal) = tValue; // constraint RHS

    }, "Tie constraint data");
}

/****************************************************************************/
void Plato::TieMultipointConstraint::
updateLengths(const Omega_h::MeshSets& aMeshSets,
           OrdinalType& lengthChild,
           OrdinalType& lengthParent,
           OrdinalType& lengthNnz)
/****************************************************************************/
{
    auto& tNodeSets = aMeshSets[Omega_h::NODE_SET];

    // parse child nodes
    auto tChildNodeSetsIter = tNodeSets.find(this->mChildNodeSet);
    auto tChildNodeLids = (tChildNodeSetsIter->second);
    auto tNumberChildNodes = tChildNodeLids.size();
    
    // parse parent nodes
    auto tParentNodeSetsIter = tNodeSets.find(this->mParentNodeSet);
    auto tParentNodeLids = (tParentNodeSetsIter->second);
    auto tNumberParentNodes = tParentNodeLids.size();

    // Check that the number of child and parent nodes match
    if (tNumberChildNodes != tNumberParentNodes)
    {
        std::ostringstream tMsg;
        tMsg << "CHILD AND PARENT NODESETS FOR TIE CONSTRAINT NOT OF EQUAL LENGTH. \n";
        THROWERR(tMsg.str())
    }

    lengthChild += tNumberChildNodes;
    lengthParent += tNumberParentNodes;
    lengthNnz += tNumberChildNodes;
}

}
// namespace Plato

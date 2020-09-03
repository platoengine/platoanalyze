/*
 * TieMultipointConstraint.hpp
 *
 *  Created on: May 26, 2020
 */

/* #pragma once */

#include "AnalyzeMacros.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Derived class for tie multipoint constraint
 *
**********************************************************************************/
template<typename SimplexPhysicsType>
class TieMultipointConstraint : public Plato::MultipointConstraint<SimplexPhysicsType>
{
public:
    TieMultipointConstraint(const std::string & aName, Teuchos::ParameterList & aParam);

    virtual ~TieMultipointConstraint(){}

    /*!
     \brief Get constraint matrix and RHS data.
     \param aMeshSets Omega_h mesh sets that contains nodeset data.
     \param mpcRowMap CRS-style rowMap for constraint data.
     \param mpcColumnIndices CRS-style columnIndices for constraint data.
     \param mpcEntries CRS-style entries for constraint data.
     \param mpcValues Value list for constraint RHS.
     \param offsetChild Starting location in rowMap/RHS where constrained nodes/values will be added.
     \param offsetNnz Starting location in columnIndices/entries where constraining nodes/coefficients will be added.
     */
    virtual void get(const Omega_h::MeshSets& aMeshSets,
                     LocalOrdinalVector & mpcChildNodes,
                     LocalOrdinalVector & mpcParentNodes,
                     Plato::CrsMatrixType::RowMapVector & mpcRowMap,
                     Plato::CrsMatrixType::OrdinalVector & mpcColumnIndices,
                     Plato::CrsMatrixType::ScalarVector & mpcEntries,
                     ScalarVector & mpcValues,
                     OrdinalType offsetChild,
                     OrdinalType offsetParent,
                     OrdinalType offsetNnz) override;
    
    // ! Get number of nodes in the constrained nodeset.
    virtual void updateLengths(const Omega_h::MeshSets& aMeshSets,
                              OrdinalType& lengthChild,
                              OrdinalType& lengthParent,
                              OrdinalType& lengthNnz) override;

private:
    std::string mChildNodeSet;
    std::string mParentNodeSet;

    /* Plato::OrdinalType mChildNodeSet; */
    /* Plato::OrdinalType mParentNodeSet; */

    Plato::Scalar mValue;

};
// class TieMultipointConstraint

/****************************************************************************/
template<typename SimplexPhysicsType>
Plato::TieMultipointConstraint<SimplexPhysicsType>::
TieMultipointConstraint(const std::string & aName, Teuchos::ParameterList & aParam) :
    Plato::MultipointConstraint<SimplexPhysicsType>(aName)
/****************************************************************************/
{
    /* mChildNodeSet = "Child"; */
    /* mParentNodeSet = "Parent"; */

    mChildNodeSet = aParam.get<std::string>("Child");
    mParentNodeSet = aParam.get<std::string>("Parent");

    /* mChildNodeSet = aParam.get<Plato::OrdinalType>("Child"); */
    /* mParentNodeSet = aParam.get<Plato::OrdinalType>("Parent"); */

    mValue = aParam.get<Plato::Scalar>("Value");
}

/****************************************************************************/
template<typename SimplexPhysicsType>
void Plato::TieMultipointConstraint<SimplexPhysicsType>::
get(const Omega_h::MeshSets& aMeshSets,
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
    constexpr Plato::OrdinalType dofsPerNode = SimplexPhysicsType::mNumDofsPerNode;
    auto& tNodeSets = aMeshSets[Omega_h::NODE_SET];
    auto tValue = this->mValue;
    
    // parse child nodes
    auto tChildNodeSetsIter = tNodeSets.find(this->mChildNodeSet);
    auto tChildNodeLids = (tChildNodeSetsIter->second);
    auto tNumberChildNodes = tChildNodeLids.size();

    /* auto tChildNodeLids = mChildNodeSet; */
    /* auto tNumberChildNodes = 1; */
    
    // parse parent nodes
    auto tParentNodeSetsIter = tNodeSets.find(this->mParentNodeSet);
    auto tParentNodeLids = (tParentNodeSetsIter->second);
    auto tNumberParentNodes = tParentNodeLids.size();

    /* auto tParentNodeLids = mParentNodeSet; */
    /* auto tNumberParentNodes = 1; */

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

        /* tChildNodes(offsetChild+nodeOrdinal) = tChildNodeLids; // child node ID */
        /* tParentNodes(offsetParent+nodeOrdinal) = tParentNodeLids; // parent node ID */

        tRowMap(offsetChild+nodeOrdinal) = offsetChild + nodeOrdinal; // row map
        tColumnIndices(offsetNnz+nodeOrdinal) = offsetParent + nodeOrdinal; // column indices (local parent node ID)
        tEntries(offsetNnz+nodeOrdinal) = 1.0; // entries (constraint coefficients)

        tValues(offsetChild+nodeOrdinal) = tValue; // constraint RHS

    }, "Tie constraint data");
}

/****************************************************************************/
template<typename SimplexPhysicsType>
void Plato::TieMultipointConstraint<SimplexPhysicsType>::
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

    /* auto tChildNodeLids = mChildNodeSet; */
    /* auto tNumberChildNodes = 1; */
    
    // parse parent nodes
    auto tParentNodeSetsIter = tNodeSets.find(this->mParentNodeSet);
    auto tParentNodeLids = (tParentNodeSetsIter->second);
    auto tNumberParentNodes = tParentNodeLids.size();

    /* auto tParentNodeLids = mParentNodeSet; */
    /* auto tNumberParentNodes = 1; */

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

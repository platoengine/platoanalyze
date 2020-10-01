/*
 * PbcMultipointConstraint.hpp
 *
 *  Created on: September 22, 2020
 */

#include "PbcMultipointConstraint.hpp"

namespace Plato
{

/****************************************************************************/
Plato::PbcMultipointConstraint::
PbcMultipointConstraint(const Omega_h::Mesh & aMesh,
                        const Omega_h::MeshSets & aMeshSets,
                        const std::string & aName, 
                        Teuchos::ParameterList & aParam) :
                        Plato::MultipointConstraint(aName)
/****************************************************************************/
{
    // parse RHS value
    mValue = aParam.get<Plato::Scalar>("Value");

    // parse child node set
    auto& tNodeSets = aMeshSets[Omega_h::NODE_SET];
    std::string tChildNodeSet = aParam.get<std::string>("Child");
    auto tChildNodeSetsIter = tNodeSets.find(tChildNodeSet);
    auto tChildNodeLids = (tChildNodeSetsIter->second);
    auto tNumberChildNodes = tChildNodeLids.size();
    
    // Fill in child nodes
    Kokkos::resize(mChildNodes, tNumberChildNodes);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        mChildNodes(nodeOrdinal) = tChildNodeLids[nodeOrdinal]; // child node ID
    }, "Child node IDs");

    // construct translation math map object
    Plato::Geometry::Translation<Plato::Scalar> tTranslation(aParam);
    
    // map child nodes
    Plato::ScalarMultiVector tChildNodeLocations       ("child node locations",        Plato::Geometry::cSpaceDim, tNumberChildNodes);
    Plato::ScalarMultiVector tMappedChildNodeLocations ("mapped child node locations", Plato::Geometry::cSpaceDim, tNumberChildNodes);

    this->mapChildVertexLocations(aMesh, tTranslation, tChildNodeLocations, tMappedChildNodeLocations);
    
    // find elements that contain mapped child node locations
    Plato::LocalOrdinalVector tParentElements("mapped elements", tNumberChildNodes);
    Plato::Geometry::findParentElements<Plato::Scalar>(aMesh, tChildNodeLocations, tMappedChildNodeLocations, tParentElements);

    // get global IDs of unique parent nodes
    Plato::LocalOrdinalVector tParentGlobalLocalMap;
    this->getUniqueParentNodes(aMesh, tParentElements, tParentGlobalLocalMap);
    
    // fill in mpc matrix values
    this->setMatrixValues(aMesh, tParentElements, tMappedChildNodeLocations, tParentGlobalLocalMap);
}

/****************************************************************************/
void Plato::PbcMultipointConstraint::
get(LocalOrdinalVector & mpcChildNodes,
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
    auto tValue = mValue;
    auto tNumberChildNodes = mChildNodes.size();
    auto tNumberParentNodes = mParentNodes.size();

    // fill in parent nodes
    auto tParentNodes = mpcParentNodes;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberParentNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        tParentNodes(offsetParent+nodeOrdinal) = mParentNodes(nodeOrdinal); // parent node ID
    }, "parent nodes");

    // fill in chuld nodes and constraint info
    auto tChildNodes = mpcChildNodes;
    auto tRowMap = mpcRowMap;
    auto tColumnIndices = mpcColumnIndices;
    auto tEntries = mpcEntries;
    auto tValues = mpcValues;

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        tChildNodes(offsetChild+nodeOrdinal) = mChildNodes(nodeOrdinal); // child node ID

        auto tRowStart = mMpcMatrix->mRowMap(nodeOrdinal);
        auto tRowEnd = mMpcMatrix->mRowMap(nodeOrdinal+1);
        auto tNnz = tRowEnd - tRowStart;
        for(Plato::OrdinalType entryOrdinal = tRowStart; entryOrdinal<tRowEnd; entryOrdinal++)
        {
            tColumnIndices(offsetNnz+entryOrdinal) = offsetParent + mMpcMatrix->mColumnIndices(entryOrdinal); // column indices
            tEntries(offsetNnz+entryOrdinal) = mMpcMatrix->mEntries(entryOrdinal); // entries (constraint coefficients)
        }

        tRowMap(offsetChild+nodeOrdinal) = offsetNnz + mMpcMatrix->mRowMap(nodeOrdinal); // row map

        tValues(offsetChild+nodeOrdinal) = tValue; // constraint RHS
        
    }, "child nodes, mpc matrix, and rhs values");

}

/****************************************************************************/
void Plato::PbcMultipointConstraint::
updateLengths(OrdinalType& lengthChild,
              OrdinalType& lengthParent,
              OrdinalType& lengthNnz)
/****************************************************************************/
{
    auto tNumberChildNodes = mChildNodes.size();
    auto tNumberParentNodes = mParentNodes.size();
    auto tNumberNonzero = mMpcMatrix->mRowMap(tNumberChildNodes+1);

    lengthChild += tNumberChildNodes;
    lengthParent += tNumberParentNodes;
    lengthNnz += tNumberNonzero;
}

/****************************************************************************/
void Plato::PbcMultipointConstraint::
mapChildVertexLocations(const Omega_h::Mesh & aMesh,
                        const Plato::Geometry::Translation<Plato::Scalar> & aTranslation,
                        Plato::ScalarMultiVector aLocations,
                        Plato::ScalarMultiVector aMappedLocations)
/****************************************************************************/
{
    auto tCoords = aMesh.coords();
    auto tNumberChildNodes = mChildNodes.size();

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        Plato::OrdinalType childNode = mChildNodes(nodeOrdinal);
        for(size_t iDim=0; iDim < Plato::Geometry::cSpaceDim; ++iDim)
        {
            aLocations(iDim, nodeOrdinal) = tCoords[childNode*Plato::Geometry::cSpaceDim+iDim];
        }
        aTranslation(nodeOrdinal, aLocations, aMappedLocations);
    }, "get verts and apply map");
}

/****************************************************************************/
void Plato::PbcMultipointConstraint::
getUniqueParentNodes(const Omega_h::Mesh & aMesh,
                     LocalOrdinalVector aParentElements,
                     LocalOrdinalVector aParentGlobalLocalMap)
/****************************************************************************/
{
    auto tNVerts = aMesh.nverts();
    auto tNumberParentElements = aParentElements.size();
    auto tNVertsPerElem = Plato::Geometry::cNVertsPerElem;
    auto tCells2Nodes = aMesh.ask_elem_verts();

    // initialize array for storing parent element vertex ordinals
    Plato::LocalOrdinalVector tNodeCounter("parent node counting", tNVerts);
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(0), tNodeCounter);

    // fill in parent element vertex ordinals
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberParentElements), LAMBDA_EXPRESSION(Plato::OrdinalType iElemOrdinal)
    {
        tElement = aParentElements(iElemOrdinal); 
        if(tElement == -2)
        {
            std::ostringstream tMsg;
            tMsg << "NO PARENT ELEMENT COULD BE FOUND FOR CHILD NODE IN PBC MULTIPOINT CONSTRAINT. \n";
            THROWERR(tMsg.str())
        }
        if(tElement == -1)
        {
            std::ostringstream tMsg;
            tMsg << "CHILD NODE COULD NOT BE MAPPED IN PBC MULTIPOINT CONSTRAINT. \n";
            THROWERR(tMsg.str())
        }
        for(Plato::OrdinalType iVertOrdinal=0; iVertOrdinal < tNVertsPerElem; ++iVertOrdinal)
        {
            Plato::OrdinalType tVertIndex = tCells2Nodes[tElement*tNVertsPerElem + iVertOrdinal];
            tNodeCounter(tVertIndex) = 1;
        }
    }, "mark 1 for parent element vertices");
    
    // get number of unique parent nodes
    Plato::OrdinalType tSum(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0,tNVerts),
    LAMBDA_EXPRESSION(const Plato::OrdinalType& aOrdinal, Plato::OrdinalType & aUpdate)
    {
        aUpdate += tNodeCounter(aOrdinal);
    }, tSum);
    Kokkos::resize(mParentNodes,tSum);

    // fill in unique parent nodes
    Plato::OrdinalType tOffset(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<Plato::OrdinalType>(0,tNVerts),
    KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        const Plato::OrdinalType tVal = tNodeCounter(iOrdinal);
        if( tIsFinal ) 
        { 
            mParentNodes(aUpdate) = iOrdinal; 
        }
        aUpdate += tVal;
    }, tOffset);

    // create map from global node ID to local parent node ID
    Kokkos::resize(aParentGlobalLocalMap,tNVerts);
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(-1), aParentGlobalLocalMap);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tSum), LAMBDA_EXPRESSION(Plato::OrdinalType parentOrdinal)
    {
        Plato::OrdinalType tGlobalVertId = mParentNodes(parentOrdinal);
        aParentGlobalLocalMap(tGlobalVertId) = parentOrdinal;
    }, "map from global vertex ID to local parent node ID");
}

/****************************************************************************/
void Plato::PbcMultipointConstraint::
setMatrixValues(const Omega_h::Mesh & aMesh,
                LocalOrdinalVector aParentElements,
                Plato::ScalarMultiVector aMappedLocations,
                LocalOrdinalVector aParentGlobalLocalMap)
/****************************************************************************/
{
    auto tCells2Nodes = aMesh.ask_elem_verts();

    auto tNumChildNodes = mChildNodes.size();
    auto tNumParentNodes = mParentNodes.size();
    
    // build rowmap
    Plato::CrsMatrixType::RowMapVector tRowMap("row map", tNumChildNodes+1);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(OrdinalT iRowOrdinal)
    {
        tRowMap(iRowOrdinal) = Plato::Geometry::cNVertsPerElem;
    }, "nonzeros");

    Plato::OrdinalType tNumEntries(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<OrdinalT>(0,tNumChildNodes+1),
    KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        const Plato::OrdinalType tVal = tRowMap(iOrdinal);
        if( tIsFinal )
        {
          tRowMap(iOrdinal) = aUpdate;
        }
        aUpdate += tVal;
    }, tNumEntries);
    
    // determine column map and entries
    Plato::CrsMatrixType::OrdinalVector tColMap("column indices", tNumEntries);
    Plato::CrsMatrixType::ScalarVector tEntries("matrix entries", tNumEntries);

    Plato::Geometry::GetBasis<Plato::Scalar> tGetBasis(aMesh);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType iRowOrdinal)
    {

        auto iElemOrdinal = aParentElements(iRowOrdinal);

        // global element node IDs
        Plato::OrdinalType i0 = tCells2Nodes[iElemOrdinal*Plato::Geometry::cNVertsPerElem  ];
        Plato::OrdinalType i1 = tCells2Nodes[iElemOrdinal*Plato::Geometry::cNVertsPerElem+1];
        Plato::OrdinalType i2 = tCells2Nodes[iElemOrdinal*Plato::Geometry::cNVertsPerElem+2];
        Plato::OrdinalType i3 = tCells2Nodes[iElemOrdinal*Plato::Geometry::cNVertsPerElem+3];

        // local parent node IDs
        Plato::OrdinalType iP0 = aParentGlobalLocalMap(i0);
        Plato::OrdinalType iP1 = aParentGlobalLocalMap(i1);
        Plato::OrdinalType iP2 = aParentGlobalLocalMap(i2);
        Plato::OrdinalType iP3 = aParentGlobalLocalMap(i3);

        if(iP0 == -1 || iP1 == -1 || iP2 == -1 || iP3 == -1)
        {
            std::ostringstream tMsg;
            tMsg << "PARENT ELEMENT NODE ID NOT RECOGNIZED AS PARENT NODE IN PBC MULTIPOINT CONSTRAINT. \n";
            THROWERR(tMsg.str())
        }

        // basis function values
        Plato::Scalar tBasisValues[Plato::Geometry::cNVertsPerElem];
        tGetBasis(aMappedLocations, iRowOrdinal, iElemOrdinal, tBasisValues);

        // fill in colmap and entries
        auto iEntryOrdinal = tRowMap(iRowOrdinal);

        tColMap(iEntryOrdinal  ) = iP0;
        tColMap(iEntryOrdinal+1) = iP1;
        tColMap(iEntryOrdinal+2) = iP2;
        tColMap(iEntryOrdinal+3) = iP3;

        tEntries(iEntryOrdinal  ) = tBasisValues[0];
        tEntries(iEntryOrdinal+1) = tBasisValues[1];
        tEntries(iEntryOrdinal+2) = tBasisValues[2];
        tEntries(iEntryOrdinal+3) = tBasisValues[3];

    }, "colmap and entries");

    // fill in mpc matrix
    mMpcMatrix = Teuchos::rcp( new Plato::CrsMatrixType(tRowMap, tColMap, tEntries, tNumChildNodes, tNumParentNodes, 1, 1) );

}

}
// namespace Plato

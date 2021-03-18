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
PbcMultipointConstraint(const Plato::SpatialModel & aSpatialModel,
                        const std::string & aName, 
                        Teuchos::ParameterList & aParam) :
                        Plato::MultipointConstraint(aName)
/****************************************************************************/
{
    // Ensure mesh is 3D
    auto tSpaceDim = aSpatialModel.Mesh.dim();
    if ( tSpaceDim != Plato::Geometry::cSpaceDim )
    {
        std::ostringstream tMsg;
        tMsg << "INVALID MESH DIMENSION: PBC MULTIPOINT CONSTRAINTS ONLY IMPLEMENTED FOR 3D MESHES.";
        THROWERR(tMsg.str())
    }
    //
    // parse translation vector
    bool tIsVector = aParam.isType<Teuchos::Array<Plato::Scalar>>("Vector");

    Plato::Scalar tTranslationX;
    Plato::Scalar tTranslationY;
    Plato::Scalar tTranslationZ;
    if ( tIsVector )
    {
        auto tVector = aParam.get<Teuchos::Array<Plato::Scalar>>("Vector");
        tTranslationX = tVector[0];
        tTranslationY = tVector[1];
        tTranslationZ = tVector[2];
    }
    else
    {
        std::ostringstream tMsg;
        tMsg << "TRANSLATION VECTOR FOR PBC MULTIPOINT CONSTRAINT NOT PARSED: CHECK INPUT PARAMETER KEYWORDS.";
        THROWERR(tMsg.str())
    }

    auto tLength = tTranslationX * tTranslationX
                 + tTranslationY * tTranslationY
                 + tTranslationZ * tTranslationZ;

    if( tLength == 0.0 )
    {
        std::ostringstream tMsg;
        tMsg << "TRANSLATION VECTOR FOR PBC MULTIPOINT CONSTRAINT HAS NO LENGTH.";
        THROWERR(tMsg.str())
    }
      
    // parse RHS value
    mValue = aParam.get<Plato::Scalar>("Value");

    // parse child node set
    auto& tNodeSets = aSpatialModel.MeshSets[Omega_h::NODE_SET];
    std::string tChildNodeSet = aParam.get<std::string>("Child");
    auto tChildNodeSetsIter = tNodeSets.find(tChildNodeSet);
    if( tChildNodeSetsIter == tNodeSets.end() )
    {
        std::ostringstream tMsg;
        tMsg << "Could not find Node Set with name = '" << tChildNodeSet.c_str()
                << "'. Node Set is not defined in input geometry/mesh file.\n";
        THROWERR(tMsg.str())
    }
    auto tChildNodeLids = (tChildNodeSetsIter->second);
    auto tNumberChildNodes = tChildNodeLids.size();
    
    // Fill in child nodes
    Kokkos::resize(mChildNodes, tNumberChildNodes);

    this->updateNodesets(tNumberChildNodes, tChildNodeLids);
    
    // map child nodes
    Plato::ScalarMultiVector tChildNodeLocations       ("child node locations",        Plato::Geometry::cSpaceDim, tNumberChildNodes);
    Plato::ScalarMultiVector tMappedChildNodeLocations ("mapped child node locations", Plato::Geometry::cSpaceDim, tNumberChildNodes);

    this->mapChildVertexLocations(aSpatialModel.Mesh, tTranslationX, tTranslationY, tTranslationZ, tChildNodeLocations, tMappedChildNodeLocations);

    // get parent domain element data
    std::string tParentDomainName = aParam.get<std::string>("Parent");
    Omega_h::LOs tDomainCellMap;
    bool tFindName = 0;
    for(auto& tDomain : aSpatialModel.Domains)
    {
        auto tName = tDomain.getDomainName();
        if( tName == tParentDomainName )
            tDomainCellMap = tDomain.cellOrdinals();
            tFindName = 1;
    }
    if( tFindName == 0 )
    {
        std::ostringstream tMsg;
        tMsg << "PARENT DOMAIN FOR PBC MULTIPOINT CONSTRAINT NOT FOUND.";
        THROWERR(tMsg.str())
    }
    
    // find elements that contain mapped child node locations (in specified domain)
    Plato::LocalOrdinalVector tParentElements("mapped elements", tNumberChildNodes);
    Plato::Geometry::findParentElements<Plato::Scalar>(aSpatialModel.Mesh, tDomainCellMap, tChildNodeLocations, tMappedChildNodeLocations, tParentElements);

    // get global IDs of unique parent nodes
    Plato::LocalOrdinalVector tParentGlobalLocalMap;
    this->getUniqueParentNodes(aSpatialModel.Mesh, tParentElements, tParentGlobalLocalMap);
    
    // fill in mpc matrix values
    this->setMatrixValues(aSpatialModel.Mesh, tParentElements, tMappedChildNodeLocations, tParentGlobalLocalMap);
}

/****************************************************************************/
void Plato::PbcMultipointConstraint::
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
    auto tNumberParentNodes = mParentNodes.size();

    // fill in parent nodes
    auto tMpcParentNodes = aMpcParentNodes;
    auto tParentNodes = mParentNodes;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberParentNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        tMpcParentNodes(aOffsetParent+nodeOrdinal) = tParentNodes(nodeOrdinal); // parent node ID
    }, "parent nodes");

    // fill in chuld nodes and constraint info
    const auto& tMpcRowMap = mMpcMatrix->rowMap();
    const auto& tMpcColumnIndices = mMpcMatrix->columnIndices();
    const auto& tMpcEntries = mMpcMatrix->entries();
      
    auto tMpcChildNodes = aMpcChildNodes;
    auto tRowMap = aMpcRowMap;
    auto tColumnIndices = aMpcColumnIndices;
    auto tEntries = aMpcEntries;
    auto tValues = aMpcValues;

    auto tChildNodes = mChildNodes;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        tMpcChildNodes(aOffsetChild+nodeOrdinal) = tChildNodes(nodeOrdinal); // child node ID

        auto tRowStart = tMpcRowMap(nodeOrdinal);
        auto tRowEnd = tMpcRowMap(nodeOrdinal+1);
        for(Plato::OrdinalType entryOrdinal = tRowStart; entryOrdinal<tRowEnd; entryOrdinal++)
        {
            tColumnIndices(aOffsetNnz + entryOrdinal) = aOffsetParent + tMpcColumnIndices(entryOrdinal); // column indices
            tEntries(aOffsetNnz + entryOrdinal) = tMpcEntries(entryOrdinal); // entries (constraint coefficients)
        }

        tRowMap(aOffsetChild + nodeOrdinal) = aOffsetNnz + tMpcRowMap(nodeOrdinal); // row map
        tRowMap(aOffsetChild + nodeOrdinal + 1) = aOffsetNnz + tMpcRowMap(nodeOrdinal + 1); // row map

        tValues(aOffsetChild + nodeOrdinal) = tValue; // constraint RHS
        
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
    const auto& tMpcColMap = mMpcMatrix->columnIndices();
    auto tNumberNonzero = tMpcColMap.size();

    lengthChild += tNumberChildNodes;
    lengthParent += tNumberParentNodes;
    lengthNnz += tNumberNonzero;

}

/****************************************************************************/
void Plato::PbcMultipointConstraint::
updateNodesets(const OrdinalType& tNumberChildNodes,
               const Omega_h::LOs& tChildNodeLids)
/****************************************************************************/
{
    auto tChildNodes = mChildNodes;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        tChildNodes(nodeOrdinal) = tChildNodeLids[nodeOrdinal]; // child node ID
    }, "Child node IDs");
}

/****************************************************************************/
void Plato::PbcMultipointConstraint::
mapChildVertexLocations(Omega_h::Mesh & aMesh,
                        const Plato::Scalar aTranslationX,
                        const Plato::Scalar aTranslationY,
                        const Plato::Scalar aTranslationZ,
                        Plato::ScalarMultiVector & aLocations,
                        Plato::ScalarMultiVector & aMappedLocations)
/****************************************************************************/
{
    auto tCoords = aMesh.coords();
    auto tNumberChildNodes = mChildNodes.size();

    auto tChildNodes = mChildNodes;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        Plato::OrdinalType childNode = tChildNodes(nodeOrdinal);
        for(size_t iDim=0; iDim < Plato::Geometry::cSpaceDim; ++iDim)
        {
            aLocations(iDim, nodeOrdinal) = tCoords[childNode*Plato::Geometry::cSpaceDim+iDim];
        }
        // perform translation mapping
        aMappedLocations(0, nodeOrdinal) = aLocations(0, nodeOrdinal) + aTranslationX;
        aMappedLocations(1, nodeOrdinal) = aLocations(1, nodeOrdinal) + aTranslationY;
        aMappedLocations(2, nodeOrdinal) = aLocations(2, nodeOrdinal) + aTranslationZ;
    }, "get verts and apply map");
}

/****************************************************************************/
void Plato::PbcMultipointConstraint::
getUniqueParentNodes(Omega_h::Mesh & aMesh,
                     LocalOrdinalVector & aParentElements,
                     LocalOrdinalVector & aParentGlobalLocalMap)
/****************************************************************************/
{
    auto tNVerts = aMesh.nverts();
    auto tNumberParentElements = aParentElements.size();
    auto tNVertsPerElem = Plato::Geometry::cNVertsPerElem;
    auto tCells2Nodes = aMesh.ask_elem_verts();

    // initialize array for storing parent element vertex ordinals
    Plato::LocalOrdinalVector tNodeCounter("parent node counting", tNVerts);
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(0), tNodeCounter);

    // check for missing parent elements or parent nodes
    Plato::OrdinalType tNumMissingParent(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumberParentElements),
    LAMBDA_EXPRESSION(const Plato::OrdinalType& aElemOrdinal, Plato::OrdinalType & aUpdate)
    {
        if ( aParentElements(aElemOrdinal) == -2 ) 
        {  
            aUpdate++;
        }
    }, tNumMissingParent);
    if ( tNumMissingParent > 0 )
    {
        std::ostringstream tMsg;
        tMsg << "NO PARENT ELEMENT COULD BE FOUND FOR AT LEAST ONE CHILD NODE IN PBC MULTIPOINT CONSTRAINT. \n";
        THROWERR(tMsg.str())
    }

    Plato::OrdinalType tNumMissingMap(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumberParentElements),
    LAMBDA_EXPRESSION(const Plato::OrdinalType& aElemOrdinal, Plato::OrdinalType & aUpdate)
    {
        if ( aParentElements(aElemOrdinal) == -1 ) 
        {  
            aUpdate++;
        }
    }, tNumMissingMap);
    if ( tNumMissingMap > 0 )
    {
        std::ostringstream tMsg;
        tMsg << "AT LEAST ONE CHILD NODE COULD NOT BE MAPPED IN PBC MULTIPOINT CONSTRAINT. \n";
        THROWERR(tMsg.str())
    }

    // fill in parent element vertex ordinals
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumberParentElements), LAMBDA_EXPRESSION(Plato::OrdinalType iElemOrdinal)
    {
        Plato::OrdinalType tElement = aParentElements(iElemOrdinal); 
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
    auto tParentNodes = mParentNodes;
    Kokkos::parallel_scan (Kokkos::RangePolicy<Plato::OrdinalType>(0,tNVerts),
    KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        const Plato::OrdinalType tVal = tNodeCounter(iOrdinal);
        if( tIsFinal ) 
        { 
            tParentNodes(aUpdate) = iOrdinal; 
        }
        aUpdate += tVal;
    }, tOffset);

    // create map from global node ID to local parent node ID
    Kokkos::resize(aParentGlobalLocalMap,tNVerts);
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(-1), aParentGlobalLocalMap);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tSum), LAMBDA_EXPRESSION(Plato::OrdinalType parentOrdinal)
    {
        Plato::OrdinalType tGlobalVertId = tParentNodes(parentOrdinal);
        aParentGlobalLocalMap(tGlobalVertId) = parentOrdinal;
    }, "map from global vertex ID to local parent node ID");
}

/****************************************************************************/
void Plato::PbcMultipointConstraint::
setMatrixValues(Omega_h::Mesh & aMesh,
                LocalOrdinalVector & aParentElements,
                Plato::ScalarMultiVector & aMappedLocations,
                LocalOrdinalVector & aParentGlobalLocalMap)
/****************************************************************************/
{
    auto tCells2Nodes = aMesh.ask_elem_verts();

    auto tNumChildNodes = mChildNodes.size();
    auto tNumParentNodes = mParentNodes.size();
    
    // build rowmap
    Plato::CrsMatrixType::RowMapVector tRowMap("row map", tNumChildNodes+1);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType iRowOrdinal)
    {
        tRowMap(iRowOrdinal) = Plato::Geometry::cNVertsPerElem;
    }, "nonzeros");

    Plato::OrdinalType tNumEntries(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumChildNodes+1),
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

#include "MultipointConstraints.hpp"

namespace Plato
{

/****************************************************************************/
MultipointConstraints::MultipointConstraints(const Plato::SpatialModel & aSpatialModel,
                                             const OrdinalType & aNumDofsPerNode, 
                                             Teuchos::ParameterList & aParams) :
                                             MPCs(),
                                             mNumNodes(aSpatialModel.Mesh.nverts()),
                                             mNumDofsPerNode(aNumDofsPerNode),
                                             mTransformMatrix(Teuchos::null),
                                             mTransformMatrixTranspose(Teuchos::null)
/****************************************************************************/
{
    for(Teuchos::ParameterList::ConstIterator tIndex = aParams.begin(); tIndex != aParams.end(); ++tIndex)
    {
        const Teuchos::ParameterEntry & tEntry = aParams.entry(tIndex);
        const std::string & tMyName = aParams.name(tIndex);

        TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error, " Parameter in Multipoint Constraints block not valid. Expect lists only.");

        Teuchos::ParameterList& tSublist = aParams.sublist(tMyName);
        Plato::MultipointConstraintFactory tMultipointConstraintFactory(tSublist);

        std::shared_ptr<MultipointConstraint> tMyMPC = tMultipointConstraintFactory.create(aSpatialModel, tMyName);
        MPCs.push_back(tMyMPC);
    }
}

/****************************************************************************/
void MultipointConstraints::get(Teuchos::RCP<Plato::CrsMatrixType> & mpcMatrix,
                                ScalarVector & mpcValues)
/****************************************************************************/
{
    OrdinalType numChildNodes(0);
    OrdinalType numParentNodes(0);
    OrdinalType numConstraintNonzeros(0);
    for(std::shared_ptr<MultipointConstraint> & mpc : MPCs)
        mpc->updateLengths(numChildNodes, numParentNodes, numConstraintNonzeros);

    Kokkos::resize(mChildNodes, numChildNodes);
    Kokkos::resize(mParentNodes, numParentNodes);
    Kokkos::resize(mpcValues, numChildNodes);
    Plato::CrsMatrixType::RowMapVector mpcRowMap("row map", numChildNodes+1);
    Plato::CrsMatrixType::OrdinalVector mpcColumnIndices("column indices", numConstraintNonzeros);
    Plato::CrsMatrixType::ScalarVector mpcEntries("matrix entries", numConstraintNonzeros);

    auto tChildNodes = mChildNodes;
    auto tParentNodes = mParentNodes;

    OrdinalType offsetChild(0);
    OrdinalType offsetParent(0);
    OrdinalType offsetNnz(0);
    for(std::shared_ptr<MultipointConstraint> & mpc : MPCs)
    {
        mpc->get(tChildNodes, tParentNodes, mpcRowMap, mpcColumnIndices, mpcEntries, mpcValues, offsetChild, offsetParent, offsetNnz);
        mpc->updateLengths(offsetChild, offsetParent, offsetNnz);
    }

    // Build full CRS matrix to return
    mpcMatrix = Teuchos::rcp( new Plato::CrsMatrixType(mpcRowMap, mpcColumnIndices, mpcEntries, numChildNodes, numParentNodes, 1, 1) );
}

/****************************************************************************/
void MultipointConstraints::getMaps(LocalOrdinalVector & nodeTypes,
                                    LocalOrdinalVector & nodeConNum)
/****************************************************************************/
{
    OrdinalType tNumChildNodes = mChildNodes.size();

    Kokkos::resize(nodeTypes, mNumNodes);   // new ID after condensation
    Kokkos::resize(nodeConNum, mNumNodes);  // constraint number for child nodes
    Plato::LocalOrdinalVector tCondensedNodeCounter("count number of nodes after condensation", mNumNodes);

    Plato::blas1::fill(static_cast<Plato::OrdinalType>(-1), nodeTypes);  // child nodes marked with -1
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(-1), nodeConNum); // non-child nodes marked with -1
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(1), tCondensedNodeCounter); // non-child nodes marked with 1

    auto tChildNodes = mChildNodes;

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType childOrdinal)
    {
        OrdinalType childNode = tChildNodes(childOrdinal);
        tCondensedNodeCounter(childNode) = 0; // mark child DOF with 0
        nodeConNum(childNode) = childOrdinal;
    }, "Set child node type and constraint number");

    // assign condensed DOF ordinals
    Plato::OrdinalType tNumCondensedDofs(0);

    Kokkos::parallel_scan (Kokkos::RangePolicy<Plato::OrdinalType>(0,mNumNodes),
    KOKKOS_LAMBDA (const Plato::OrdinalType& nodeOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        const Plato::OrdinalType tVal = tCondensedNodeCounter(nodeOrdinal);
        if (tIsFinal && tVal) // non-child
        {  
            nodeTypes(nodeOrdinal) = aUpdate; 
        }
        aUpdate += tVal;
    }, tNumCondensedDofs);
}

/****************************************************************************/
void MultipointConstraints::assembleTransformMatrix(const Teuchos::RCP<Plato::CrsMatrixType> & aMpcMatrix,
                                                                        const LocalOrdinalVector & aNodeTypes,
                                                                        const LocalOrdinalVector & aNodeConNum)
/****************************************************************************/
{
    OrdinalType tBlockSize = mNumDofsPerNode*mNumDofsPerNode;

    const auto& tMpcRowMap = aMpcMatrix->rowMap();
    const auto& tMpcColumnIndices = aMpcMatrix->columnIndices();
    const auto& tMpcEntries = aMpcMatrix->entries();

    OrdinalType tNumChildNodes = tMpcRowMap.size() - 1;
    OrdinalType tNumParentNodes = mParentNodes.size();
    OrdinalType tMpcNnz = tMpcEntries.size();
    OrdinalType tOutNnz = tBlockSize*((mNumNodes - tNumChildNodes) + tMpcNnz);

    Plato::CrsMatrixType::RowMapVector outRowMap("transform matrix row map", mNumNodes+1);
    Plato::CrsMatrixType::OrdinalVector outColumnIndices("transform matrix column indices", tOutNnz);
    Plato::CrsMatrixType::ScalarVector outEntries("transform matrix entries", tOutNnz);

    // build row map
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, mNumNodes), LAMBDA_EXPRESSION(Plato::OrdinalType iRowOrdinal)
    {
        OrdinalType nodeType = aNodeTypes(iRowOrdinal);
        if(nodeType == -1) // Child Node
        {
            OrdinalType conOrdinal = aNodeConNum(iRowOrdinal);
            OrdinalType tConNnz = tMpcRowMap(conOrdinal + 1) - tMpcRowMap(conOrdinal);
            outRowMap(iRowOrdinal) = tConNnz; 
        }
        else 
        {
            outRowMap(iRowOrdinal) = 1;
        }
    }, "row map");

    OrdinalType tNumEntries(0);
    Kokkos::parallel_scan (Kokkos::RangePolicy<Plato::OrdinalType>(0,mNumNodes+1),
    KOKKOS_LAMBDA (const Plato::OrdinalType& iOrdinal, Plato::OrdinalType& aUpdate, const bool& tIsFinal)
    {
        const OrdinalType tVal = outRowMap(iOrdinal);
        if( tIsFinal )
        {
          outRowMap(iOrdinal) = aUpdate;
        }
        aUpdate += tVal;
    }, tNumEntries);

    // build col map and entries
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), outEntries);

    auto tParentNodes = mParentNodes;
    auto tNumDofsPerNode = mNumDofsPerNode;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, mNumNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        OrdinalType tColMapOrdinal = outRowMap(nodeOrdinal);
        OrdinalType nodeType = aNodeTypes(nodeOrdinal);

        if(nodeType == -1) // Child Node
        {
            OrdinalType conOrdinal = aNodeConNum(nodeOrdinal);
            OrdinalType tConRowStart = tMpcRowMap(conOrdinal);
            OrdinalType tConRowEnd = tMpcRowMap(conOrdinal + 1);

            for(OrdinalType parentOrdinal=tConRowStart; parentOrdinal<tConRowEnd; parentOrdinal++)
            {
                OrdinalType tParentNode = tParentNodes(tMpcColumnIndices(parentOrdinal));
                outColumnIndices(tColMapOrdinal) = aNodeTypes(tParentNode);
                Plato::Scalar tMpcEntry = tMpcEntries(parentOrdinal);
                for(OrdinalType dofOrdinal=0; dofOrdinal<tNumDofsPerNode; dofOrdinal++)
                {
                    OrdinalType entryOrdinal = tColMapOrdinal*tBlockSize + tNumDofsPerNode*dofOrdinal + dofOrdinal; 
                    outEntries(entryOrdinal) = tMpcEntry;
                }
                tColMapOrdinal += 1;
            }
        }
        else 
        {
            outColumnIndices(tColMapOrdinal) = aNodeTypes(nodeOrdinal);
            for(OrdinalType dofOrdinal=0; dofOrdinal<tNumDofsPerNode; dofOrdinal++)
            {
                OrdinalType entryOrdinal = tColMapOrdinal*tBlockSize + tNumDofsPerNode*dofOrdinal + dofOrdinal; 
                outEntries(entryOrdinal) = 1.0;
            }
        }
    }, "Build block transformation matrix");

    // construct full CRS matrix
    OrdinalType tNdof = mNumNodes*mNumDofsPerNode;
    OrdinalType tNumCondensedDofs = (mNumNodes - tNumChildNodes)*mNumDofsPerNode;
    mTransformMatrix = Teuchos::rcp( new Plato::CrsMatrixType(outRowMap, outColumnIndices, outEntries, tNdof, tNumCondensedDofs, mNumDofsPerNode, mNumDofsPerNode) );
}

/****************************************************************************/
void MultipointConstraints::assembleRhs(const ScalarVector & aMpcValues)
/****************************************************************************/
{
    OrdinalType tNdof = mNumNodes*mNumDofsPerNode;
    OrdinalType tNumChildNodes = mChildNodes.size();

    Kokkos::resize(mRhs, tNdof);
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mRhs);

    auto tChildNodes = mChildNodes;
    auto tRhs = mRhs;
    auto tNumDofsPerNode = mNumDofsPerNode;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType childOrdinal)
    {
        OrdinalType childNode = tChildNodes(childOrdinal);
        for(OrdinalType dofOrdinal=0; dofOrdinal<tNumDofsPerNode; dofOrdinal++)
        {
            OrdinalType entryOrdinal = tNumDofsPerNode*childNode + dofOrdinal; 
            tRhs(entryOrdinal) = aMpcValues(childOrdinal);
        }
    }, "Set RHS vector values");
}

/****************************************************************************/
void MultipointConstraints::setupTransform()
/****************************************************************************/
{
    // fill in all constraint data
    Teuchos::RCP<Plato::CrsMatrixType> mpcMatrix;
    ScalarVector                       mpcValues;
    this->get(mpcMatrix, mpcValues);
    
    // fill in child DOFs
    mNumChildNodes = mChildNodes.size();

    // get mappings from global node to node type and constraint number
    LocalOrdinalVector nodeTypes;
    LocalOrdinalVector nodeConNum;
    this->getMaps(nodeTypes, nodeConNum);

    // build transformation matrix
    this->assembleTransformMatrix(mpcMatrix, nodeTypes, nodeConNum);

    // build transpose of transformation matrix
    auto tNumRows = mTransformMatrix->numCols();
    auto tNumCols = mTransformMatrix->numRows();
    auto tRetMat = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, mNumDofsPerNode, mNumDofsPerNode ) );
    Plato::MatrixTranspose(mTransformMatrix, tRetMat);
    mTransformMatrixTranspose = tRetMat;

    // build RHS
    this->assembleRhs(mpcValues);
}

/****************************************************************************/
void MultipointConstraints::checkEssentialBcsConflicts(const LocalOrdinalVector & aBcDofs)
/****************************************************************************/
{
    auto tNumDofsPerNode = mNumDofsPerNode;
    OrdinalType tNumBcDofs = aBcDofs.size();
    OrdinalType tNumChildNodes = mChildNodes.size();
    OrdinalType tNumParentNodes = mParentNodes.size();

    auto tBcDofs = aBcDofs;
    auto tChildNodes = mChildNodes;
    auto tParentNodes = mParentNodes;
    
    // check for child node conflicts
    Plato::OrdinalType tNumChildConflicts(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumBcDofs),
    LAMBDA_EXPRESSION(const Plato::OrdinalType& aBcOrdinal, Plato::OrdinalType & aUpdate)
    {
        OrdinalType tBcDof = tBcDofs(aBcOrdinal);
        OrdinalType tBcNode = ( tBcDof - tBcDof % tNumDofsPerNode ) / tNumDofsPerNode;
        for (OrdinalType tChildOrdinal=0; tChildOrdinal<tNumChildNodes; tChildOrdinal++)
        {
            if (tChildNodes(tChildOrdinal) == tBcNode) 
            {  
                aUpdate++;
            }
        }
    }, tNumChildConflicts);
    if ( tNumChildConflicts > 0 )
    {
        std::ostringstream tMsg;
        tMsg << "MPC CHILD NODE CONFLICTS WITH ESSENTIAL BC NODE. CHECK MESH SIZES. \n";
        THROWERR(tMsg.str())
    }
    
    // check for parent node conflicts
    Plato::OrdinalType tNumParentConflicts(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumBcDofs),
    LAMBDA_EXPRESSION(const Plato::OrdinalType& aBcOrdinal, Plato::OrdinalType & aUpdate)
    {
        OrdinalType tBcDof = tBcDofs(aBcOrdinal);
        OrdinalType tBcNode = ( tBcDof - tBcDof % tNumDofsPerNode ) / tNumDofsPerNode;
        for (OrdinalType tParentOrdinal=0; tParentOrdinal<tNumParentNodes; tParentOrdinal++)
        {
            if (tParentNodes(tParentOrdinal) == tBcNode) 
            {  
                aUpdate++;
            }
        }
    }, tNumParentConflicts);
    if ( tNumParentConflicts > 0 )
    {
        std::ostringstream tMsg;
        tMsg << "MPC PARENT NODE CONFLICTS WITH ESSENTIAL BC NODE. CHECK MESH SIZES. \n";
        THROWERR(tMsg.str())
    }
}

} // namespace Plato


#include "MultipointConstraints.hpp"

namespace Plato
{

/****************************************************************************/
MultipointConstraints::MultipointConstraints(Omega_h::Mesh & aMesh,
                                             const Omega_h::MeshSets & aMeshSets, 
                                             const OrdinalType & aNumDofsPerNode, 
                                             Teuchos::ParameterList & aParams) :
                                             MPCs(),
                                             mNumNodes(aMesh.nverts()),
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

        std::shared_ptr<MultipointConstraint> tMyMPC = tMultipointConstraintFactory.create(aMesh, aMeshSets, tMyName);
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

    Kokkos::resize(nodeTypes, mNumNodes);
    Kokkos::resize(nodeConNum, mNumNodes);

    Plato::blas1::fill(static_cast<Plato::OrdinalType>(0), nodeTypes);
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(-1), nodeConNum);

    auto tChildNodes = mChildNodes;

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType childOrdinal)
    {
        OrdinalType childNode = tChildNodes(childOrdinal);
        nodeTypes(childNode) = -1; // Mark child DOF
        nodeConNum(childNode) = childOrdinal;
    }, "Set child node type and constraint number");

    /* LocalOrdinalVector tCondensedOrdinals("condensed DOF ordinals", 1); */
    /* Plato::blas1::fill(static_cast<Plato::OrdinalType>(0), tCondensedOrdinals); */
    /* Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, mNumNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal) */
    /* { */
    /*     if (nodeTypes(nodeOrdinal) != -1) // not child node */
    /*     { */  
    /*         nodeTypes(nodeOrdinal) = tCondensedOrdinals(0); */ 
    /*         Kokkos::atomic_increment(&tCondensedOrdinals(0)); */
    /*     } */
    /* }, "Map from global node ID to condensed node ID"); */
    
    // assign condensed DOF ordinals
    Plato::OrdinalType tNumCondensedDofs(0);
    Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, mNumNodes),
    LAMBDA_EXPRESSION(const Plato::OrdinalType& nodeOrdinal, Plato::OrdinalType & aUpdate)
    {
        if (nodeTypes(nodeOrdinal) != -1) // not child node
        {  
            nodeTypes(nodeOrdinal) = aUpdate; 
            aUpdate++;
        }
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
            OrdinalType tConNnz = tConRowEnd - tConRowStart;
            outRowMap(nodeOrdinal + 1) = tColMapOrdinal + tConNnz; 

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
            outRowMap(nodeOrdinal + 1) = tColMapOrdinal + 1;
            outColumnIndices(tColMapOrdinal) = aNodeTypes(nodeOrdinal);
            for(OrdinalType dofOrdinal=0; dofOrdinal<tNumDofsPerNode; dofOrdinal++)
            {
                OrdinalType entryOrdinal = tColMapOrdinal*tBlockSize + tNumDofsPerNode*dofOrdinal + dofOrdinal; 
                outEntries(entryOrdinal) = 1.0;
            }
        }
    }, "Build block transformation matrix");

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


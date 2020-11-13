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
void MultipointConstraints::get(LocalOrdinalVector & mpcChildNodes,
                                LocalOrdinalVector & mpcParentNodes,
                                Teuchos::RCP<Plato::CrsMatrixType> & mpcMatrix,
                                ScalarVector & mpcValues)
/****************************************************************************/
{
    OrdinalType numChildNodes(0);
    OrdinalType numParentNodes(0);
    OrdinalType numConstraintNonzeros(0);
    for(std::shared_ptr<MultipointConstraint> & mpc : MPCs)
        mpc->updateLengths(numChildNodes, numParentNodes, numConstraintNonzeros);

    Kokkos::resize(mpcChildNodes, numChildNodes);
    Kokkos::resize(mpcParentNodes, numParentNodes);
    Kokkos::resize(mpcValues, numChildNodes);
    Plato::CrsMatrixType::RowMapVector mpcRowMap("row map", numChildNodes+1);
    Plato::CrsMatrixType::OrdinalVector mpcColumnIndices("column indices", numConstraintNonzeros);
    Plato::CrsMatrixType::ScalarVector mpcEntries("matrix entries", numConstraintNonzeros);

    OrdinalType offsetChild(0);
    OrdinalType offsetParent(0);
    OrdinalType offsetNnz(0);
    for(std::shared_ptr<MultipointConstraint> & mpc : MPCs)
    {
        mpc->get(mpcChildNodes, mpcParentNodes, mpcRowMap, mpcColumnIndices, mpcEntries, mpcValues, offsetChild, offsetParent, offsetNnz);
        mpc->updateLengths(offsetChild, offsetParent, offsetNnz);
    }

    // Build full CRS matrix to return
    mpcMatrix = Teuchos::rcp( new Plato::CrsMatrixType(mpcRowMap, mpcColumnIndices, mpcEntries, numChildNodes, numParentNodes, 1, 1) );
}

/****************************************************************************/
void MultipointConstraints::getMaps(const LocalOrdinalVector & aMpcChildNodes,
                                    LocalOrdinalVector & nodeTypes,
                                    LocalOrdinalVector & nodeConNum)
/****************************************************************************/
{
    OrdinalType tNumChildNodes = aMpcChildNodes.size();

    Kokkos::resize(nodeTypes, mNumNodes);
    Kokkos::resize(nodeConNum, mNumNodes);

    Plato::blas1::fill(static_cast<Plato::OrdinalType>(0), nodeTypes);
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(-1), nodeConNum);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType childOrdinal)
    {
        OrdinalType childNode = aMpcChildNodes(childOrdinal);
        nodeTypes(childNode) = -1; // Mark child DOF
        nodeConNum(childNode) = childOrdinal;
    }, "Set child node type and constraint number");

    LocalOrdinalVector tCondensedOrdinals("column indices", 1);
    Plato::blas1::fill(static_cast<Plato::OrdinalType>(0), tCondensedOrdinals);
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, mNumNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        if (nodeTypes(nodeOrdinal) != -1) // not child node
        {  
            nodeTypes(nodeOrdinal) = tCondensedOrdinals(0); 
            Kokkos::atomic_increment(&tCondensedOrdinals(0));
        }
    }, "Map from global node ID to condensed node ID");
}

/****************************************************************************/
void MultipointConstraints::assembleTransformMatrix(const Teuchos::RCP<Plato::CrsMatrixType> & aMpcMatrix,
                                                                        const LocalOrdinalVector & aMpcParentNodes,
                                                                        const LocalOrdinalVector & aNodeTypes,
                                                                        const LocalOrdinalVector & aNodeConNum)
/****************************************************************************/
{
    OrdinalType tBlockSize = mNumDofsPerNode*mNumDofsPerNode;

    const auto& tMpcRowMap = aMpcMatrix->rowMap();
    const auto& tMpcColumnIndices = aMpcMatrix->columnIndices();
    const auto& tMpcEntries = aMpcMatrix->entries();

    OrdinalType tNumChildNodes = tMpcRowMap.size() - 1;
    OrdinalType tNumParentNodes = aMpcParentNodes.size();
    OrdinalType tMpcNnz = tMpcEntries.size();
    OrdinalType tOutNnz = tBlockSize*((mNumNodes - tNumChildNodes) + tMpcNnz);

    Plato::CrsMatrixType::RowMapVector outRowMap("transform matrix row map", mNumNodes+1);
    Plato::CrsMatrixType::OrdinalVector outColumnIndices("transform matrix column indices", tOutNnz);
    Plato::CrsMatrixType::ScalarVector outEntries("transform matrix entries", tOutNnz);

    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), outEntries);

    auto tNumDofsPerNode = mNumDofsPerNode;
    LocalOrdinalVector tNumMisassignedNodes("misassigned node counter", 1); 
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, mNumNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        OrdinalType tColMapOrdinal = outRowMap(nodeOrdinal);
        OrdinalType nodeType = aNodeTypes(nodeOrdinal);
        if(nodeType == -1) // Child Node
        {
            OrdinalType conOrdinal = aNodeConNum(nodeOrdinal);
            if (conOrdinal == -1)
            {
                Kokkos::atomic_increment(&tNumMisassignedNodes(0));
            }

            OrdinalType tConRowStart = tMpcRowMap(conOrdinal);
            OrdinalType tConRowEnd = tMpcRowMap(conOrdinal + 1);
            OrdinalType tConNnz = tConRowEnd - tConRowStart;
            outRowMap(nodeOrdinal + 1) = tColMapOrdinal + tConNnz; 

            for(OrdinalType parentOrdinal=tConRowStart; parentOrdinal<tConRowEnd; parentOrdinal++)
            {
                OrdinalType tParentNode = aMpcParentNodes(tMpcColumnIndices(parentOrdinal));
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

    if ( tNumMisassignedNodes(0) > 0 )
    {
        std::ostringstream tMsg;
        tMsg << "At least one constrained node in MPCs is not associated with correct constraint number. Transformation matrix will be incorrect. \n";
        THROWERR(tMsg.str())
    }

    OrdinalType tNdof = mNumNodes*mNumDofsPerNode;
    OrdinalType tNumCondensedDofs = (mNumNodes - tNumChildNodes)*mNumDofsPerNode;
    mTransformMatrix = Teuchos::rcp( new Plato::CrsMatrixType(outRowMap, outColumnIndices, outEntries, tNdof, tNumCondensedDofs, mNumDofsPerNode, mNumDofsPerNode) );
}

/****************************************************************************/
void MultipointConstraints::assembleRhs(const LocalOrdinalVector & aMpcChildNodes,
                                                            const ScalarVector & aMpcValues)
/****************************************************************************/
{
    OrdinalType tNdof = mNumNodes*mNumDofsPerNode;
    OrdinalType tNumChildNodes = aMpcChildNodes.size();

    Kokkos::resize(mRhs, tNdof);
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mRhs);

    auto tRhs = mRhs;

    auto tNumDofsPerNode = mNumDofsPerNode;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType childOrdinal)
    {
        OrdinalType childNode = aMpcChildNodes(childOrdinal);
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
    LocalOrdinalVector                 mpcChildNodes;
    LocalOrdinalVector                 mpcParentNodes;
    Teuchos::RCP<Plato::CrsMatrixType> mpcMatrix;
    ScalarVector                       mpcValues;
    this->get(mpcChildNodes, mpcParentNodes, mpcMatrix, mpcValues);
    
    // fill in child DOFs
    mNumChildNodes = mpcChildNodes.size();

    // get mappings from global node to node type and constraint number
    LocalOrdinalVector nodeTypes;
    LocalOrdinalVector nodeConNum;
    this->getMaps(mpcChildNodes, nodeTypes, nodeConNum);

    // build transformation matrix
    OrdinalType tNumChildNodes = mpcChildNodes.size();
    this->assembleTransformMatrix(mpcMatrix, mpcParentNodes, nodeTypes, nodeConNum);

    // build transpose of transformation matrix
    auto tNumRows = mTransformMatrix->numCols();
    auto tNumCols = mTransformMatrix->numRows();
    auto tRetMat = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, mNumDofsPerNode, mNumDofsPerNode ) );
    Plato::MatrixTranspose(mTransformMatrix, tRetMat);
    mTransformMatrixTranspose = tRetMat;

    // build RHS
    this->assembleRhs(mpcChildNodes, mpcValues);
}

} // namespace Plato


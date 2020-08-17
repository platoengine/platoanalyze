#ifndef MULTIPOINT_CONSTRAINTS_HPP
#define MULTIPOINT_CONSTRAINTS_HPP

#include <sstream>

#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoMathHelpers.hpp"
#include "MultipointConstraintFactory.hpp"
#include "MultipointConstraint.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Owner class that contains a vector of MultipointConstraint objects.
 */
template<typename SimplexPhysicsType>
class MultipointConstraints
/******************************************************************************/
{
private:
    std::vector<std::shared_ptr<MultipointConstraint<SimplexPhysicsType>>> MPCs;
    const OrdinalType mNumNodes;
    Teuchos::RCP<Plato::CrsMatrixType> mTransformMatrix;
    Teuchos::RCP<Plato::CrsMatrixType> mTransformMatrixTranspose;
    ScalarVector mRhs;
    LocalOrdinalVector mChildDofs;

public :

    /*!
     \brief Constructor that parses and creates a vector of MultipointConstraint objects
     based on the ParameterList.
     */
    MultipointConstraints(const OrdinalType & aNumNodes, Teuchos::ParameterList & aParams);

    /*!
     \brief Get node ordinals and values for constraints.
     */
    void get(const Omega_h::MeshSets & aMeshSets,
             LocalOrdinalVector & mpcChildNodes,
             LocalOrdinalVector & mpcParentNodes,
             Teuchos::RCP<Plato::CrsMatrixType> & mpcMatrix,
             ScalarVector & mpcValues);

    // brief get mappings from DOF to DOF type and constraint number
    void getMaps(const LocalOrdinalVector & aMpcChildNodes,
                 LocalOrdinalVector & nodeTypes,
                 LocalOrdinalVector & nodeConNum);

    // brief assemble transform matrix for constraint enforcement
    void assembleTransformMatrix(const Teuchos::RCP<Plato::CrsMatrixType> & aMpcMatrix,
                                 const LocalOrdinalVector & aMpcParentNodes,
                                 const LocalOrdinalVector & aNodeTypes,
                                 const LocalOrdinalVector & aNodeConNum);

    // brief assemble RHS vector for transformation
    void assembleRhs(const LocalOrdinalVector & aMpcChildNodes,
                     const ScalarVector & aMpcValues);
    
    // brief get list of chold DOFs
    void listChildDofs(const LocalOrdinalVector & aMpcChildNodes);

    // brief setup transform matrices and RHS
    void setupTransform(const Omega_h::MeshSets & aMeshSets);

    // brief getters
    decltype(mTransformMatrix)          getTransformMatrix()           { return mTransformMatrix; }
    decltype(mTransformMatrixTranspose) getTransformMatrixTranspose()  { return mTransformMatrixTranspose; }
    decltype(mRhs)                      getRhsVector()                 { return mRhs; }
    decltype(mChildDofs)                getChildDofs()                 { return mChildDofs; }
    
    // brief const getters
    const decltype(mTransformMatrix)          getTransformMatrix()           const { return mTransformMatrix; }
    const decltype(mTransformMatrixTranspose) getTransformMatrixTranspose()  const { return mTransformMatrixTranspose; }
    const decltype(mRhs)                      getRhsVector()                 const { return mRhs; }
    const decltype(mChildDofs)                getChildDofs()                 const { return mChildDofs; }
};

/****************************************************************************/
template<typename SimplexPhysicsType>
MultipointConstraints<SimplexPhysicsType>::MultipointConstraints(const OrdinalType & aNumNodes, Teuchos::ParameterList & aParams) :
        MPCs(),
        mNumNodes(aNumNodes),
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
        Plato::MultipointConstraintFactory<SimplexPhysicsType> tMultipointConstraintFactory(tSublist);

        std::shared_ptr<MultipointConstraint<SimplexPhysicsType>> tMyMPC = tMultipointConstraintFactory.create(tMyName);
        MPCs.push_back(tMyMPC);
    }
}

/****************************************************************************/
template<typename SimplexPhysicsType>
void MultipointConstraints<SimplexPhysicsType>::get(const Omega_h::MeshSets & aMeshSets,
                                                    LocalOrdinalVector & mpcChildNodes,
                                                    LocalOrdinalVector & mpcParentNodes,
                                                    Teuchos::RCP<Plato::CrsMatrixType> & mpcMatrix,
                                                    ScalarVector & mpcValues)
/****************************************************************************/
{
    OrdinalType numChildNodes(0);
    OrdinalType numParentNodes(0);
    OrdinalType numConstraintNonzeros(0);
    for(std::shared_ptr<MultipointConstraint<SimplexPhysicsType>> & mpc : MPCs)
        mpc->updateLengths(aMeshSets, numChildNodes, numParentNodes, numConstraintNonzeros);

    Kokkos::resize(mpcChildNodes, numChildNodes);
    Kokkos::resize(mpcParentNodes, numParentNodes);
    Kokkos::resize(mpcValues, numChildNodes);
    Plato::CrsMatrixType::RowMapVector mpcRowMap("row map", numChildNodes+1);
    Plato::CrsMatrixType::OrdinalVector mpcColumnIndices("column indices", numConstraintNonzeros);
    Plato::CrsMatrixType::ScalarVector mpcEntries("matrix entries", numConstraintNonzeros);

    OrdinalType offsetChild(0);
    OrdinalType offsetParent(0);
    OrdinalType offsetNnz(0);
    for(std::shared_ptr<MultipointConstraint<SimplexPhysicsType>> & mpc : MPCs)
    {
        mpc->get(aMeshSets, mpcChildNodes, mpcParentNodes, mpcRowMap, mpcColumnIndices, mpcEntries, mpcValues, offsetChild, offsetParent, offsetNnz);
        mpc->updateLengths(aMeshSets, offsetChild, offsetParent, offsetNnz);
    }

    // Add total number of nonzeros to the end of rowMap
    mpcRowMap(offsetChild) = numConstraintNonzeros;

    // Build full CRS matrix to return
    mpcMatrix = Teuchos::rcp( new Plato::CrsMatrixType(mpcRowMap, mpcColumnIndices, mpcEntries, numChildNodes, numParentNodes, 1, 1) );
}

/****************************************************************************/
template<typename SimplexPhysicsType>
void MultipointConstraints<SimplexPhysicsType>::getMaps(const LocalOrdinalVector & aMpcChildNodes,
                                                        LocalOrdinalVector & nodeTypes,
                                                        LocalOrdinalVector & nodeConNum)
/****************************************************************************/
{
    OrdinalType tNumChildNodes = aMpcChildNodes.size();

    Kokkos::resize(nodeTypes, mNumNodes);
    Kokkos::resize(nodeConNum, mNumNodes);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, mNumNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        nodeTypes(nodeOrdinal) = 0; 
        nodeConNum(nodeOrdinal) = -1; 
    }, "Initialize Node type");

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType childOrdinal)
    {
        OrdinalType childNode = aMpcChildNodes(childOrdinal);
        nodeTypes(childNode) = 1; // Mark child DOF
        nodeConNum(childNode) = childOrdinal;
    }, "Set child node type and constraint number");
}

/****************************************************************************/
template<typename SimplexPhysicsType>
void MultipointConstraints<SimplexPhysicsType>::assembleTransformMatrix(const Teuchos::RCP<Plato::CrsMatrixType> & aMpcMatrix,
                                                                        const LocalOrdinalVector & aMpcParentNodes,
                                                                        const LocalOrdinalVector & aNodeTypes,
                                                                        const LocalOrdinalVector & aNodeConNum)
/****************************************************************************/
{
    OrdinalType tNumDofsPerNode = SimplexPhysicsType::mNumDofsPerNode;
    OrdinalType tBlockSize = tNumDofsPerNode*tNumDofsPerNode;

    const auto& tMpcRowMap = aMpcMatrix->rowMap();
    const auto& tMpcColumnIndices = aMpcMatrix->columnIndices();
    const auto& tMpcEntries = aMpcMatrix->entries();

    OrdinalType tNumChildNodes = tMpcRowMap.size() - 1;
    OrdinalType tMpcNnz = tMpcEntries.size();
    OrdinalType tOutNnz = tBlockSize*((mNumNodes - tNumChildNodes) + tMpcNnz);

    Plato::CrsMatrixType::RowMapVector outRowMap("transform matrix row map", mNumNodes+1);
    Plato::CrsMatrixType::OrdinalVector outColumnIndices("transform matrix column indices", tOutNnz);
    Plato::CrsMatrixType::ScalarVector outEntries("transform matrix entries", tOutNnz);

    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), outEntries);

    outRowMap(0) = 0;
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, mNumNodes), LAMBDA_EXPRESSION(Plato::OrdinalType nodeOrdinal)
    {
        OrdinalType tColMapOrdinal = outRowMap(nodeOrdinal);
        OrdinalType nodeType = aNodeTypes(nodeOrdinal);
        if(nodeType == 1) // Child Node
        {
            OrdinalType conOrdinal = aNodeConNum(nodeOrdinal);
            if (conOrdinal == -1)
            {
                std::ostringstream tMsg;
                tMsg << "Constrained node not associated with constraint number. \n";
                THROWERR(tMsg.str())
            }

            OrdinalType tConRowStart = tMpcRowMap(conOrdinal);
            OrdinalType tConRowEnd = tMpcRowMap(conOrdinal + 1);
            OrdinalType tConNnz = tConRowEnd - tConRowStart;
            outRowMap(nodeOrdinal + 1) = tColMapOrdinal + tConNnz; 

            for(OrdinalType parentOrdinal=tConRowStart; parentOrdinal<tConRowEnd; parentOrdinal++)
            {
                OrdinalType tParentNode = aMpcParentNodes(tMpcColumnIndices(parentOrdinal));
                outColumnIndices(tColMapOrdinal) = tParentNode;
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
            outColumnIndices(tColMapOrdinal) = nodeOrdinal;
            for(OrdinalType dofOrdinal=0; dofOrdinal<tNumDofsPerNode; dofOrdinal++)
            {
                OrdinalType entryOrdinal = tColMapOrdinal*tBlockSize + tNumDofsPerNode*dofOrdinal + dofOrdinal; 
                outEntries(entryOrdinal) = 1.0;
            }
        }
    }, "Build block transformation matrix");

    OrdinalType tNdof = mNumNodes*tNumDofsPerNode;
    mTransformMatrix = Teuchos::rcp( new Plato::CrsMatrixType(outRowMap, outColumnIndices, outEntries, tNdof, tNdof, tNumDofsPerNode, tNumDofsPerNode) );
}

/****************************************************************************/
template<typename SimplexPhysicsType>
void MultipointConstraints<SimplexPhysicsType>::assembleRhs(const LocalOrdinalVector & aMpcChildNodes,
                                                            const ScalarVector & aMpcValues)
/****************************************************************************/
{
    OrdinalType tNumDofsPerNode = SimplexPhysicsType::mNumDofsPerNode;
    OrdinalType tNdof = mNumNodes*tNumDofsPerNode;
    OrdinalType tNumChildNodes = aMpcChildNodes.size();

    Kokkos::resize(mRhs, tNdof);
    Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mRhs);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType childOrdinal)
    {
        OrdinalType childNode = aMpcChildNodes(childOrdinal);
        for(OrdinalType dofOrdinal=0; dofOrdinal<tNumDofsPerNode; dofOrdinal++)
        {
            OrdinalType entryOrdinal = tNumDofsPerNode*childNode + dofOrdinal; 
            mRhs(entryOrdinal) = aMpcValues(childOrdinal);
        }
    }, "Set RHS vector values");
}

/****************************************************************************/
template<typename SimplexPhysicsType>
void MultipointConstraints<SimplexPhysicsType>::listChildDofs(const LocalOrdinalVector & aMpcChildNodes)
/****************************************************************************/
{
    OrdinalType tNumDofsPerNode = SimplexPhysicsType::mNumDofsPerNode;
    OrdinalType tNumChildNodes = aMpcChildNodes.size();
    OrdinalType tNumChildDofs = tNumChildNodes*tNumDofsPerNode;

    Kokkos::resize(mChildDofs, tNumChildDofs);

    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumChildNodes), LAMBDA_EXPRESSION(Plato::OrdinalType childOrdinal)
    {
        OrdinalType childNode = aMpcChildNodes(childOrdinal);
        for(OrdinalType dofOrdinal=0; dofOrdinal<tNumDofsPerNode; dofOrdinal++)
        {
            OrdinalType childDofOrdinal = tNumDofsPerNode*childOrdinal + dofOrdinal;
            mChildDofs(childDofOrdinal) = tNumDofsPerNode*childNode + dofOrdinal;
        }
    }, "Get child DOFs");
}

/****************************************************************************/
template<typename SimplexPhysicsType>
void MultipointConstraints<SimplexPhysicsType>::setupTransform(const Omega_h::MeshSets & aMeshSets)
/****************************************************************************/
{
    // fill in all constraint data
    LocalOrdinalVector                 mpcChildNodes;
    LocalOrdinalVector                 mpcParentNodes;
    Teuchos::RCP<Plato::CrsMatrixType> mpcMatrix;
    ScalarVector                       mpcValues;
    this->get(aMeshSets, mpcChildNodes, mpcParentNodes, mpcMatrix, mpcValues);
    
    // fill in child DOFs
    this->listChildDofs(mpcChildNodes);

    // get mappings from global node to node type and constraint number
    LocalOrdinalVector nodeTypes;
    LocalOrdinalVector nodeConNum;
    this->getMaps(mpcChildNodes, nodeTypes, nodeConNum);

    // build transformation matrix
    OrdinalType tNumChildNodes = mpcChildNodes.size();
    this->assembleTransformMatrix(mpcMatrix, mpcParentNodes, nodeTypes, nodeConNum);

    // build transpose of transformation matrix
    OrdinalType tNumDofsPerNode = SimplexPhysicsType::mNumDofsPerNode;
    auto tNumRows = mTransformMatrix->numCols();
    auto tNumCols = mTransformMatrix->numRows();
    auto tRetMat = Teuchos::rcp( new Plato::CrsMatrixType( tNumRows, tNumCols, tNumDofsPerNode, tNumDofsPerNode ) );
    Plato::MatrixTranspose(mTransformMatrix, tRetMat);
    mTransformMatrixTranspose = tRetMat;

    // build RHS
    this->assembleRhs(mpcChildNodes, mpcValues);
}

} // namespace Plato

#endif


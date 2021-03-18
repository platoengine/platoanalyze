#ifndef MULTIPOINT_CONSTRAINTS_HPP
#define MULTIPOINT_CONSTRAINTS_HPP

#include <sstream>

#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"
#include "BLAS1.hpp"
#include "PlatoMathHelpers.hpp"
#include "MultipointConstraintFactory.hpp"
#include "MultipointConstraint.hpp"
#include "SpatialModel.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Owner class that contains a vector of MultipointConstraint objects.
 */
class MultipointConstraints
/******************************************************************************/
{
private:
    std::vector<std::shared_ptr<MultipointConstraint>> MPCs;
    const OrdinalType mNumDofsPerNode;
    const OrdinalType mNumNodes;
    LocalOrdinalVector mChildNodes;
    LocalOrdinalVector mParentNodes;
    Teuchos::RCP<Plato::CrsMatrixType> mTransformMatrix;
    Teuchos::RCP<Plato::CrsMatrixType> mTransformMatrixTranspose;
    ScalarVector mRhs;
    OrdinalType mNumChildNodes;

public :

    /*!
     \brief Constructor that parses and creates a vector of MultipointConstraint objects
     based on the ParameterList.
     */
    MultipointConstraints(const Plato::SpatialModel & aSpatialModel,
                          const OrdinalType & aNumDofsPerNode, 
                          Teuchos::ParameterList & aParams);

    /*!
     \brief Get node ordinals and values for constraints.
     */
    void get(Teuchos::RCP<Plato::CrsMatrixType> & mpcMatrix,
             ScalarVector & mpcValues);

    // brief get mappings from DOF to DOF type and constraint number
    void getMaps(LocalOrdinalVector & nodeTypes,
                 LocalOrdinalVector & nodeConNum);

    // brief assemble transform matrix for constraint enforcement
    void assembleTransformMatrix(const Teuchos::RCP<Plato::CrsMatrixType> & aMpcMatrix,
                                 const LocalOrdinalVector & aNodeTypes,
                                 const LocalOrdinalVector & aNodeConNum);

    // brief assemble RHS vector for transformation
    void assembleRhs(const ScalarVector & aMpcValues);

    // brief setup transform matrices and RHS
    void setupTransform();
    
    // brief check for MPC and Essential BC conflicts
    void checkEssentialBcsConflicts(const LocalOrdinalVector & aBcDofs);

    // brief getters
    decltype(mTransformMatrix)          getTransformMatrix()           { return mTransformMatrix; }
    decltype(mTransformMatrixTranspose) getTransformMatrixTranspose()  { return mTransformMatrixTranspose; }
    decltype(mRhs)                      getRhsVector()                 { return mRhs; }
    Plato::OrdinalType                  getNumTotalNodes()             { return mNumNodes; }
    Plato::OrdinalType                  getNumCondensedNodes()         { return mNumNodes - mNumChildNodes; }
    Plato::OrdinalType                  getNumDofsPerNode()            { return mNumDofsPerNode; }
    
    // brief const getters
    const decltype(mTransformMatrix)          getTransformMatrix()           const { return mTransformMatrix; }
    const decltype(mTransformMatrixTranspose) getTransformMatrixTranspose()  const { return mTransformMatrixTranspose; }
    const decltype(mRhs)                      getRhsVector()                 const { return mRhs; }
    Plato::OrdinalType                        getNumTotalNodes()             const { return mNumNodes; }
    Plato::OrdinalType                        getNumCondensedNodes()         const { return mNumNodes - mNumChildNodes; }
    Plato::OrdinalType                        getNumDofsPerNode()            const { return mNumDofsPerNode; }
};

} // namespace Plato

#endif


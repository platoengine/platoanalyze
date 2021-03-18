/*
 * PbcMultipointConstraint.hpp
 *
 *  Created on: September 22, 2020
 */

/* #pragma once */

#ifndef PBC_MULTIPOINT_CONSTRAINT_HPP
#define PBC_MULTIPOINT_CONSTRAINT_HPP

#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

#include "MultipointConstraint.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"
#include "BLAS1.hpp"
#include "Plato_MeshMapUtils.hpp"
#include "SpatialModel.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Derived class for PBC multipoint constraint
 *
**********************************************************************************/
class PbcMultipointConstraint : public Plato::MultipointConstraint
{

public:
    PbcMultipointConstraint(const Plato::SpatialModel & aSpatialModel,
                            const std::string & aName, 
                            Teuchos::ParameterList & aParam);

    virtual ~PbcMultipointConstraint(){}

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
    void get(LocalOrdinalVector & aMpcChildNodes,
             LocalOrdinalVector & aMpcParentNodes,
             Plato::CrsMatrixType::RowMapVector & aMpcRowMap,
             Plato::CrsMatrixType::OrdinalVector & aMpcColumnIndices,
             Plato::CrsMatrixType::ScalarVector & aMpcEntries,
             ScalarVector & aMpcValues,
             OrdinalType aOffsetChild,
             OrdinalType aOffsetParent,
             OrdinalType aOffsetNnz) override;
    
    // ! Get number of nodes in the constrained nodeset.
    void updateLengths(OrdinalType& lengthChild,
                       OrdinalType& lengthParent,
                       OrdinalType& lengthNnz) override;

    // ! Fill in node set members
    void updateNodesets(const OrdinalType& tNumberChildNodes,
                        const Omega_h::LOs& tChildNodeLids);

    // ! Perform translation mapping from child nodes to parent locations
    void mapChildVertexLocations(Omega_h::Mesh & aMesh,
                                 const Plato::Scalar aTranslationX,
                                 const Plato::Scalar aTranslationY,
                                 const Plato::Scalar aTranslationZ,
                                 Plato::ScalarMultiVector & aLocations,
                                 Plato::ScalarMultiVector & aMappedLocations);
    
    // ! Use mapped parent elements to find global IDs of unique parent nodes
    void getUniqueParentNodes(Omega_h::Mesh & aMesh,
                              LocalOrdinalVector & aParentElements,
                              LocalOrdinalVector & aParentGlobalLocalMap);

    void setMatrixValues(Omega_h::Mesh & aMesh,
                         LocalOrdinalVector & aParentElements,
                         Plato::ScalarMultiVector & aMappedLocations,
                         LocalOrdinalVector & aParentGlobalLocalMap);

private:
    LocalOrdinalVector                 mChildNodes;
    LocalOrdinalVector                 mParentNodes;
    Plato::Scalar                      mValue;
    Teuchos::RCP<Plato::CrsMatrixType> mMpcMatrix;

};
// class PbcMultipointConstraint

}
// namespace Plato

#endif

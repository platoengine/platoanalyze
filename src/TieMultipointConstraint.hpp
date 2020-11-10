/*
 * TieMultipointConstraint.hpp
 *
 *  Created on: May 26, 2020
 */

/* #pragma once */

#ifndef TIE_MULTIPOINT_CONSTRAINT_HPP
#define TIE_MULTIPOINT_CONSTRAINT_HPP

#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

#include "MultipointConstraint.hpp"
#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Derived class for tie multipoint constraint
 *
**********************************************************************************/
class TieMultipointConstraint : public Plato::MultipointConstraint
{
public:
    TieMultipointConstraint(const Omega_h::MeshSets & aMeshSets,
                            const std::string & aName, 
                            Teuchos::ParameterList & aParam);

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
                        const Omega_h::LOs& tChildNodeLids,
                        const Omega_h::LOs& tParentNodeLids);

private:
    LocalOrdinalVector    mParentNodes;
    LocalOrdinalVector    mChildNodes;
    Plato::Scalar         mValue;

};
// class TieMultipointConstraint

}
// namespace Plato

#endif

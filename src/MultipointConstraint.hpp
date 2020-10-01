#ifndef MULTIPOINT_CONSTRAINT_HPP
#define MULTIPOINT_CONSTRAINT_HPP

#include <sstream>

#include <Omega_h_assoc.hpp>
#include <Teuchos_ParameterList.hpp>

#include "AnalyzeMacros.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief virtual base class for multipoint constraints.
 */
class MultipointConstraint
/******************************************************************************/
{
public:

    MultipointConstraint(const std::string & aName) :
                        name(aName)
    {
    }

    ~MultipointConstraint()
    {
    }

    /*!
     \brief Pure virtual function 
     \Get constraint matrix and RHS data.
     \param aMeshSets Omega_h mesh sets that contains nodeset data.
     \param mpcRowMap CRS-style rowMap for constraint data.
     \param mpcColumnIndices CRS-style columnIndices for constraint data.
     \param mpcEntries CRS-style entries for constraint data.
     \param mpcValues Value list for constraint RHS.
     \param offsetChild Starting location for storage of child nodes.
     \param offsetParent Starting location for storage of parent nodes.
     \param offsetNnz Starting location in columnIndices/entries where constraining nodes/coefficients will be added.
     */
    virtual void get(LocalOrdinalVector & aMpcChildNodes,
                 LocalOrdinalVector & aMpcParentNodes,
                 Plato::CrsMatrixType::RowMapVector & aMpcRowMap,
                 Plato::CrsMatrixType::OrdinalVector & aMpcColumnIndices,
                 Plato::CrsMatrixType::ScalarVector & aMpcEntries,
                 ScalarVector & aMpcValues,
                 OrdinalType aOffsetChild,
                 OrdinalType aOffsetParent,
                 OrdinalType aOffsetNnz) = 0;

    // ! Update number of nodes in the child and parent nodesets and number of nonzeros in constraint matrix.
    virtual void updateLengths(OrdinalType& lengthChild,
                               OrdinalType& lengthParent,
                               OrdinalType& lengthNnz) = 0;
    
protected:
    const std::string name;
};

} // namespace Plato

#endif


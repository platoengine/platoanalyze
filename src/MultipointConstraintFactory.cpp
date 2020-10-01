/*
 * MultipointConstraintFactory.hpp
 *
 *  Created on: May 26, 2020
 */
#include "MultipointConstraintFactory.hpp"

namespace Plato
{

/******************************************************************************//**
* \brief Create a multipoint constraint
**********************************************************************************/
std::shared_ptr<MultipointConstraint> MultipointConstraintFactory::
create(Omega_h::Mesh & aMesh, 
       const Omega_h::MeshSets & aMeshSets,
       const std::string& aName)
{
    const std::string tType = mParamList.get<std::string>("Type");

    if("Tie" == tType)
    {
        return std::make_shared<Plato::TieMultipointConstraint>(aMeshSets, aName, mParamList);
    }
    else if("PBC" == tType)
    {
        return std::make_shared<Plato::PbcMultipointConstraint>(aMesh, aMeshSets, aName, mParamList);
    }
    return std::shared_ptr<Plato::MultipointConstraint>(nullptr);
}

}
// namespace Plato

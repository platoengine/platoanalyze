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
create(const Plato::SpatialModel & aSpatialModel, 
       const std::string& aName)
{
    const std::string tType = mParamList.get<std::string>("Type");

    if("Tie" == tType)
    {
        return std::make_shared<Plato::TieMultipointConstraint>(aSpatialModel.MeshSets, aName, mParamList);
    }
    else if("PBC" == tType)
    {
        return std::make_shared<Plato::PbcMultipointConstraint>(aSpatialModel, aName, mParamList);
    }
    return std::shared_ptr<Plato::MultipointConstraint>(nullptr);
}

}
// namespace Plato

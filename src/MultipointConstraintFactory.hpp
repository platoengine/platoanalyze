/*
 * MultipointConstraintFactory.hpp
 *
 *  Created on: May 26, 2020
 */

#include "MultipointConstraint.hpp"
#include "TieMultipointConstraint.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Factory for creating multipoint constraints.
 *
**********************************************************************************/
template<typename SimplexPhysicsType>
class MultipointConstraintFactory
{
public:
    /******************************************************************************//**
    * \brief Multipoint constraint factory constructor.
    * \param [in] aParamList input parameter list
    **********************************************************************************/
    MultipointConstraintFactory(Teuchos::ParameterList& aParamList) :
            mParamList(aParamList){}

    /******************************************************************************//**
    * \brief Create a multipoint constraint.
    * \return multipoint constraint
    **********************************************************************************/
    std::shared_ptr<Plato::MultipointConstraint<SimplexPhysicsType>> create(const std::string& aName);

private:
    Teuchos::ParameterList& mParamList; /*!< Input parameter list */
};
// class MultipointConstraintFactory

/******************************************************************************//**
* \brief Create a multipoint constraint
**********************************************************************************/
template<typename SimplexPhysicsType>
std::shared_ptr<MultipointConstraint<SimplexPhysicsType>> MultipointConstraintFactory<SimplexPhysicsType>::create(const std::string& aName)
{
    const std::string tType = mParamList.get<std::string>("Type");

    if("Tie" == tType)
    {
        return std::make_shared<Plato::TieMultipointConstraint<SimplexPhysicsType>>(aName, mParamList);
    }
    /* else if("PBC" == tType) */
    /* { */
        /* return std::make_shared<Plato::PbcMultipointConstraint<SimplexPhysicsType>>(aName, mParamList); */
    /* } */
    return std::shared_ptr<Plato::MultipointConstraint<SimplexPhysicsType>>(nullptr);
}

}
// namespace Plato

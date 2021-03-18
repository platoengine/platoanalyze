/*
 * MultipointConstraintFactory.hpp
 *
 *  Created on: May 26, 2020
 */
#ifndef MULTIPOINT_CONSTRAINT_FACTORY_HPP
#define MULTIPOINT_CONSTRAINT_FACTORY_HPP

#include "MultipointConstraint.hpp"
#include "TieMultipointConstraint.hpp"
#include "PbcMultipointConstraint.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Factory for creating multipoint constraints.
 *
**********************************************************************************/
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
    std::shared_ptr<Plato::MultipointConstraint> create(const Plato::SpatialModel & aSpatialModel,
                                                        const std::string& aName);

private:
    Teuchos::ParameterList& mParamList; /*!< Input parameter list */
};
// class MultipointConstraintFactory

}
// namespace Plato

#endif

/*
 * NoPenalty.hpp
 *
 *  Created on: Apr 13, 2020
 *      Author: doble
 */

#ifndef SRC_PLATO_NOPENALTY_HPP_
#define SRC_PLATO_NOPENALTY_HPP_

#include <Teuchos_ParameterList.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Dummy Solid Isotropic Material Penalization (SIMP) Penalty Model
**********************************************************************************/
class NoPenalty
{
public:
    /******************************************************************************//**
     * \brief Constructor
    **********************************************************************************/
    NoPenalty()
    {
    }

    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aParamList input parameter list
    **********************************************************************************/
    NoPenalty(Teuchos::ParameterList & aParamList)
    {
    }

    /******************************************************************************//**
     * \brief Use for additive continuation, it does nothing in a no-penalty model.
    **********************************************************************************/
    void update()
    {
    } 

    /******************************************************************************//**
     * \brief Return a value of 1.0 for a no-penalty model
     * \param [in] aInputParams input parameters
     * \return penalized ersatz material
    **********************************************************************************/
    template<typename ScalarType>
    DEVICE_TYPE inline ScalarType operator()(ScalarType aInput) const
    {
        ScalarType tOutput = 1.0;
        return (tOutput);
    }
};

}




#endif /* SRC_PLATO_NOPENALTY_HPP_ */

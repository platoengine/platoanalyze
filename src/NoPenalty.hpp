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

/******************************************************************************/
class NoPenalty
/******************************************************************************/
{

public:
    NoPenalty()
    {
    }

    NoPenalty(Teuchos::ParameterList & aParamList)
    {
    }

    template<typename ScalarType>
    DEVICE_TYPE inline ScalarType operator()(ScalarType aInput) const
    {
        ScalarType tOutput = 1.0;
        return (tOutput);
    }
};

}




#endif /* SRC_PLATO_NOPENALTY_HPP_ */

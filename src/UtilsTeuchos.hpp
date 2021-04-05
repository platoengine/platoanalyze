/*
 * UtilsTeuchos.hpp
 *
 *  Created on: Apr 5, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace teuchos
{

/******************************************************************************//**
 * \fn void is_material_defined
 *
 * \brief Check if material is defined, if not, throw an error.
 *
 * \param [in] aMaterialName material sublist name
 * \param [in] aInputs       parameter list with input data information
**********************************************************************************/
inline void is_material_defined
(const std::string & aMaterialName,
 const Teuchos::ParameterList & aInputs)
{
    if(!aInputs.sublist("Material Models").isSublist(aMaterialName))
    {
        THROWERR(std::string("Material with tag '") + aMaterialName + "' is not defined in 'Material Models' block")
    }
}
// function is_material_defined

}

}
// namespace Plato

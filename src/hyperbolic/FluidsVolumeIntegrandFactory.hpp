/*
 * FluidsVolumeIntegrandFactory.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include <Teuchos_ParameterList.hpp>

#include "SpatialModel.hpp"
#include "PlatoUtilities.hpp"
#include "AbstractVolumeIntegrand.hpp"

#include "hyperbolic/InternalThermalForces.hpp"

namespace Plato
{

namespace Fluids
{


/***************************************************************************//**
 * \strut VolumeIntegrandFactory
 *
 * \brief Factory for internal force integrals for computational fluid dynamics
 *   applications.
 *
 ******************************************************************************/
struct VolumeIntegrandFactory
{

/***************************************************************************//**
 * \tparam PhysicsT    Physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \fn inline std::shared_ptr<AbstractVolumeIntegrand> createInternalThermalForces
 *
 * \brief Return shared pointer to an abstract cell volume integral instance.
 *
 * \param [in] aDomain  spatial domain metadata
 * \param [in] aDataMap output database
 * \param [in] aInputs  input file metadata
 *
 ******************************************************************************/
template <typename PhysicsT, typename EvaluationT>
inline std::shared_ptr<Plato::AbstractVolumeIntegrand<PhysicsT, EvaluationT>>
createInternalThermalForces
(const Plato::SpatialDomain & aDomain,
 Plato::DataMap & aDataMap,
 Teuchos::ParameterList & aInputs)
{
    if(aInputs.isSublist("Hyperbolic") == false)
    {
        THROWERR("'Hyperbolic' Parameter List is not defined.")
    }
    auto tHyperbolic = aInputs.sublist("Hyperbolic");

    auto tScenario = tHyperbolic.get<std::string>("Scenario","Analysis");
    auto tLowerScenario = Plato::tolower(tScenario);
    if( tLowerScenario == "density to" )
    {
        return ( std::make_shared<Plato::Fluids::SIMP::InternalThermalForces<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else if( tLowerScenario == "analysis" || tLowerScenario == "levelset to" )
    {
        return ( std::make_shared<Plato::Fluids::InternalThermalForces<PhysicsT, EvaluationT>>(aDomain, aDataMap, aInputs) );
    }
    else
    {
        THROWERR(std::string("Scenario '") + tScenario + "' is not supported. Options are 1) Analysis, 2) Density TO or 3) Levelset TO.")
    }
}

};
// struct VolumeIntegrandFactory

}
// namespace Fluids

}
// namespace Plato

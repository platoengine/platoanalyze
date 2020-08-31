/*
 * PathDependentScalarFunctionFactory.hpp
 *
 *  Created on: Mar 1, 2020
 */

#pragma once

#include <memory>
#include <string>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include <Teuchos_ParameterList.hpp>

#include "LocalScalarFunctionInc.hpp"
#include "InfinitesimalStrainPlasticity.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Factory for scalar functions interface with local path-dependent states
 **********************************************************************************/
template<typename PhysicsT>
class PathDependentScalarFunctionFactory
{
public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    PathDependentScalarFunctionFactory () {}

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~PathDependentScalarFunctionFactory() {}

    /******************************************************************************//**
     * \brief Create interface for the evaluation of path-dependent scalar function
     *  operators, e.g. value and sensitivities.
     * \param [in] aMesh         mesh database
     * \param [in] aMeshSets     side sets database
     * \param [in] aDataMap      output database
     * \param [in] aInputParams  problem inputs in XML file
     * \param [in] aFunctionName scalar function name, i.e. type
     * \return shared pointer to the interface of path-dependent scalar functions
     **********************************************************************************/
    std::shared_ptr<Plato::LocalScalarFunctionInc>
    create(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFunctionName);
};
// class PathDependentScalarFunctionFactory

}
// namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::PathDependentScalarFunctionFactory<Plato::InfinitesimalStrainPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::PathDependentScalarFunctionFactory<Plato::InfinitesimalStrainPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::PathDependentScalarFunctionFactory<Plato::InfinitesimalStrainPlasticity<3>>;
#endif

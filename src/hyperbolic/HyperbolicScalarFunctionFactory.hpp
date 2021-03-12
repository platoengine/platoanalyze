#pragma once

#include <memory>

#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"
#include "HyperbolicScalarFunctionBase.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************//**
 * @brief Scalar function base factory
 **********************************************************************************/
template<typename PhysicsT>
class ScalarFunctionFactory
{
public:
    /******************************************************************************//**
     * @brief Constructor
     **********************************************************************************/
    ScalarFunctionFactory () {}

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    ~ScalarFunctionFactory() {}

    /******************************************************************************//**
     * @brief Create method
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams parameter input
     * @param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    std::shared_ptr<Plato::Hyperbolic::ScalarFunctionBase>
    create(
        Plato::SpatialModel    & aSpatialModel,
        Plato::DataMap         & aDataMap,
        Teuchos::ParameterList & aInputParams,
        std::string            & aFunctionName);
}; // class ScalarFunctionFactory

} // namespace Hyperbolic

} // namespace Plato

#include "HyperbolicMechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::Mechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::Mechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Hyperbolic::ScalarFunctionFactory<::Plato::Hyperbolic::Mechanics<3>>;
#endif

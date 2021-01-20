#pragma once

#include <memory>

#include "SpatialModel.hpp"
#include "PlatoSequence.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/updated_lagrangian/ScalarFunctionBase.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

/******************************************************************************//**
 * \brief Scalar function base factory
 **********************************************************************************/
template<typename PhysicsT>
class ScalarFunctionBaseFactory
{
public:
    /******************************************************************************//**
     * \brief Constructor
     **********************************************************************************/
    ScalarFunctionBaseFactory () {}

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~ScalarFunctionBaseFactory() {}

    /******************************************************************************//**
     * \brief Create method
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams parameter input
     * \param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::ScalarFunctionBase> 
    create(
              Plato::SpatialModel                 & aSpatialModel,
        const Plato::Sequence<PhysicsT::SpaceDim> & aSequence,
              Plato::DataMap                      & aDataMap,
              Teuchos::ParameterList              & aInputParams,
              std::string                         & aFunctionName);
};
// class ScalarFunctionBaseFactory

} // namespace UpdatedLagrangian

} // namespace Elliptic

} // namespace Plato

#include "elliptic/updated_lagrangian/Mechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Elliptic::UpdatedLagrangian::ScalarFunctionBaseFactory<::Plato::Elliptic::UpdatedLagrangian::Mechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Elliptic::UpdatedLagrangian::ScalarFunctionBaseFactory<::Plato::Elliptic::UpdatedLagrangian::Mechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Elliptic::UpdatedLagrangian::ScalarFunctionBaseFactory<::Plato::Elliptic::UpdatedLagrangian::Mechanics<3>>;
#endif

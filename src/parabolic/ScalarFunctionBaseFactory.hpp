#pragma once

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include "PlatoStaticsTypes.hpp"
#include "parabolic/ScalarFunctionBase.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Parabolic
{
/******************************************************************************//**
 * @brief Scalar function base factory
 **********************************************************************************/
template<typename PhysicsT>
class ScalarFunctionBaseFactory
{
public:
    /******************************************************************************//**
     * @brief Constructor
     **********************************************************************************/
    ScalarFunctionBaseFactory () {}

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    ~ScalarFunctionBaseFactory() {}

    /******************************************************************************//**
     * @brief Create method
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams parameter input
     * @param [in] aFunctionName name of function in parameter list
     **********************************************************************************/
    std::shared_ptr<Plato::Parabolic::ScalarFunctionBase> 
    create(Omega_h::Mesh& aMesh,
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap & aDataMap,
           Teuchos::ParameterList& aInputParams,
           std::string& aFunctionName);
};
// class ScalarFunctionBaseFactory

} // namespace Parabolic

} // namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "StabilizedMechanics.hpp"
#include "Thermomechanics.hpp"
#include "StabilizedThermomechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::Thermal<1>>;
extern template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<1>>;
extern template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<1>>;
extern template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::Thermal<2>>;
extern template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<2>>;
extern template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<2>>;
extern template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::Thermal<3>>;
extern template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::Thermomechanics<3>>;
extern template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::StabilizedMechanics<3>>;
extern template class Plato::Parabolic::ScalarFunctionBaseFactory<::Plato::StabilizedThermomechanics<3>>;
#endif

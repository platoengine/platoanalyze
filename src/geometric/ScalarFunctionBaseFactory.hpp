#pragma once

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include "PlatoStaticsTypes.hpp"
#include <Teuchos_ParameterList.hpp>
#include "geometric/ScalarFunctionBase.hpp"

namespace Plato
{

namespace Geometric
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
    std::shared_ptr<Plato::Geometric::ScalarFunctionBase> 
    create(Omega_h::Mesh& aMesh,
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap & aDataMap,
           Teuchos::ParameterList& aInputParams,
           std::string& aFunctionName);
};
// class ScalarFunctionBaseFactory

} // namespace Geometric

} // namespace Plato

#include "Geometrical.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Geometric::ScalarFunctionBaseFactory<::Plato::Geometrical<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Geometric::ScalarFunctionBaseFactory<::Plato::Geometrical<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Geometric::ScalarFunctionBaseFactory<::Plato::Geometrical<3>>;
#endif

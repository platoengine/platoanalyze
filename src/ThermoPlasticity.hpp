#ifndef PLATO_THERMOPLASTICITY_HPP
#define PLATO_THERMOPLASTICITY_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "SimplexThermoPlasticity.hpp"
#include "J2PlasticityLocalResidual.hpp"
#include "AnalyzeMacros.hpp"

namespace Plato
{

namespace ThermoPlasticityFactory
{

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************//**
     * \brief Create a PLATO local vector function  inc (i.e. local residual equations)
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap PLATO Analyze physics-based database
     * \param [in] aInputParams input parameters
    **********************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<EvaluationType>>
    createLocalVectorFunctionInc(Omega_h::Mesh& aMesh, 
                                 Omega_h::MeshSets& aMeshSets,
                                 Plato::DataMap& aDataMap, 
                                 Teuchos::ParameterList& aInputParams)
    {
        if(aInputParams.isSublist("Plasticity Model") == false)
        {
            THROWERR("Plasticity Model Sublist is not defined.")
        }

        auto tPlasticityParamList = aInputParams.get<Teuchos::ParameterList>("Plasticity Model");

        if(tPlasticityParamList.isSublist("J2ThermoPlasticity"))
        {
          constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
          return std::make_shared
            <J2PlasticityLocalResidual<EvaluationType, Plato::SimplexThermoPlasticity<tSpaceDim>>>
            (aMesh, aMeshSets, aDataMap, aInputParams);
        }
        else
        {
          const std::string tError = std::string("Unknown Plasticity Model.  Options are: J2ThermoPlasticity.")
              + "User is advised to select one of the available options.";
          THROWERR(tError)
        }
    }
}; // struct FunctionFactory

} // namespace ThermoPlasticityFactory


/****************************************************************************//**
 * \brief Concrete class for use as the PhysicsT template argument in VectorFunctionVMS
 *******************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class ThermoPlasticity: public Plato::SimplexThermoPlasticity<SpaceDimParam>
{
public:
    typedef Plato::ThermoPlasticityFactory::FunctionFactory FunctionFactory;
    using SimplexT        = Plato::SimplexThermoPlasticity<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};

} // namespace Plato

#endif

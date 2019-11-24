#ifndef PLATO_PLASTICITY_HPP
#define PLATO_PLASTICITY_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/SimplexPlasticity.hpp"
#include "plato/J2PlasticityLocalResidual.hpp"
#include "plato/AnalyzeMacros.hpp"

namespace Plato
{

namespace PlasticityFactory
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
        if(aInputParams.isSublist("Plasticity Model"))
        {
            THROWERR("Plasticity Model Sublist is not defined.")
        }

        auto tPlasticityParamList = aInputParams.get<Teuchos::ParameterList>("Plasticity Model");

        if(tPlasticityParamList.isSublist("J2 Plasticity"))
        {
          constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
          return std::make_shared
            <J2PlasticityLocalResidual<EvaluationType, Plato::SimplexPlasticity<tSpaceDim>>>
            (aMesh, aMeshSets, aDataMap, aInputParams);
        }
        else
        {
          THROWERR("Plasticity Model is node defined.  Options are: J2 Plasticity.  User is advised to select one of the available options")
        }
    }
}; // struct FunctionFactory

} // namespace PlasticityFactory


/****************************************************************************//**
 * \brief Concrete class for use as the PhysicsT template argument in VectorFunctionVMS
 *******************************************************************************/
template<Plato::OrdinalType SpaceDimParam>
class Plasticity: public Plato::SimplexPlasticity<SpaceDimParam>
{
public:
    typedef Plato::PlasticityFactory::FunctionFactory FunctionFactory;
    using SimplexT = Plato::SimplexPlasticity<SpaceDimParam>;
    static constexpr auto SpaceDim = SpaceDimParam;
};

} // namespace Plato

#endif

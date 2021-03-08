#ifndef PLATO_PLASTICITY_HPP
#define PLATO_PLASTICITY_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "SimplexPlasticity.hpp"
#include "J2PlasticityLocalResidual.hpp"
#include "AnalyzeMacros.hpp"

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
    createLocalVectorFunctionInc(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputParams
    )
    {
        if(aInputParams.isSublist("Material Models") == false)
        {
            THROWERR("'Material Models' Sublist is not defined.")
        }
        Teuchos::ParameterList tMaterialModelsList = aInputParams.sublist("Material Models");
        Teuchos::ParameterList tMaterialModelList  = tMaterialModelsList.sublist(aSpatialDomain.getMaterialName());
        if(tMaterialModelList.isSublist("Plasticity Model") == false)
        {
            THROWERR("Plasticity Model Sublist is not defined.")
        }

        auto tPlasticityParamList = tMaterialModelList.get<Teuchos::ParameterList>("Plasticity Model");

        if(tPlasticityParamList.isSublist("J2 Plasticity"))
        {
          constexpr Plato::OrdinalType tSpaceDim = EvaluationType::SpatialDim;
          return std::make_shared
            <J2PlasticityLocalResidual<EvaluationType, Plato::SimplexPlasticity<tSpaceDim>>>
            (aSpatialDomain, aDataMap, aInputParams);
        }
        else
        {
          THROWERR("Unknown Plasticity Model.  Options are: J2 Plasticity.  User is advised to select one of the available options.")
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

#ifndef PLATO_ELECTROMECHANICS_HPP
#define PLATO_ELECTROMECHANICS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "Simplex.hpp"
#include "SimplexElectromechanics.hpp"

#include "elliptic/AbstractVectorFunction.hpp"
#include "elliptic/InternalElectroelasticEnergy.hpp"
#include "elliptic/ElectroelastostaticResidual.hpp"
#include "elliptic/EMStressPNorm.hpp"

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

namespace ElectromechanicsFactory
{

/******************************************************************************/
struct FunctionFactory
{
    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aParamList, 
              std::string              aStrVectorFunctionType
    )
    /******************************************************************************/
    {

        if(aStrVectorFunctionType == "Elliptic")
        {
            auto tPenaltyParams = aParamList.sublist("Elliptic").sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::Elliptic::ElectroelastostaticResidual<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::Elliptic::ElectroelastostaticResidual<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::Elliptic::ElectroelastostaticResidual<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            }
            if(tPenaltyType == "NoPenalty")
            {
                return std::make_shared<Plato::Elliptic::ElectroelastostaticResidual<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aParamList, tPenaltyParams);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
        }
    }
    /******************************************************************************/
    template<typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aParamList, 
              std::string              aStrScalarFunctionType,
              std::string              aStrScalarFunctionName
    )
    /******************************************************************************/
    {

        if(aStrScalarFunctionType == "Internal Electroelastic Energy")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionName).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::Elliptic::InternalElectroelasticEnergy<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::Elliptic::InternalElectroelasticEnergy<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::Elliptic::InternalElectroelasticEnergy<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            if(tPenaltyType == "NoPenalty")
            {
                return std::make_shared<Plato::Elliptic::InternalElectroelasticEnergy<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else if(aStrScalarFunctionType == "Stress P-Norm")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionName).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::Elliptic::EMStressPNorm<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::Elliptic::EMStressPNorm<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::Elliptic::EMStressPNorm<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            if(tPenaltyType == "NoPenalty")
            {
                return std::make_shared<Plato::Elliptic::EMStressPNorm<EvaluationType, Plato::NoPenalty>>
                    (aSpatialDomain, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
    }
}; // struct FunctionFactory

} // namespace ElectromechanicsFactory

template<Plato::OrdinalType SpaceDimParam>
class Electromechanics: public Plato::SimplexElectromechanics<SpaceDimParam>
{
public:
    typedef Plato::ElectromechanicsFactory::FunctionFactory FunctionFactory;
    using SimplexT = SimplexElectromechanics<SpaceDimParam>;
    static constexpr int SpaceDim = SpaceDimParam;
};

} // namespace Plato

#endif

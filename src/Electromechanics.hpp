#ifndef PLATO_ELECTROMECHANICS_HPP
#define PLATO_ELECTROMECHANICS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "Simplex.hpp"
#include "SimplexElectromechanics.hpp"
#include "AbstractVectorFunction.hpp"
#include "ElectroelastostaticResidual.hpp"
#include "InternalElectroelasticEnergy.hpp"
#include "Volume.hpp"
#include "EMStressPNorm.hpp"
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
    std::shared_ptr<Plato::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(Omega_h::Mesh& aMesh, 
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap, 
                         Teuchos::ParameterList& aParamList, 
                         std::string aStrVectorFunctionType)
    /******************************************************************************/
    {

        if(aStrVectorFunctionType == "Electroelastostatics")
        {
            auto tPenaltyParams = aParamList.sublist("Electroelastostatics").sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::ElectroelastostaticResidual<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::ElectroelastostaticResidual<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::ElectroelastostaticResidual<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
            }
            if(tPenaltyType == "NoPenalty")
            {
                return std::make_shared<Plato::ElectroelastostaticResidual<EvaluationType, Plato::NoPenalty>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams);
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
    std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
    createScalarFunction(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap, 
                         Teuchos::ParameterList & aParamList, 
                         std::string aStrScalarFunctionType,
                         std::string aStrScalarFunctionName)
    /******************************************************************************/
    {

        if(aStrScalarFunctionType == "Internal Electroelastic Energy")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionName).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::InternalElectroelasticEnergy<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::InternalElectroelasticEnergy<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::InternalElectroelasticEnergy<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            if(tPenaltyType == "NoPenalty")
            {
                return std::make_shared<Plato::InternalElectroelasticEnergy<EvaluationType, Plato::NoPenalty>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
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
                return std::make_shared<Plato::EMStressPNorm<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::EMStressPNorm<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::EMStressPNorm<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            if(tPenaltyType == "NoPenalty")
            {
                return std::make_shared<Plato::EMStressPNorm<EvaluationType, Plato::NoPenalty>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else if(aStrScalarFunctionType == "Volume")
        {
            auto tPenaltyParams = aParamList.sublist(aStrScalarFunctionName).sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::Volume<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::Volume<EvaluationType, Plato::RAMP>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::Volume<EvaluationType, Plato::Heaviside>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            if(tPenaltyType == "NoPenalty")
            {
                return std::make_shared<Plato::Volume<EvaluationType, Plato::NoPenalty>>(aMesh, aMeshSets, aDataMap, aParamList, tPenaltyParams, aStrScalarFunctionName);
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

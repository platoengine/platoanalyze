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
    createVectorFunction(Omega_h::Mesh& aMesh, 
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap, 
                         Teuchos::ParameterList& aProblemParams, 
                         std::string aStrVectorFunctionType)
    /******************************************************************************/
    {

        if(aStrVectorFunctionType == "Elliptic")
        {
            auto tPenaltyParams = aProblemParams.sublist("Elliptic").sublist("Penalty Function");
            std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::Elliptic::ElectroelastostaticResidual<EvaluationType, Plato::MSIMP>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::Elliptic::ElectroelastostaticResidual<EvaluationType, Plato::RAMP>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tPenaltyParams);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::Elliptic::ElectroelastostaticResidual<EvaluationType, Plato::Heaviside>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tPenaltyParams);
            }
            if(tPenaltyType == "NoPenalty")
            {
                return std::make_shared<Plato::Elliptic::ElectroelastostaticResidual<EvaluationType, Plato::NoPenalty>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tPenaltyParams);
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
    createScalarFunction(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap& aDataMap, 
                         Teuchos::ParameterList & aProblemParams, 
                         std::string aStrScalarFunctionType,
                         std::string aStrScalarFunctionName)
    /******************************************************************************/
    {

        auto tPenaltyParams = aProblemParams.sublist("Criteria").sublist(aStrScalarFunctionName).sublist("Penalty Function");
        std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
        if(aStrScalarFunctionType == "Internal Electroelastic Energy")
        {
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::Elliptic::InternalElectroelasticEnergy<EvaluationType, Plato::MSIMP>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::Elliptic::InternalElectroelasticEnergy<EvaluationType, Plato::RAMP>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::Elliptic::InternalElectroelasticEnergy<EvaluationType, Plato::Heaviside>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tPenaltyParams, aStrScalarFunctionName);
            }
            if(tPenaltyType == "NoPenalty")
            {
                return std::make_shared<Plato::Elliptic::InternalElectroelasticEnergy<EvaluationType, Plato::NoPenalty>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else if(aStrScalarFunctionType == "Stress P-Norm")
        {
            if(tPenaltyType == "SIMP")
            {
                return std::make_shared<Plato::Elliptic::EMStressPNorm<EvaluationType, Plato::MSIMP>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "RAMP")
            {
                return std::make_shared<Plato::Elliptic::EMStressPNorm<EvaluationType, Plato::RAMP>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tPenaltyParams, aStrScalarFunctionName);
            }
            else 
            if(tPenaltyType == "Heaviside")
            {
                return std::make_shared<Plato::Elliptic::EMStressPNorm<EvaluationType, Plato::Heaviside>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tPenaltyParams, aStrScalarFunctionName);
            }
            else
            if(tPenaltyType == "NoPenalty")
            {
                return std::make_shared<Plato::Elliptic::EMStressPNorm<EvaluationType, Plato::NoPenalty>>
                         (aMesh, aMeshSets, aDataMap, aProblemParams, tPenaltyParams, aStrScalarFunctionName);
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

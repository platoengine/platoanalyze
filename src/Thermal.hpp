#ifndef PLATO_THERMAL_HPP
#define PLATO_THERMAL_HPP

#include "Simplex.hpp"
#include "SimplexThermal.hpp"
#include "AbstractVectorFunction.hpp"
#include "AbstractVectorFunctionInc.hpp"
#include "AbstractScalarFunctionInc.hpp"
#include "ThermostaticResidual.hpp"
#include "HeatEquationResidual.hpp"
#include "InternalThermalEnergy.hpp"
#include "TemperatureAverage.hpp"
#include "ThermalFluxRate.hpp"
#include "FluxPNorm.hpp"
#include "Volume.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"


namespace Plato {

namespace ThermalFactory {
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
struct FunctionFactory{
/******************************************************************************/
  template <typename EvaluationType>
  std::shared_ptr<AbstractVectorFunction<EvaluationType>>
  createVectorFunction(
    Omega_h::Mesh& aMesh,
    Omega_h::MeshSets& aMeshSets, 
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aParamList,
    std::string strVectorFunctionType )
  {

    if( strVectorFunctionType == "Thermostatics" ){
      auto penaltyParams = aParamList.sublist(strVectorFunctionType).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<ThermostaticResidual<EvaluationType, Plato::MSIMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<ThermostaticResidual<EvaluationType, Plato::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<ThermostaticResidual<EvaluationType, Plato::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else {
      throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
    }
  }
  template <typename EvaluationType>
  std::shared_ptr<AbstractVectorFunctionInc<EvaluationType>>
  createVectorFunctionInc(
    Omega_h::Mesh& aMesh,
    Omega_h::MeshSets& aMeshSets, 
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aParamList,
    std::string strVectorFunctionType )
  {
    if( strVectorFunctionType == "Heat Equation" ){
      auto penaltyParams = aParamList.sublist(strVectorFunctionType).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<HeatEquationResidual<EvaluationType, Plato::MSIMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<HeatEquationResidual<EvaluationType, Plato::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<HeatEquationResidual<EvaluationType, Plato::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else {
      throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
    }
  }
  template <typename EvaluationType>
  std::shared_ptr<Plato::AbstractScalarFunction<EvaluationType>>
  createScalarFunction( 
    Omega_h::Mesh& aMesh,
    Omega_h::MeshSets& aMeshSets,
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aParamList,
    std::string strScalarFunctionType,
    std::string strScalarFunctionName )
  {

    if( strScalarFunctionType == "Internal Thermal Energy" ){
      auto penaltyParams = aParamList.sublist(strScalarFunctionName).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<Plato::InternalThermalEnergy<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<Plato::InternalThermalEnergy<EvaluationType, Plato::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<Plato::InternalThermalEnergy<EvaluationType, Plato::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else
    if( strScalarFunctionType == "Flux P-Norm" ){
      auto penaltyParams = aParamList.sublist(strScalarFunctionName).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<Plato::FluxPNorm<EvaluationType, Plato::MSIMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<Plato::FluxPNorm<EvaluationType, Plato::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<Plato::FluxPNorm<EvaluationType, Plato::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else 
    if( strScalarFunctionType == "Volume" ){
      auto penaltyParams = aParamList.sublist(strScalarFunctionName).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<Plato::Volume<EvaluationType, Plato::MSIMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams,strScalarFunctionName);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<Plato::Volume<EvaluationType, Plato::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams,strScalarFunctionName);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<Plato::Volume<EvaluationType, Plato::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams,strScalarFunctionName);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else {
      throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
    }
  }
  template <typename EvaluationType>
  std::shared_ptr<Plato::AbstractScalarFunctionInc<EvaluationType>>
  createScalarFunctionInc( 
    Omega_h::Mesh& aMesh,
    Omega_h::MeshSets& aMeshSets,
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aParamList,
    std::string strScalarFunctionType,
    std::string strScalarFunctionName )
  {
    if( strScalarFunctionType == "Thermal Flux Rate" ){
      return std::make_shared<Plato::ThermalFluxRate<EvaluationType>>(aMesh, aMeshSets, aDataMap,aParamList);
    } else
    if( strScalarFunctionType == "Internal Thermal Energy" ){
      auto penaltyParams = aParamList.sublist(strScalarFunctionName).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<Plato::InternalThermalEnergyInc<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<Plato::InternalThermalEnergyInc<EvaluationType, Plato::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<Plato::InternalThermalEnergyInc<EvaluationType, Plato::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else
    if( strScalarFunctionType == "Temperature Average" ){
      auto penaltyParams = aParamList.sublist(strScalarFunctionName).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<Plato::TemperatureAverageInc<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<Plato::TemperatureAverageInc<EvaluationType, Plato::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<Plato::TemperatureAverageInc<EvaluationType, Plato::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else {
      throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
    }
  }
};

} // namespace ThermalFactory

template <Plato::OrdinalType SpaceDimParam>
class Thermal : public Plato::SimplexThermal<SpaceDimParam> {
  public:
    typedef Plato::ThermalFactory::FunctionFactory<SpaceDimParam> FunctionFactory;
    using SimplexT = SimplexThermal<SpaceDimParam>;
    static constexpr Plato::OrdinalType SpaceDim = SpaceDimParam;
};
// class Thermal

} //namespace Plato

#endif

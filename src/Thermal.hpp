#ifndef PLATO_THERMAL_HPP
#define PLATO_THERMAL_HPP

#include "Simplex.hpp"
#include "SimplexThermal.hpp"

#include "parabolic/AbstractVectorFunction.hpp"
#include "parabolic/AbstractScalarFunction.hpp"
#include "parabolic/HeatEquationResidual.hpp"
#include "parabolic/InternalThermalEnergy.hpp"
#include "parabolic/TemperatureAverage.hpp"
#include "parabolic/ThermalFluxRate.hpp"

#include "elliptic/AbstractVectorFunction.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/ThermostaticResidual.hpp"
#include "elliptic/InternalThermalEnergy.hpp"
#include "elliptic/FluxPNorm.hpp"

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

namespace Plato {

namespace ThermalFactory {
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
struct FunctionFactory{
/******************************************************************************/
    template <typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<EvaluationType>>
    createVectorFunction(
        const Plato::SpatialDomain & aSpatialDomain,
        Plato::DataMap             & aDataMap,
        Teuchos::ParameterList     & aParamList,
        std::string                  strVectorFunctionType
    )
    {

        if( strVectorFunctionType == "Elliptic" )
        {
            auto penaltyParams = aParamList.sublist(strVectorFunctionType).sublist("Penalty Function");
            std::string penaltyType = penaltyParams.get<std::string>("Type");
            if( penaltyType == "SIMP" )
            {
                return std::make_shared<Plato::Elliptic::ThermostaticResidual<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aParamList, penaltyParams);
            }
            else
            if( penaltyType == "RAMP" )
            {
                return std::make_shared<Plato::Elliptic::ThermostaticResidual<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aParamList, penaltyParams);
            }
            else
            if( penaltyType == "Heaviside" )
            {
                return std::make_shared<Plato::Elliptic::ThermostaticResidual<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aParamList, penaltyParams);
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
/*
  template <typename EvaluationType>
  std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<EvaluationType>>
  createVectorFunctionParabolic(
    Omega_h::Mesh& aMesh,
    Omega_h::MeshSets& aMeshSets, 
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aParamList,
    std::string strVectorFunctionType )
  {
    if( strVectorFunctionType == "Parabolic" ){
      auto penaltyParams = aParamList.sublist(strVectorFunctionType).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<Plato::Parabolic::HeatEquationResidual<EvaluationType, Plato::MSIMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<Plato::Parabolic::HeatEquationResidual<EvaluationType, Plato::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<Plato::Parabolic::HeatEquationResidual<EvaluationType, Plato::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else {
      throw std::runtime_error("Unknown 'PDE Constraint' specified in 'Plato Problem' ParameterList");
    }
  }
*/

    template <typename EvaluationType>
    std::shared_ptr<Plato::Elliptic::AbstractScalarFunction<EvaluationType>>
    createScalarFunction( 
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string strScalarFunctionType,
              std::string strScalarFunctionName )
    {

        if( strScalarFunctionType == "Internal Thermal Energy" )
        {
            auto penaltyParams = aParamList.sublist(strScalarFunctionName).sublist("Penalty Function");

            std::string penaltyType = penaltyParams.get<std::string>("Type");
            if( penaltyType == "SIMP" )
            {
                return std::make_shared<Plato::Elliptic::InternalThermalEnergy<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aParamList, penaltyParams, strScalarFunctionName);
            }
            else
            if( penaltyType == "RAMP" )
            {
                return std::make_shared<Plato::Elliptic::InternalThermalEnergy<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aParamList, penaltyParams, strScalarFunctionName);
            }
            else
            if( penaltyType == "Heaviside" )
            {
                return std::make_shared<Plato::Elliptic::InternalThermalEnergy<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aParamList, penaltyParams, strScalarFunctionName);
            }
            else
            {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        } else
        if( strScalarFunctionType == "Flux P-Norm" )
        {
            auto penaltyParams = aParamList.sublist(strScalarFunctionName).sublist("Penalty Function");

            std::string penaltyType = penaltyParams.get<std::string>("Type");
            if( penaltyType == "SIMP" )
            {
                return std::make_shared<Plato::Elliptic::FluxPNorm<EvaluationType, Plato::MSIMP>>
                    (aSpatialDomain, aDataMap, aParamList, penaltyParams, strScalarFunctionName);
            }
            else
            if( penaltyType == "RAMP" )
            {
                return std::make_shared<Plato::Elliptic::FluxPNorm<EvaluationType, Plato::RAMP>>
                    (aSpatialDomain, aDataMap, aParamList, penaltyParams, strScalarFunctionName);
            } else
            if( penaltyType == "Heaviside" )
            {
                return std::make_shared<Plato::Elliptic::FluxPNorm<EvaluationType, Plato::Heaviside>>
                    (aSpatialDomain, aDataMap, aParamList, penaltyParams, strScalarFunctionName);
            } else {
                throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
            }
        }
        else
        {
            throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
        }
    }

/*
  template <typename EvaluationType>
  std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<EvaluationType>>
  createScalarFunctionParabolic( 
    Omega_h::Mesh& aMesh,
    Omega_h::MeshSets& aMeshSets,
    Plato::DataMap& aDataMap,
    Teuchos::ParameterList& aParamList,
    std::string strScalarFunctionType,
    std::string strScalarFunctionName )
  {
    if( strScalarFunctionType == "Thermal Flux Rate" ){
      return std::make_shared<Plato::Parabolic::ThermalFluxRate<EvaluationType>>(aMesh, aMeshSets, aDataMap,aParamList);
    } else
    if( strScalarFunctionType == "Internal Thermal Energy" ){
      auto penaltyParams = aParamList.sublist(strScalarFunctionName).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<Plato::Parabolic::InternalThermalEnergy<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<Plato::Parabolic::InternalThermalEnergy<EvaluationType, Plato::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<Plato::Parabolic::InternalThermalEnergy<EvaluationType, Plato::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else
    if( strScalarFunctionType == "Temperature Average" ){
      auto penaltyParams = aParamList.sublist(strScalarFunctionName).sublist("Penalty Function");
      std::string penaltyType = penaltyParams.get<std::string>("Type");
      if( penaltyType == "SIMP" ){
        return std::make_shared<Plato::Parabolic::TemperatureAverage<EvaluationType, Plato::MSIMP>>(aMesh, aMeshSets, aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else
      if( penaltyType == "RAMP" ){
        return std::make_shared<Plato::Parabolic::TemperatureAverage<EvaluationType, Plato::RAMP>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else
      if( penaltyType == "Heaviside" ){
        return std::make_shared<Plato::Parabolic::TemperatureAverage<EvaluationType, Plato::Heaviside>>(aMesh,aMeshSets,aDataMap,aParamList,penaltyParams, strScalarFunctionName);
      } else {
        throw std::runtime_error("Unknown 'Type' specified in 'Penalty Function' ParameterList");
      }
    } else {
      throw std::runtime_error("Unknown 'Objective' specified in 'Plato Problem' ParameterList");
    }
  }
*/
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

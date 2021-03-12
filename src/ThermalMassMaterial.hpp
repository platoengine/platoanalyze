#pragma once

#include <Teuchos_ParameterList.hpp>
#include "PlatoStaticsTypes.hpp"
#include "MaterialModel.hpp"

namespace Plato {

  /******************************************************************************/
  /*!
    \brief Base class for ThermalMass material models
  */
    template<int SpatialDim>
    class ThermalMassMaterial : public MaterialModel<SpatialDim>
  /******************************************************************************/
  {
  
    public:
      ThermalMassMaterial(const Teuchos::ParameterList& paramList);
  };

  /******************************************************************************/
  template<int SpatialDim>
  ThermalMassMaterial<SpatialDim>::
  ThermalMassMaterial(const Teuchos::ParameterList& paramList) : MaterialModel<SpatialDim>(paramList)
  /******************************************************************************/
  {
      this->parseScalar("Mass Density", paramList);
      this->parseScalar("Specific Heat", paramList);
  }

  /******************************************************************************/
  /*!
    \brief Factory for creating material models
  */
    template<int SpatialDim>
    class ThermalMassModelFactory
  /******************************************************************************/
  {
    public:
      ThermalMassModelFactory(const Teuchos::ParameterList& paramList) : mParamList(paramList) {}
      Teuchos::RCP<Plato::MaterialModel<SpatialDim>> create(std::string aModelName);
    private:
      const Teuchos::ParameterList& mParamList;
  };

  /******************************************************************************/
  template<int SpatialDim>
  Teuchos::RCP<MaterialModel<SpatialDim>>
  ThermalMassModelFactory<SpatialDim>::create(std::string aModelName)
  /******************************************************************************/
  {
      if (!mParamList.isSublist("Material Models"))
      {
          REPORT("'Material Models' list not found! Returning 'nullptr'");
          return Teuchos::RCP<Plato::MaterialModel<SpatialDim>>(nullptr);
      }
      else
      {
          auto tModelsParamList = mParamList.get<Teuchos::ParameterList>("Material Models");

          if (!tModelsParamList.isSublist(aModelName))
          {
              std::stringstream ss;
              ss << "Requested a material model ('" << aModelName << "') that isn't defined";
              THROWERR(ss.str());
          }

          auto tModelParamList = tModelsParamList.sublist(aModelName);

          if( tModelParamList.isSublist("Thermal Mass") )
          {
              return Teuchos::rcp(new Plato::ThermalMassMaterial<SpatialDim>(tModelParamList.sublist("Thermal Mass")));
          }
          else
          {
              THROWERR("Expected 'Thermal Mass' ParameterList");
          }
      }
  }
} // namespace Plato

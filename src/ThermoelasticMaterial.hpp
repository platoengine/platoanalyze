#pragma once

#include <Teuchos_ParameterList.hpp>
#include "PlatoStaticsTypes.hpp"
#include "MaterialModel.hpp"

namespace Plato {

  /******************************************************************************/
  /*!
    \brief Base class for Thermoelastic material models
  */
    template<int SpatialDim>
    class ThermoelasticMaterial : public MaterialModel<SpatialDim>
  /******************************************************************************/
  {
  
    public:
      ThermoelasticMaterial(const Teuchos::ParameterList& paramList);

    private:
      void parseElasticStiffness(const Teuchos::ParameterList& paramList);
  };

  /******************************************************************************/
  template<int SpatialDim>
  ThermoelasticMaterial<SpatialDim>::
  ThermoelasticMaterial(const Teuchos::ParameterList& paramList) : MaterialModel<SpatialDim>(paramList)
  /******************************************************************************/
  {
      this->parseElasticStiffness(paramList);
      this->parseTensor("Thermal Expansivity", paramList);
      this->parseTensor("Thermal Conductivity", paramList);

      this->parseScalarConstant("Reference Temperature", paramList, 23.0);
      this->parseScalarConstant("Temperature Scaling", paramList, 1.0);
  }

  /******************************************************************************/
  template<int SpatialDim>
  void ThermoelasticMaterial<SpatialDim>::
  parseElasticStiffness(const Teuchos::ParameterList& paramList)
  /******************************************************************************/
  {
      auto tParams = paramList.sublist("Elastic Stiffness");
      if (tParams.isSublist("Youngs Modulus"))
      {
          this->setRank4VoigtFunctor("Elastic Stiffness", Plato::IsotropicStiffnessFunctor<SpatialDim>(tParams));
      }
      else
      if (tParams.isType<Plato::Scalar>("Youngs Modulus"))
      {
          this->setRank4VoigtConstant("Elastic Stiffness", Plato::IsotropicStiffnessConstant<SpatialDim>(tParams));
      }
      else
      {
          this->parseRank4Voigt("Elastic Stiffness", tParams);
      }
  }

  /******************************************************************************/
  /*!
    \brief Factory for creating material models
  */
    template<int SpatialDim>
    class ThermoelasticModelFactory
  /******************************************************************************/
  {
    public:
      ThermoelasticModelFactory(const Teuchos::ParameterList& paramList) : mParamList(paramList) {}
      Teuchos::RCP<Plato::MaterialModel<SpatialDim>> create();
    private:
      const Teuchos::ParameterList& mParamList;
  };

  /******************************************************************************/
  template<int SpatialDim>
  Teuchos::RCP<MaterialModel<SpatialDim>>
  ThermoelasticModelFactory<SpatialDim>::create()
  /******************************************************************************/
  {
    auto modelParamList = mParamList.get<Teuchos::ParameterList>("Material Model");

    if( modelParamList.isSublist("Thermoelastic") )
    {
      return Teuchos::rcp(new Plato::ThermoelasticMaterial<SpatialDim>(modelParamList.sublist("Thermoelastic")));
    }
    else
    THROWERR("Expected 'Thermoelastic' ParameterList");
  }

} // namespace Plato

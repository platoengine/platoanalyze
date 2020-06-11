#ifndef LINEARTHERMALMATERIAL_HPP
#define LINEARTHERMALMATERIAL_HPP

#include <Teuchos_ParameterList.hpp>
#include "MaterialModel.hpp"

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*!
 \brief Base class for Linear Thermal material models
 */
template<int SpatialDim>
class ThermalConductionModel : public MaterialModel<SpatialDim>
/******************************************************************************/
{
  public:
    ThermalConductionModel(const Teuchos::ParameterList& paramList);
};

/******************************************************************************/
template<int SpatialDim>
ThermalConductionModel<SpatialDim>::
ThermalConductionModel(const Teuchos::ParameterList& paramList) : MaterialModel<SpatialDim>(paramList)
/******************************************************************************/
{
    this->parseTensor("Thermal Conductivity", paramList);
}

/******************************************************************************/
/*!
 \brief Factory for creating material models
 */
template<int SpatialDim>
class ThermalConductionModelFactory
/******************************************************************************/
{
public:
    ThermalConductionModelFactory(const Teuchos::ParameterList& aParamList) :
            mParamList(aParamList)
    {
    }
    Teuchos::RCP<MaterialModel<SpatialDim>> create();
private:
    const Teuchos::ParameterList& mParamList;
};
/******************************************************************************/
template<int SpatialDim>
Teuchos::RCP<MaterialModel<SpatialDim>> ThermalConductionModelFactory<SpatialDim>::create()
/******************************************************************************/
{
    auto tModelParamList = mParamList.get < Teuchos::ParameterList > ("Material Model");

    if(tModelParamList.isSublist("Thermal Conduction"))
    {
        return Teuchos::rcp(new ThermalConductionModel<SpatialDim>(tModelParamList.sublist("Thermal Conduction")));
    }
    else
    THROWERR("Expected 'Thermal Conduction' ParameterList");
}

}

#endif

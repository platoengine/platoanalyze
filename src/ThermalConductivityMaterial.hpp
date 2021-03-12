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
    Teuchos::RCP<MaterialModel<SpatialDim>> create(std::string aModelName);
private:
    const Teuchos::ParameterList& mParamList;
};
/******************************************************************************/
template<int SpatialDim>
Teuchos::RCP<MaterialModel<SpatialDim>>
ThermalConductionModelFactory<SpatialDim>::create(std::string aModelName)
/******************************************************************************/
{
    if (!mParamList.isSublist("Material Models"))
    {
        REPORT("'Material Models' list not found! Returning 'nullptr'");
        return Teuchos::RCP<Plato::MaterialModel<SpatialDim>>(nullptr);
    }
    else
    {
        auto tModelsParamList = mParamList.get < Teuchos::ParameterList > ("Material Models");

        if (!tModelsParamList.isSublist(aModelName))
        {
            std::stringstream ss;
            ss << "Requested a material model ('" << aModelName << "') that isn't defined";
            THROWERR(ss.str());
        }

        auto tModelParamList = tModelsParamList.sublist(aModelName);
        if(tModelParamList.isSublist("Thermal Conduction"))
        {
            return Teuchos::rcp(new ThermalConductionModel<SpatialDim>(tModelParamList.sublist("Thermal Conduction")));
        }
        else
        THROWERR("Expected 'Thermal Conduction' ParameterList");
    }
}

}

#endif

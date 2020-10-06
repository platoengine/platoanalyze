#ifndef LINEARELECTROELASTICMATERIAL_HPP
#define LINEARELECTROELASTICMATERIAL_HPP

#include <Omega_h_matrix.hpp>
#include <Teuchos_ParameterList.hpp>

#include "PlatoStaticsTypes.hpp"

namespace Plato {

/******************************************************************************/
/*!
  \brief Base class for Linear Electroelastic material models
*/
  template<int SpatialDim>
  class LinearElectroelasticMaterial
/******************************************************************************/
{
  protected:
    static constexpr auto mNumVoigtTerms = (SpatialDim == 3) ? 6 : 
                                           ((SpatialDim == 2) ? 3 :
                                          (((SpatialDim == 1) ? 1 : 0)));
    static_assert(mNumVoigtTerms, "SpatialDim must be 1, 2, or 3.");

    Omega_h::Matrix<mNumVoigtTerms,mNumVoigtTerms> mCellStiffness;
    Omega_h::Matrix<SpatialDim, mNumVoigtTerms> mCellPiezoelectricCoupling;
    Omega_h::Matrix<SpatialDim, SpatialDim> mCellPermittivity;

    Plato::Scalar mAlpha;
  
  public:
    LinearElectroelasticMaterial();
    decltype(mCellStiffness)             getStiffnessMatrix()    const {return mCellStiffness;}
    decltype(mCellPiezoelectricCoupling) getPiezoMatrix()        const {return mCellPiezoelectricCoupling;}
    decltype(mCellPermittivity)          getPermittivityMatrix() const {return mCellPermittivity;}
    decltype(mAlpha)                     getAlpha()              const {return mAlpha;}
};

/******************************************************************************/
template<int SpatialDim>
LinearElectroelasticMaterial<SpatialDim>::
LinearElectroelasticMaterial()
/******************************************************************************/
{
  for(int i=0; i<mNumVoigtTerms; i++)
    for(int j=0; j<mNumVoigtTerms; j++)
      mCellStiffness(i,j) = 0.0;

  for(int i=0; i<SpatialDim; i++)
    for(int j=0; j<mNumVoigtTerms; j++)
      mCellPiezoelectricCoupling(i,j) = 0.0;

  for(int i=0; i<SpatialDim; i++)
    for(int j=0; j<SpatialDim; j++)
      mCellPermittivity(i,j) = 0.0;

  mAlpha = 1.0;
}

/******************************************************************************/
/*!
  \brief Derived class for isotropic linear thermoelastic material model
*/
  template<int SpatialDim>
  class IsotropicLinearElectroelasticMaterial : public LinearElectroelasticMaterial<SpatialDim>
/******************************************************************************/
{
  public:
    IsotropicLinearElectroelasticMaterial(const Teuchos::ParameterList& paramList);
    virtual ~IsotropicLinearElectroelasticMaterial(){}
};
// class IsotropicLinearElectroelasticMaterial

/******************************************************************************/
/*!
  \brief Factory for creating material models
*/
  template<int SpatialDim>
  class ElectroelasticModelFactory
/******************************************************************************/
{
  public:
    ElectroelasticModelFactory(const Teuchos::ParameterList& paramList) : mParamList(paramList) {}
    Teuchos::RCP<Plato::LinearElectroelasticMaterial<SpatialDim>> create(std::string aModelName);
  private:
    const Teuchos::ParameterList& mParamList;
};
/******************************************************************************/
template<int SpatialDim>
Teuchos::RCP<LinearElectroelasticMaterial<SpatialDim>>
ElectroelasticModelFactory<SpatialDim>::create(std::string aModelName)
/******************************************************************************/
{
    if (!mParamList.isSublist("Material Models"))
    {
        REPORT("'Material Models' list not found! Returning 'nullptr'");
        return Teuchos::RCP<Plato::LinearElectroelasticMaterial<SpatialDim>>(nullptr);
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

        if( tModelParamList.isSublist("Isotropic Linear Electroelastic") )
        {
          return Teuchos::rcp(new Plato::IsotropicLinearElectroelasticMaterial<SpatialDim>
              (tModelParamList.sublist("Isotropic Linear Electroelastic")));
        }
        return Teuchos::RCP<Plato::LinearElectroelasticMaterial<SpatialDim>>(nullptr);
    }
}

} // namespace Plato

#endif

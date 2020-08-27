#ifndef THERMAL_FLUX_HPP
#define THERMAL_FLUX_HPP

#include "PlatoStaticsTypes.hpp"
#include "MaterialModel.hpp"

namespace Plato
{

/******************************************************************************/
/*! Thermal flux functor.
  
    given a temperature gradient, compute the thermal flux
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class ThermalFlux
{
  private:
    Plato::MaterialModelType mModelType;

    Plato::TensorFunctor<SpaceDim> mConductivityFunctor;

    Plato::TensorConstant<SpaceDim> mConductivityConstant;

  public:

    ThermalFlux(const Teuchos::RCP<Plato::MaterialModel<SpaceDim>> aMaterialModel)
    {
        mModelType = aMaterialModel->type();
        if (mModelType == Plato::MaterialModelType::Nonlinear)
        {
            mConductivityFunctor = aMaterialModel->getTensorFunctor("Thermal Conductivity");
        } else
        if (mModelType == Plato::MaterialModelType::Linear)
        {
            mConductivityConstant = aMaterialModel->getTensorConstant("Thermal Conductivity");
        }
    }


    template<typename TScalarType, typename TGradScalarType, typename TFluxScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT<TFluxScalarType> tflux,
                Plato::ScalarMultiVectorT<TGradScalarType> tgrad,
                Plato::ScalarVectorT     <TScalarType>     temperature) const {

      // compute thermal flux
      //
      if (mModelType == Plato::MaterialModelType::Linear)
      {
        for( Plato::OrdinalType iDim=0; iDim<SpaceDim; iDim++){
          tflux(cellOrdinal,iDim) = 0.0;
          for( Plato::OrdinalType jDim=0; jDim<SpaceDim; jDim++){
            tflux(cellOrdinal,iDim) -= tgrad(cellOrdinal,jDim)*mConductivityConstant(iDim, jDim);
          }
        }
      } else
      if (mModelType == Plato::MaterialModelType::Nonlinear)
      {
        TScalarType cellT = temperature(cellOrdinal);
        for( Plato::OrdinalType iDim=0; iDim<SpaceDim; iDim++){
          tflux(cellOrdinal,iDim) = 0.0;
          for( Plato::OrdinalType jDim=0; jDim<SpaceDim; jDim++){
            tflux(cellOrdinal,iDim) -= tgrad(cellOrdinal,jDim)*mConductivityFunctor(cellT, iDim, jDim);
          }
        }
      }
    }

    template<typename TGradScalarType, typename TFluxScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT<TFluxScalarType> tflux,
                Plato::ScalarMultiVectorT<TGradScalarType> tgrad) const {

      // compute thermal flux
      //
      for( Plato::OrdinalType iDim=0; iDim<SpaceDim; iDim++){
        tflux(cellOrdinal,iDim) = 0.0;
        for( Plato::OrdinalType jDim=0; jDim<SpaceDim; jDim++){
          tflux(cellOrdinal,iDim) -= tgrad(cellOrdinal,jDim)*mConductivityConstant(iDim, jDim);
        }
      }
    }
};
// class ThermalFlux

} // namespace Plato
#endif

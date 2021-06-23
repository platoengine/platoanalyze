#ifndef HELMHOLTZ_FLUX_HPP
#define HELMHOLTZ_FLUX_HPP

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************/
/*! Helhomltz flux functor.
  
    given a filtered density gradient, scale by length scale squared
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class HelmholtzFlux
{
  private:
    Plato::Scalar mLengthScale;

  public:

    HelmholtzFlux(const Plato::Scalar aLengthScale) {
      mLengthScale = aLengthScale;

      printf("\n Length Scale input to flux constructor Is: %f \n ",aLengthScale);
      printf("\n Length Scale output to flux constructor Is: %f \n ",mLengthScale);
    }

    template<typename HGradScalarType, typename HFluxScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType cellOrdinal,
                Plato::ScalarMultiVectorT<HFluxScalarType> tflux,
                Plato::ScalarMultiVectorT<HGradScalarType> tgrad) const {

      // scale filtered density gradient
      //
      Plato::Scalar tLengthScaleSquared = mLengthScale*mLengthScale;

      printf("\n Length Scale in flux operator() Is: %f \n ",mLengthScale);

      for( Plato::OrdinalType iDim=0; iDim<SpaceDim; iDim++){
        tflux(cellOrdinal,iDim) = tLengthScaleSquared*tgrad(cellOrdinal,iDim);
      }
    }
};
// class HelmholtzFlux

} // namespace Helmholtz

} // namespace Plato
#endif

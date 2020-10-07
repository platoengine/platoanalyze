#ifndef INERTIAL_CONTENT_HPP
#define INERTIAL_CONTENT_HPP

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Inertial content functor.
  
    given an acceleration vector, compute the inertial content, \rho \bm{a}
*/
/******************************************************************************/
template<int SpaceDim>
class InertialContent
{
  private:
    const Plato::Scalar mCellDensity;
    const Plato::Scalar mRayleighA;

  public:
    InertialContent(const Teuchos::RCP<Plato::LinearElasticMaterial<SpaceDim>> aMaterialModel ) :
            mCellDensity (aMaterialModel->getMassDensity()),
            mRayleighA   (aMaterialModel->getRayleighA()) {}

    template<typename TScalarType, typename TContentScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType aCellOrdinal,
                Plato::ScalarMultiVectorT<TContentScalarType> aContent,
                Plato::ScalarMultiVectorT<TScalarType> aAcceleration) const {

      // compute inertial content
      //
      for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
      {
          aContent(aCellOrdinal, tDimIndex) = aAcceleration(aCellOrdinal, tDimIndex)*mCellDensity;
      }
    }

    template<typename TVelocityType, typename TAccelerationType, typename TContentScalarType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType aCellOrdinal,
                Plato::ScalarMultiVectorT<TContentScalarType> aContent,
                Plato::ScalarMultiVectorT<TVelocityType>      aVelocity,
                Plato::ScalarMultiVectorT<TAccelerationType>  aAcceleration) const {

      // compute inertial content
      //
      for(Plato::OrdinalType tDimIndex = 0; tDimIndex < SpaceDim; tDimIndex++)
      {
          aContent(aCellOrdinal, tDimIndex) = aAcceleration(aCellOrdinal, tDimIndex)*mCellDensity
                                            + aVelocity(aCellOrdinal, tDimIndex)*mCellDensity*mRayleighA;
      }
    }
};
// class InertialContent

} // namespace Plato

#endif

#pragma once

#include "ImplicitFunctors.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "ExpInstMacros.hpp"
#include "BLAS1.hpp"
#include "alg/Basis.hpp"
#include "Plato_TopOptFunctors.hpp"
#include <Omega_h_mesh.hpp>

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename SimplexPhysicsT>
class VolumeAverageCriterionDenominator : 
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;
    static constexpr Plato::OrdinalType mNumNodesPerCell = SimplexPhysicsT::mNumNodesPerCell;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    
    Plato::Scalar mQuadratureWeight;

    Plato::ScalarVector mSpatialWeights; /*!< spatially varying weights */

  public:
    /**************************************************************************/
    VolumeAverageCriterionDenominator(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              std::string            & aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aFunctionName)
    /**************************************************************************/
    {

//TODO quadrature
      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType d = 2; d <= SpaceDim; d++)
      { 
        mQuadratureWeight /= Plato::Scalar(d);
      }

      auto tNumCells = mSpatialDomain.numCells();
      Kokkos::resize(mSpatialWeights, tNumCells);
      Plato::blas1::fill(static_cast<Plato::Scalar>(1.0), mSpatialWeights);
    }

    /******************************************************************************//**
     * \brief Set spatial weights
     * \param [in] aInput scalar vector of spatial weights
    **********************************************************************************/
    void setSpatialWeights(Plato::ScalarVector & aInput) override
    {
        Kokkos::resize(mSpatialWeights, aInput.size());
        Plato::blas1::copy(aInput, mSpatialWeights);
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientWorkset<SpaceDim> tComputeGradient;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell volume", tNumCells);

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient", tNumCells, mNumNodesPerCell, SpaceDim);

      auto quadratureWeight = mQuadratureWeight;
      auto tSpatialWeights  = mSpatialWeights;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        tComputeGradient(cellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(cellOrdinal) *= quadratureWeight;

        aResult(cellOrdinal) = tCellVolume(cellOrdinal) * tSpatialWeights(cellOrdinal);
      },"Compute Weighted Volume Average Criterion Demoninator");

    }
};
// class VolumeAverageCriterionDenominator

} // namespace Elliptic

} // namespace Plato


#pragma once

#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "Strain.hpp"
#include "LinearStress.hpp"
#include "TensorPNorm.hpp"
#include "ElasticModelFactory.hpp"
#include "ImplicitFunctors.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "ExpInstMacros.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

#include "alg/Basis.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "PlatoMeshExpr.hpp"
#include "UtilsOmegaH.hpp"
#include "alg/Cubature.hpp"
#include "BLAS2.hpp"
#include <Omega_h_expr.hpp>
#include <Omega_h_mesh.hpp>

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class VolAvgStressPNormDenominator : 
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;
    
    using Plato::SimplexMechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::mNumDofsPerCell;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;
    
    Plato::Scalar mQuadratureWeight;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim,1/*number of terms*/,IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;

    std::string mSpatialWeightingFunctionString = "1.0";

    Omega_h::Reals mFxnValues;

    void computeSpatialWeightingValues(const Plato::SpatialDomain & aSpatialDomain)
    {
      // get refCellQuadraturePoints, quadratureWeights
      //
      Plato::OrdinalType tQuadratureDegree = 1;

      Plato::OrdinalType tNumPoints = Plato::Cubature::getNumCubaturePoints(SpaceDim, tQuadratureDegree);

      Kokkos::View<Plato::Scalar**, Plato::Layout, Plato::MemSpace>
          tRefCellQuadraturePoints("ref quadrature points", tNumPoints, SpaceDim);
      Kokkos::View<Plato::Scalar*, Plato::Layout, Plato::MemSpace> tQuadratureWeights("quadrature weights", tNumPoints);

      Plato::Cubature::getCubature(SpaceDim, tQuadratureDegree, tRefCellQuadraturePoints, tQuadratureWeights);

      // get basis values
      //
      Plato::Basis tBasis(SpaceDim);
      Plato::OrdinalType tNumFields = tBasis.basisCardinality();
      Kokkos::View<Plato::Scalar**, Plato::Layout, Plato::MemSpace>
          tRefCellBasisValues("ref basis values", tNumFields, tNumPoints);
      tBasis.getValues(tRefCellQuadraturePoints, tRefCellBasisValues);

      // map points to physical space
      //
      Plato::OrdinalType tNumCells = aSpatialDomain.numCells();
      Kokkos::View<Plato::Scalar***, Plato::Layout, Plato::MemSpace>
          tQuadraturePoints("quadrature points", tNumCells, tNumPoints, SpaceDim);

      Plato::mapPoints<SpaceDim>(aSpatialDomain, tRefCellQuadraturePoints, tQuadraturePoints);

      // get integrand values at quadrature points
      //
      Plato::getFunctionValues<SpaceDim>(tQuadraturePoints, mSpatialWeightingFunctionString, mFxnValues);
    }

  public:
    /**************************************************************************/
    VolAvgStressPNormDenominator(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aFunctionName),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction)
    /**************************************************************************/
    {

//TODO quadrature
      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType d = 2; d <= SpaceDim; d++)
      { 
        mQuadratureWeight /= Plato::Scalar(d);
      }

      auto params = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);

      TensorNormFactory<mNumVoigtTerms, EvaluationType> normFactory;
      mNorm = normFactory.create(params);

      if (params.isType<std::string>("Function"))
        mSpatialWeightingFunctionString = params.get<std::string>("Function");
      
      this->computeSpatialWeightingValues(aSpatialDomain);
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

      using StrainScalarType = 
        typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>,
                            StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight", tNumCells);

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient", tNumCells, mNumNodesPerCell, SpaceDim);
    
      Plato::ScalarMultiVectorT<ResultScalarType>
        tWeightedOne("weighted one", tNumCells, mNumVoigtTerms);
      Plato::blas2::fill(0.0, tWeightedOne);

      Plato::Scalar tOne = 1.0;

      auto quadratureWeight = mQuadratureWeight;
      auto applyWeighting   = mApplyWeighting;
      auto tFxnValues       = mFxnValues;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        tComputeGradient(cellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(cellOrdinal) *= quadratureWeight * tFxnValues[cellOrdinal];

        tWeightedOne(cellOrdinal, 0) = tOne;
        // apply weighting
        //
        applyWeighting(cellOrdinal, tWeightedOne, aControl);

      },"Compute Weighted Stress Pnorm Demoninator");

      mNorm->evaluate(aResult, tWeightedOne, aControl, tCellVolume);

    }

    /**************************************************************************/
    void
    postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar)
    /**************************************************************************/
    {
      mNorm->postEvaluate(resultVector, resultScalar);
    }

    /**************************************************************************/
    void
    postEvaluate( Plato::Scalar& resultValue )
    /**************************************************************************/
    {
      mNorm->postEvaluate(resultValue);
    }
};
// class VolAvgStressPNormDenominator

} // namespace Elliptic

} // namespace Plato


#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC(Plato::Elliptic::VolAvgStressPNormDenominator, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC(Plato::Elliptic::VolAvgStressPNormDenominator, Plato::SimplexMechanics, 3)
#endif

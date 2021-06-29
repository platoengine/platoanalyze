#ifndef HYPERBOLIC_STRESS_P_NORM_HPP
#define HYPERBOLIC_STRESS_P_NORM_HPP

#include "SimplexMechanics.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "Strain.hpp"
#include "LinearStress.hpp"
#include "TensorPNorm.hpp"
#include "ElasticModelFactory.hpp"
#include "ImplicitFunctors.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "hyperbolic/HyperbolicAbstractScalarFunction.hpp"
#include "ToMap.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class StressPNorm : 
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;
    
    using Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMechanics<mSpaceDim>::mNumDofsPerCell;

    using Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType       = typename EvaluationType::StateScalarType;
    using StateDotScalarType    = typename EvaluationType::StateDotScalarType;
    using StateDotDotScalarType = typename EvaluationType::StateDotDotScalarType;
    using ControlScalarType     = typename EvaluationType::ControlScalarType;
    using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
    using ResultScalarType      = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>;
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<mSpaceDim>;

    std::shared_ptr<CubatureType> mCubatureRule;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim,mNumVoigtTerms,IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;

    Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> mMaterialModel;


  public:
    /**************************************************************************/
    StressPNorm(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              Teuchos::ParameterList & aFunctionParams,
        const std::string            & aFunctionName
    ) :
        FunctionBaseType   (aSpatialDomain, aDataMap, aFunctionName),
        mCubatureRule      (std::make_shared<CubatureType>()),
        mIndicatorFunction (aFunctionParams.sublist("Penalty Function")),
        mApplyWeighting    (mIndicatorFunction)
    /**************************************************************************/
    {
      Plato::ElasticModelFactory<mSpaceDim> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());

      TensorNormFactory<mNumVoigtTerms, EvaluationType> normFactory;
      mNorm = normFactory.create(aFunctionParams);
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>       & aState,
        const Plato::ScalarMultiVectorT <StateDotScalarType>    & aStateDot,
        const Plato::ScalarMultiVectorT <StateDotDotScalarType> & aStateDotDot,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      using SimplexPhysics = typename Plato::SimplexMechanics<mSpaceDim>;
      using StrainScalarType =
        typename Plato::fad_type_t<SimplexPhysics, StateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
      Plato::Strain<mSpaceDim>                 tVoigtStrain;
      Plato::LinearStress<EvaluationType,
                          SimplexPhysics>      tVoigtStress(mMaterialModel);

      Plato::ScalarVectorT      <ConfigScalarType> tCellVolume ("cell weight", tNumCells);
      Plato::ScalarArray3DT     <ConfigScalarType> tGradient   ("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);
      Plato::ScalarMultiVectorT <StrainScalarType> tStrain     ("strain", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT <ResultScalarType> tStress     ("stress", tNumCells, mNumVoigtTerms);

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto tApplyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute strain
        //
        tVoigtStrain(aCellOrdinal, tStrain, aState, tGradient);

        // compute stress
        //
        tVoigtStress(aCellOrdinal, tStress, tStrain);
      
        // apply weighting
        //
        tApplyWeighting(aCellOrdinal, tStress, aControl);

      },"Compute Stress");

      mNorm->evaluate(aResult, tStress, aControl, tCellVolume);
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
}; // class StressPNorm

} // namespace Hyperbolic

} // namespace Plato

#endif

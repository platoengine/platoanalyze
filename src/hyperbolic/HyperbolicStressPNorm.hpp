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

    using Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using DisplacementScalarType = typename EvaluationType::DisplacementScalarType;
    using VelocityScalarType     = typename EvaluationType::VelocityScalarType;
    using AccelerationScalarType = typename EvaluationType::AccelerationScalarType;
    using ControlScalarType      = typename EvaluationType::ControlScalarType;
    using ConfigScalarType       = typename EvaluationType::ConfigScalarType;
    using ResultScalarType       = typename EvaluationType::ResultScalarType;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>> mCubatureRule;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim,mNumVoigtTerms,IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;

    Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> mMaterialModel;


  public:
    /**************************************************************************/
    StressPNorm(
        Omega_h::Mesh&          aMesh,
        Omega_h::MeshSets&      aMeshSets,
        Plato::DataMap&         aDataMap, 
        Teuchos::ParameterList& aProblemParams, 
        Teuchos::ParameterList& aPenaltyParams,
        std::string&            aFunctionName
    ) :
        Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
        mCubatureRule      (std::make_shared<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>>()),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction)
    /**************************************************************************/
    {
      Plato::ElasticModelFactory<mSpaceDim> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create();

      auto params = aProblemParams.get<Teuchos::ParameterList>(aFunctionName);

      TensorNormFactory<mNumVoigtTerms, EvaluationType> normFactory;
      mNorm = normFactory.create(params);
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT<typename EvaluationType::DisplacementScalarType> & aState,
        const Plato::ScalarMultiVectorT<typename EvaluationType::VelocityScalarType>     & aStateDot,
        const Plato::ScalarMultiVectorT<typename EvaluationType::AccelerationScalarType> & aStateDotDot,
        const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType>      & aControl,
        const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType>           & aConfig,
        Plato::ScalarVectorT<typename EvaluationType::ResultScalarType>                  & aResult,
        Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto numCells = mMesh.nelems();

      Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
      Plato::Strain<mSpaceDim>                 tVoigtStrain;
      Plato::LinearStress<mSpaceDim>           tVoigtStress(mMaterialModel);

      using StrainScalarType = 
        typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>,
                            DisplacementScalarType, ConfigScalarType>;

      Plato::ScalarVectorT      <ConfigScalarType> tCellVolume ("cell weight",numCells);
      Plato::ScalarArray3DT     <ConfigScalarType> tGradient   ("gradient",numCells,mNumNodesPerCell,mSpaceDim);
      Plato::ScalarMultiVectorT <StrainScalarType> tStrain     ("strain",numCells,mNumVoigtTerms);
      Plato::ScalarMultiVectorT <ResultScalarType> tStress     ("stress",numCells,mNumVoigtTerms);

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto tApplyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
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

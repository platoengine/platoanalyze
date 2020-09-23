#pragma once

#include "elliptic/AbstractScalarFunction.hpp"

#include "SimplexThermomechanics.hpp"
#include "ThermoelasticMaterial.hpp"
#include "SimplexFadTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "TMKinematics.hpp"
#include "TMKinetics.hpp"
#include "TensorPNorm.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ExpInstMacros.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TMStressPNorm : 
  public Plato::SimplexThermomechanics<EvaluationType::SpatialDim>,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    static constexpr int TDofOffset = SpaceDim;
    
    using Plato::SimplexThermomechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerCell;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Teuchos::RCP<Plato::MaterialModel<SpaceDim>> mMaterialModel;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim,mNumVoigtTerms,IndicatorFunctionType> mApplyWeighting;
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;

  public:
    /**************************************************************************/
    TMStressPNorm(Omega_h::Mesh& aMesh,
                  Omega_h::MeshSets& aMeshSets,
                  Plato::DataMap& aDataMap, 
                  Teuchos::ParameterList& aProblemParams, 
                  Teuchos::ParameterList& aPenaltyParams,
                  std::string& aFunctionName) :
              Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
              mIndicatorFunction(aPenaltyParams),
              mApplyWeighting(mIndicatorFunction),
              mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
      Plato::ThermoelasticModelFactory<SpaceDim> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create();

      auto params = aProblemParams.get<Teuchos::ParameterList>(aFunctionName);

      TensorNormFactory<mNumVoigtTerms, EvaluationType> normFactory;
      mNorm = normFactory.create(params);
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto numCells = mMesh.nelems();

      using GradScalarType = 
        typename Plato::fad_type_t<Plato::SimplexThermomechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> computeGradient;
      Plato::TMKinematics<SpaceDim>                  kinematics;
      Plato::TMKinetics<SpaceDim>                    kinetics(mMaterialModel);

      Plato::InterpolateFromNodal<SpaceDim, mNumDofsPerNode, TDofOffset> interpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType> cellVolume("cell weight", numCells);

      Plato::ScalarArray3DT<ConfigScalarType> gradient("gradient", numCells, mNumNodesPerCell, SpaceDim);

      Plato::ScalarMultiVectorT<GradScalarType> strain("strain", numCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType> tgrad("tgrad", numCells, SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> stress("stress", numCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> flux ("flux",  numCells, SpaceDim);

      Plato::ScalarVectorT<StateScalarType> temperature("Gauss point temperature", numCells);

      auto quadratureWeight = mCubatureRule->getCubWeight();
      auto basisFunctions = mCubatureRule->getBasisFunctions();

      auto applyWeighting   = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        computeGradient(cellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight;

        // compute strain
        //
        kinematics(cellOrdinal, strain, tgrad, aState, gradient);

        // compute stress
        //
        interpolateFromNodal(cellOrdinal, basisFunctions, aState, temperature);
        kinetics(cellOrdinal, stress, flux, strain, tgrad, temperature);
      
        // apply weighting
        //
        applyWeighting(cellOrdinal, stress, aControl);

      },"Compute Stress");

      mNorm->evaluate(aResult, stress, aControl, cellVolume);

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
// class TMStressPNorm

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC(Plato::Elliptic::TMStressPNorm, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC(Plato::Elliptic::TMStressPNorm, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC(Plato::Elliptic::TMStressPNorm, Plato::SimplexThermomechanics, 3)
#endif

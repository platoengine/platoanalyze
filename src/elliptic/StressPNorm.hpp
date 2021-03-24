#ifndef STRESS_P_NORM_HPP
#define STRESS_P_NORM_HPP

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

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class StressPNorm : 
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;
    
    using Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMechanics<mSpaceDim>::mNumDofsPerCell;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Omega_h::Matrix< mNumVoigtTerms, mNumVoigtTerms> mCellStiffness;
    
    Plato::Scalar mQuadratureWeight;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim,mNumVoigtTerms,IndicatorFunctionType> mApplyWeighting;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;

  public:
    /**************************************************************************/
    StressPNorm(
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
      Plato::ElasticModelFactory<mSpaceDim> mmfactory(aProblemParams);
      auto materialModel = mmfactory.create(aSpatialDomain.getMaterialName());
      mCellStiffness = materialModel->getStiffnessMatrix();

//TODO quadrature
      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType d=2; d<=mSpaceDim; d++)
      { 
        mQuadratureWeight /= Plato::Scalar(d);
      }

      auto params = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);

      TensorNormFactory<mNumVoigtTerms, EvaluationType> normFactory;
      mNorm = normFactory.create(params);
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
      using SimplexPhysics = typename Plato::SimplexMechanics<mSpaceDim>;
      using StrainScalarType =
        typename Plato::fad_type_t<SimplexPhysics, StateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientWorkset<mSpaceDim> computeGradient;
      Plato::Strain<mSpaceDim>                 voigtStrain;
      Plato::LinearStress<EvaluationType,
                          SimplexPhysics>      voigtStress(mCellStiffness);

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight", tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        strain("strain", tNumCells, mNumVoigtTerms);

      Plato::ScalarArray3DT<ConfigScalarType>
        gradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        stress("stress", tNumCells, mNumVoigtTerms);

      auto quadratureWeight = mQuadratureWeight;
      auto applyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        computeGradient(cellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight;

        // compute strain
        //
        voigtStrain(cellOrdinal, strain, aState, gradient);

        // compute stress
        //
        voigtStress(cellOrdinal, stress, strain);
      
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
// class StressPNorm

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC(Plato::Elliptic::StressPNorm, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC(Plato::Elliptic::StressPNorm, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC(Plato::Elliptic::StressPNorm, Plato::SimplexMechanics, 3)
#endif

#endif

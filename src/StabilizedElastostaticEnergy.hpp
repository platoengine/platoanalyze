#ifndef STABILIZED_ELASTOSTATIC_ENERGY_HPP
#define STABILIZED_ELASTOSTATIC_ENERGY_HPP

#include "SimplexFadTypes.hpp"
#include "SimplexStabilizedMechanics.hpp"
#include "ScalarProduct.hpp"
#include "ProjectToNode.hpp"
#include "ApplyWeighting.hpp"
#include "Kinematics.hpp"
#include "Kinetics.hpp"
#include "ImplicitFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ElasticModelFactory.hpp"
#include "ToMap.hpp"
#include "ExpInstMacros.hpp"

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

/******************************************************************************//**
 * @brief Compute internal elastic energy criterion for stabilized form.
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class StabilizedElastostaticEnergy : 
  public Plato::SimplexStabilizedMechanics<EvaluationType::SpatialDim>,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    static constexpr Plato::OrdinalType mNMechDims  = mSpaceDim;
    static constexpr Plato::OrdinalType mNPressDims = 1;

    static constexpr Plato::OrdinalType mMDofOffset = 0;
    static constexpr Plato::OrdinalType mPressureDofOffset = mSpaceDim;
    
    using Plato::SimplexStabilizedMechanics<mSpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexStabilizedMechanics<mSpaceDim>::mNumDofsPerNode;
    using Plato::SimplexStabilizedMechanics<mSpaceDim>::mNumDofsPerCell;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = Plato::Elliptic::AbstractScalarFunction<EvaluationType>;
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<mSpaceDim>;
    
    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyTensorWeighting;
    Plato::ApplyWeighting<mSpaceDim, mSpaceDim,      IndicatorFunctionType> mApplyVectorWeighting;
    Plato::ApplyWeighting<mSpaceDim, 1,              IndicatorFunctionType> mApplyScalarWeighting;

    std::shared_ptr<CubatureType> mCubatureRule;

    Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> mMaterialModel;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    StabilizedElastostaticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap&          aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
        const std::string&             aFunctionName
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap, aFunctionName),
        mIndicatorFunction    (aPenaltyParams),
        mApplyTensorWeighting (mIndicatorFunction),
        mApplyVectorWeighting (mIndicatorFunction),
        mApplyScalarWeighting (mIndicatorFunction),
        mCubatureRule         (std::make_shared<CubatureType>())
    /**************************************************************************/
    {
      Plato::ElasticModelFactory<mSpaceDim> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());

      if( aProblemParams.isType<Teuchos::Array<std::string>>("Plottable") )
        mPlottable = aProblemParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
              Plato::ScalarVectorT      <ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType =
      typename Plato::fad_type_t<Plato::SimplexStabilizedMechanics
                <EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset <mSpaceDim> computeGradient;
      Plato::StabilizedKinematics   <mSpaceDim> kinematics;
      Plato::StabilizedKinetics     <mSpaceDim> kinetics(mMaterialModel);

      Plato::InterpolateFromNodal   <mSpaceDim, mSpaceDim, 0, mSpaceDim>     interpolatePGradFromNodal;
      Plato::InterpolateFromNodal   <mSpaceDim, mNumDofsPerNode, mPressureDofOffset> interpolatePressureFromNodal;

      Plato::ScalarProduct<mNumVoigtTerms> deviatorScalarProduct;
      
      Plato::ScalarVectorT      <ResultScalarType>    tVolStrain      ("volume strain",      tNumCells);
      Plato::ScalarVectorT      <ResultScalarType>    tPressure       ("GP pressure",        tNumCells);
      Plato::ScalarVectorT      <ConfigScalarType>    tCellVolume     ("cell weight",        tNumCells);
      Plato::ScalarMultiVectorT <NodeStateScalarType> tProjectedPGrad ("projected p grad",   tNumCells, mSpaceDim);
      Plato::ScalarMultiVectorT <ResultScalarType>    tCellStab       ("cell stabilization", tNumCells, mSpaceDim);
      Plato::ScalarMultiVectorT <GradScalarType>      tPGrad          ("pressure grad",      tNumCells, mSpaceDim);
      Plato::ScalarMultiVectorT <ResultScalarType>    tDevStress      ("deviatoric stress",  tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT <ResultScalarType>    tTotStress      ("cauchy stress",      tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT <GradScalarType>      tDGrad          ("displacement grad",  tNumCells, mNumVoigtTerms);
      Plato::ScalarArray3DT     <ConfigScalarType>    tGradient       ("gradient",           tNumCells, mNumNodesPerCell, mSpaceDim);

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto tBasisFunctions   = mCubatureRule->getBasisFunctions();

      auto& applyTensorWeighting = mApplyTensorWeighting;

      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(int cellOrdinal)
      {
        // compute gradient operator and cell volume
        //
        computeGradient(cellOrdinal, tGradient, aConfigWS, tCellVolume);
        tCellVolume(cellOrdinal) *= tQuadratureWeight;

        // compute symmetric gradient of displacement, pressure gradient, and temperature gradient
        //
        kinematics(cellOrdinal, tDGrad, tPGrad, aStateWS, tGradient);

        // interpolate projected PGrad, pressure, and temperature to gauss point
        //
        interpolatePressureFromNodal     ( cellOrdinal, tBasisFunctions, aStateWS, tPressure       );

        // compute the constitutive response
        //
        kinetics(cellOrdinal,     tCellVolume,
                 tProjectedPGrad, tPressure,
                 tDGrad,          tPGrad,
                 tDevStress,      tVolStrain,  tCellStab);

        for( int i=0; i<mSpaceDim; i++)
        {
            tTotStress(cellOrdinal,i) = tDevStress(cellOrdinal,i) + tPressure(cellOrdinal);
        }

        // apply weighting
        //
        applyTensorWeighting (cellOrdinal, tTotStress, aControlWS);

        // compute element internal energy (inner product of strain and weighted stress)
        //
        deviatorScalarProduct(cellOrdinal, aResultWS, tTotStress, tDGrad, tCellVolume);
      });
    }
};
// class InternalThermoelasticEnergy

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC(Plato::StabilizedElastostaticEnergy, Plato::SimplexStabilizedMechanics, 3)
#endif

#endif

#ifndef TRANSIENT_THERMOMECH_RESIDUAL_HPP
#define TRANSIENT_THERMOMECH_RESIDUAL_HPP

#include "SimplexThermomechanics.hpp"
#include "ApplyWeighting.hpp"
#include "ThermalContent.hpp"
#include "StressDivergence.hpp"
#include "FluxDivergence.hpp"
#include "parabolic/ParabolicSimplexFadTypes.hpp"
#include "TMKinematics.hpp"
#include "TMKinetics.hpp"
#include "PlatoMathHelpers.hpp"
#include "StateValues.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "InterpolateFromNodal.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"
#include "ToMap.hpp"

#include "ThermoelasticMaterial.hpp"
#include "ThermalMassMaterial.hpp"
#include "parabolic/AbstractVectorFunction.hpp"
#include "ImplicitFunctors.hpp"
#include "ProjectToNode.hpp"
#include "ApplyWeighting.hpp"
#include "NaturalBCs.hpp"

#include "parabolic/ExpInstMacros.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TransientThermomechResidual : 
  public Plato::SimplexThermomechanics<EvaluationType::SpatialDim>,
  public Plato::Parabolic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    static constexpr int NThrmDims = 1;
    static constexpr int NMechDims = SpaceDim;

    static constexpr int TDofOffset = SpaceDim;
    static constexpr int MDofOffset = 0;

    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerNode;

    using Plato::Parabolic::AbstractVectorFunction<EvaluationType>::mSpatialDomain;
    using Plato::Parabolic::AbstractVectorFunction<EvaluationType>::mDataMap;

    using StateScalarType    = typename EvaluationType::StateScalarType;
    using StateDotScalarType = typename EvaluationType::StateDotScalarType;
    using ControlScalarType  = typename EvaluationType::ControlScalarType;
    using ConfigScalarType   = typename EvaluationType::ConfigScalarType;
    using ResultScalarType   = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = Plato::Parabolic::AbstractVectorFunction<EvaluationType>;
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<SpaceDim>;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<SpaceDim, SpaceDim,       IndicatorFunctionType> mApplyFluxWeighting;
    Plato::ApplyWeighting<SpaceDim, NThrmDims,      IndicatorFunctionType> mApplyMassWeighting;

    std::shared_ptr<CubatureType> mCubatureRule;

    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NMechDims, mNumDofsPerNode, MDofOffset>> mBoundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NThrmDims, mNumDofsPerNode, TDofOffset>> mBoundaryFluxes;

    Teuchos::RCP<Plato::MaterialModel<SpaceDim>> mMaterialModel;
    Teuchos::RCP<Plato::MaterialModel<SpaceDim>> mThermalMassMaterialModel;

    std::vector<std::string> mPlotTable;

  public:
    /**************************************************************************/
    TransientThermomechResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
     ) :
         FunctionBaseType      (aSpatialDomain, aDataMap, {"Displacement X", "Displacement Y", "Displacement Z", "Temperature"}),
         mIndicatorFunction    (aPenaltyParams),
         mApplyStressWeighting (mIndicatorFunction),
         mApplyFluxWeighting   (mIndicatorFunction),
         mApplyMassWeighting   (mIndicatorFunction),
         mCubatureRule         (std::make_shared<CubatureType>()),
         mBoundaryLoads        (nullptr),
         mBoundaryFluxes       (nullptr)
    /**************************************************************************/
    {
        {
            Plato::ThermoelasticModelFactory<SpaceDim> mmfactory(aProblemParams);
            mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
        }

        {
            Plato::ThermalMassModelFactory<SpaceDim> mmfactory(aProblemParams);
            mThermalMassMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
        }

        // parse boundary Conditions
        // 
        if(aProblemParams.isSublist("Mechanical Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim, NMechDims, mNumDofsPerNode, MDofOffset>>
                (aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
        }
        // parse thermal boundary Conditions
        // 
        if(aProblemParams.isSublist("Thermal Natural Boundary Conditions"))
        {
            mBoundaryFluxes = std::make_shared<Plato::NaturalBCs<SpaceDim, NThrmDims, mNumDofsPerNode, TDofOffset>>
                (aProblemParams.sublist("Thermal Natural Boundary Conditions"));
        }

        auto tResidualParams = aProblemParams.sublist("Parabolic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlotTable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }


    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT< StateScalarType    > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType > & aStateDot,
        const Plato::ScalarMultiVectorT< ControlScalarType  > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType   > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType   > & aResult,
              Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermomechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT< GradScalarType     > tStrain          ("strain",         tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT< GradScalarType     > tGrad            ("T gradient",     tNumCells, SpaceDim);
      Plato::ScalarMultiVectorT< ResultScalarType   > tStress          ("stress",         tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT< ResultScalarType   > tFlux            ("thermal flux",   tNumCells, SpaceDim);
      Plato::ScalarArray3DT    < ConfigScalarType   > tGradient        ("gradient",       tNumCells, mNumNodesPerCell,SpaceDim);
      Plato::ScalarVectorT     < ResultScalarType   > tHeatRate        ("GP heat rate",   tNumCells);
      Plato::ScalarVectorT     < StateScalarType    > tTemperature     ("GP point T",     tNumCells);
      Plato::ScalarVectorT     < StateDotScalarType > tTemperatureRate ("GP point T dot", tNumCells);

      // create a bunch of functors:
      Plato::ComputeGradientWorkset<SpaceDim> tComputeGradient;

      Plato::TMKinematics<SpaceDim> tKinematics;
      Plato::TMKinetics<SpaceDim>   tKinetics(mMaterialModel);

      Plato::InterpolateFromNodal<SpaceDim, mNumDofsPerNode, TDofOffset> tInterpolateFromNodal;

      Plato::FluxDivergence  <SpaceDim, mNumDofsPerNode, TDofOffset> tFluxDivergence;
      Plato::StressDivergence<SpaceDim, mNumDofsPerNode, MDofOffset> tStressDivergence;

      Plato::ThermalContent<SpaceDim> tComputeHeatRate(mThermalMassMaterialModel);
      Plato::ProjectToNode<SpaceDim, mNumDofsPerNode, TDofOffset> tProjectHeatRate;
      
      auto tBasisFunctions = mCubatureRule->getBasisFunctions();
    
      auto& tApplyStressWeighting = mApplyStressWeighting;
      auto& tApplyFluxWeighting   = mApplyFluxWeighting;
      auto& tApplyMassWeighting   = mApplyMassWeighting;
      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
      {
    
        tComputeGradient(tCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(tCellOrdinal) *= tQuadratureWeight;
    
        // compute strain and temperature gradient
        //
        tKinematics(tCellOrdinal, tStrain, tGrad, aState, tGradient);

        // compute stress and thermal flux
        //
        tInterpolateFromNodal(tCellOrdinal, tBasisFunctions, aState, tTemperature);

        tKinetics(tCellOrdinal, tStress, tFlux, tStrain, tGrad, tTemperature);

        // apply weighting
        //
        tApplyStressWeighting(tCellOrdinal, tStress, aControl);

        tApplyFluxWeighting(tCellOrdinal, tFlux, aControl);

        // compute stress and flux divergence
        //
        tStressDivergence(tCellOrdinal, aResult, tStress, tGradient, tCellVolume);

        tFluxDivergence(tCellOrdinal, aResult, tFlux, tGradient, tCellVolume);

        // add capacitance terms

        // compute temperature at gausspoints
        //
        tInterpolateFromNodal(tCellOrdinal, tBasisFunctions, aStateDot, tTemperatureRate);

        // compute the time rate of internal thermal energy
        //
        tComputeHeatRate(tCellOrdinal, tHeatRate, tTemperatureRate, tTemperature);

        // apply weighting
        //
        tApplyMassWeighting(tCellOrdinal, tHeatRate, aControl);

        // project to nodes
        //
        tProjectHeatRate(tCellOrdinal, tCellVolume, tBasisFunctions, tHeatRate, aResult);

      },"stress and flux divergence");

      if(std::count(mPlotTable.begin(), mPlotTable.end(), "strain")) { Plato::toMap(mDataMap, tStrain, "strain"); }
      if(std::count(mPlotTable.begin(), mPlotTable.end(), "stress")) { Plato::toMap(mDataMap, tStress, "stress"); }
    }

    /**************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                             & aSpatialModel,
        const Plato::ScalarMultiVectorT< StateScalarType    > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType > & aStateDot,
        const Plato::ScalarMultiVectorT< ControlScalarType  > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType   > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType   > & aResult,
              Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0);
        }
        if( mBoundaryFluxes != nullptr )
        {
            mBoundaryFluxes->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0);
        }
    }
};

} // namespace Parabolic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_PARABOLIC_EXPL_DEC(Plato::Parabolic::TransientThermomechResidual, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_PARABOLIC_EXPL_DEC(Plato::Parabolic::TransientThermomechResidual, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_PARABOLIC_EXPL_DEC(Plato::Parabolic::TransientThermomechResidual, Plato::SimplexThermomechanics, 3)
#endif

#endif

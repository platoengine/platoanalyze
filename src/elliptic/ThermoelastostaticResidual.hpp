#ifndef THERMOELASTOSTATIC_RESIDUAL_HPP
#define THERMOELASTOSTATIC_RESIDUAL_HPP

#include <memory>

#include "PlatoTypes.hpp"
#include "SimplexFadTypes.hpp"
#include "SimplexThermomechanics.hpp"
#include "TMKinematics.hpp"
#include "TMKinetics.hpp"
#include "StressDivergence.hpp"
#include "FluxDivergence.hpp"
#include "elliptic/AbstractVectorFunction.hpp"
#include "ApplyWeighting.hpp"
#include "CellForcing.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ThermoelasticMaterial.hpp"
#include "NaturalBCs.hpp"
#include "BodyLoads.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ThermoelastostaticResidual :
        public Plato::SimplexThermomechanics<EvaluationType::SpatialDim>,
        public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;

    static constexpr int NThrmDims = 1;
    static constexpr int NMechDims = SpaceDim;

    static constexpr int TDofOffset = SpaceDim;
    static constexpr int MDofOffset = 0;

    using Plato::SimplexThermomechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerNode;
    using Plato::SimplexThermomechanics<SpaceDim>::mNumDofsPerCell;

    using Plato::Elliptic::AbstractVectorFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractVectorFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim, SpaceDim,       IndicatorFunctionType> mApplyFluxWeighting;
    Plato::ApplyWeighting<SpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType>> mBodyLoads;

    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NMechDims, mNumDofsPerNode, MDofOffset>> mBoundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NThrmDims, mNumDofsPerNode, TDofOffset>> mBoundaryFluxes;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    Teuchos::RCP<Plato::MaterialModel<SpaceDim>> mMaterialModel;

    std::vector<std::string> mPlottable;

public:
    /**************************************************************************/
    ThermoelastostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        Plato::Elliptic::AbstractVectorFunction<EvaluationType>(aSpatialDomain, aDataMap),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyFluxWeighting   (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mBoundaryLoads(nullptr),
        mBoundaryFluxes(nullptr),
        mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
        // create material model and get stiffness
        //
        Plato::ThermoelasticModelFactory<SpaceDim> mmfactory(aProblemParams);
        mMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());

        // parse body loads
        // 
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType>>(aProblemParams.sublist("Body Loads"));
        }
  
        // parse mechanical boundary Conditions
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
  
        auto tResidualParams = aProblemParams.sublist("Elliptic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }

    /**************************************************************************/
    void evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>   & state,
        const Plato::ScalarMultiVectorT <ControlScalarType> & control,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & config,
              Plato::ScalarMultiVectorT <ResultScalarType>  & result,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = state.extent(0);

      using GradScalarType =
      typename Plato::fad_type_t<Plato::SimplexThermomechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> computeGradient;
      Plato::TMKinematics<SpaceDim>           kinematics;
      Plato::TMKinetics<SpaceDim>             kinetics(mMaterialModel);
      
      Plato::StressDivergence<SpaceDim, mNumDofsPerNode, MDofOffset> stressDivergence;
      Plato::FluxDivergence  <SpaceDim, mNumDofsPerNode, TDofOffset> fluxDivergence;

      Plato::InterpolateFromNodal<SpaceDim, mNumDofsPerNode, TDofOffset> interpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType> cellVolume("cell weight",tNumCells);

      Plato::ScalarArray3DT<ConfigScalarType> gradient("gradient", tNumCells, mNumNodesPerCell, SpaceDim);

      Plato::ScalarMultiVectorT<GradScalarType> strain("strain", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType> tgrad("tgrad", tNumCells, SpaceDim);
    
      Plato::ScalarMultiVectorT<ResultScalarType> stress("stress", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> flux ("flux" , tNumCells, SpaceDim);
    
      Plato::ScalarVectorT<StateScalarType> temperature("Gauss point temperature", tNumCells);

      auto quadratureWeight = mCubatureRule->getCubWeight();
      auto basisFunctions = mCubatureRule->getBasisFunctions();

      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(int cellOrdinal)
      {
        computeGradient(cellOrdinal, gradient, config, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight;

        // compute strain and electric field
        //
        kinematics(cellOrdinal, strain, tgrad, state, gradient);
    
        // compute stress and electric displacement
        //
        interpolateFromNodal(cellOrdinal, basisFunctions, state, temperature);
        kinetics(cellOrdinal, stress, flux, strain, tgrad, temperature);

      }, "Cauchy stress");

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyFluxWeighting  = mApplyFluxWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(int cellOrdinal)
      {
        // apply weighting
        //
        applyStressWeighting(cellOrdinal, stress, control);
        applyFluxWeighting (cellOrdinal, flux,  control);
    
        // compute divergence
        //
        stressDivergence(cellOrdinal, result, stress, gradient, cellVolume);
        fluxDivergence (cellOrdinal, result, flux,  gradient, cellVolume);
      }, "Apply weighting and compute divergence");

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, state, control, result, -1.0 );
      }

// TODO      if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, strain, "strain");
// TODO      if( std::count(mPlottable.begin(),mPlottable.end(),"tgrad") ) toMap(mDataMap, tgrad, "tgrad");
// TODO      if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, stress, "stress");
// TODO      if( std::count(mPlottable.begin(),mPlottable.end(),"flux" ) ) toMap(mDataMap, flux, "flux" );

    }
    /**************************************************************************/
    void evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      if( mBoundaryLoads != nullptr )
      {
          mBoundaryLoads->get( aSpatialModel, aState, aControl, aConfig, aResult, -1.0 );
      }

      if( mBoundaryFluxes != nullptr )
      {
          mBoundaryFluxes->get( aSpatialModel, aState, aControl, aConfig, aResult, -1.0 );
      }
    }
};
// class ThermoelastostaticResidual

} // namespace Elliptic

} // namespace Plato
#endif

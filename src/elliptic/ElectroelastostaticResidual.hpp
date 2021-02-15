#ifndef ELECTROELASTOSTATIC_RESIDUAL_HPP
#define ELECTROELASTOSTATIC_RESIDUAL_HPP

#include <memory>

#include "PlatoTypes.hpp"
#include "SimplexFadTypes.hpp"
#include "SimplexElectromechanics.hpp"
#include "EMKinematics.hpp"
#include "EMKinetics.hpp"
#include "StressDivergence.hpp"
#include "FluxDivergence.hpp"

#include "elliptic/AbstractVectorFunction.hpp"

#include "ApplyWeighting.hpp"
#include "CellForcing.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "LinearElectroelasticMaterial.hpp"
#include "NaturalBCs.hpp"
#include "BodyLoads.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ElectroelastostaticResidual :
        public Plato::SimplexElectromechanics<EvaluationType::SpatialDim>,
        public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    static constexpr Plato::OrdinalType NElecDims = 1;
    static constexpr Plato::OrdinalType NMechDims = SpaceDim;

    static constexpr Plato::OrdinalType EDofOffset = SpaceDim;
    static constexpr Plato::OrdinalType MDofOffset = 0;

    using PhysicsType = typename Plato::SimplexElectromechanics<SpaceDim>;

    using Plato::SimplexElectromechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using PhysicsType::mNumDofsPerNode;
    using PhysicsType::mNumDofsPerCell;

    using Plato::Elliptic::AbstractVectorFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractVectorFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    ApplyWeighting<SpaceDim, SpaceDim,       IndicatorFunctionType> mApplyEDispWeighting;
    ApplyWeighting<SpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, PhysicsType>> mBodyLoads;

    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NMechDims, mNumDofsPerNode, MDofOffset>> mBoundaryLoads;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim, NElecDims, mNumDofsPerNode, EDofOffset>> mBoundaryCharges;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    Teuchos::RCP<Plato::LinearElectroelasticMaterial<SpaceDim>> mMaterialModel;

    std::vector<std::string> mPlottable;

public:
    /**************************************************************************/
    ElectroelastostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        Plato::Elliptic::AbstractVectorFunction<EvaluationType>(aSpatialDomain, aDataMap),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyEDispWeighting  (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mBoundaryLoads        (nullptr),
        mBoundaryCharges      (nullptr),
        mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
        // create material model and get stiffness
        //
        Plato::ElectroelasticModelFactory<SpaceDim> mmfactory(aProblemParams);
        mMaterialModel = mmfactory.create(mSpatialDomain.getMaterialName());

        // parse body loads
        // 
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, PhysicsType>>(aProblemParams.sublist("Body Loads"));
        }
  
        // parse mechanical boundary Conditions
        // 
        if(aProblemParams.isSublist("Mechanical Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim, NMechDims, mNumDofsPerNode, MDofOffset>>
                (aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
        }
  
        // parse electrical boundary Conditions
        // 
        if(aProblemParams.isSublist("Electrical Natural Boundary Conditions"))
        {
            mBoundaryCharges = std::make_shared<Plato::NaturalBCs<SpaceDim, NElecDims, mNumDofsPerNode, EDofOffset>>
                (aProblemParams.sublist("Electrical Natural Boundary Conditions"));
        }
  
        auto tResidualParams = aProblemParams.sublist("Electroelastostatics");
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
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType =
      typename Plato::fad_type_t<Plato::SimplexElectromechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> computeGradient;
      Plato::EMKinematics<SpaceDim>                  kinematics;
      Plato::EMKinetics<SpaceDim>                    kinetics(mMaterialModel);
      
      Plato::StressDivergence<SpaceDim, mNumDofsPerNode, MDofOffset> stressDivergence;
      Plato::FluxDivergence  <SpaceDim, mNumDofsPerNode, EDofOffset> edispDivergence;

      Plato::ScalarVectorT<ConfigScalarType> cellVolume("cell weight",tNumCells);

      Plato::ScalarArray3DT<ConfigScalarType> gradient("gradient", tNumCells, mNumNodesPerCell, SpaceDim);

      Plato::ScalarMultiVectorT<GradScalarType> strain("strain", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType> efield("efield", tNumCells, SpaceDim);
    
      Plato::ScalarMultiVectorT<ResultScalarType> stress("stress", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> edisp ("edisp" , tNumCells, SpaceDim);
    
      auto quadratureWeight = mCubatureRule->getCubWeight();
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        computeGradient(cellOrdinal, gradient, config, cellVolume);
        cellVolume(cellOrdinal) *= quadratureWeight;

        // compute strain and electric field
        //
        kinematics(cellOrdinal, strain, efield, state, gradient);
    
        // compute stress and electric displacement
        //
        kinetics(cellOrdinal, stress, edisp, strain, efield);

      }, "Cauchy stress");

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyEDispWeighting  = mApplyEDispWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType cellOrdinal)
      {
        // apply weighting
        //
        applyStressWeighting(cellOrdinal, stress, control);
        applyEDispWeighting (cellOrdinal, edisp,  control);
    
        // compute divergence
        //
        stressDivergence(cellOrdinal, result, stress, gradient, cellVolume);
        edispDivergence (cellOrdinal, result, edisp,  gradient, cellVolume);
      }, "Apply weighting and compute divergence");

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, state, control, result, -1.0 );
      }

     if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, strain, "strain", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"efield") ) toMap(mDataMap, strain, "efield", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, stress, "stress", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"edisp" ) ) toMap(mDataMap, stress, "edisp" , mSpatialDomain);

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
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0 );
        }

        if( mBoundaryCharges != nullptr )
        {
            mBoundaryCharges->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0 );
        }
    }
};
// class ElectroelastostaticResidual

} // namespace Elliptic

} // namespace Plato
#endif

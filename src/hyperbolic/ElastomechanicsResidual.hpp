#ifndef HYPERBOLIC_ELASTOMECHANICS_RESIDUAL_HPP
#define HYPERBOLIC_ELASTOMECHANICS_RESIDUAL_HPP
#include "Simp.hpp"
#include "Ramp.hpp"
#include "ToMap.hpp"
#include "Strain.hpp"
#include "Heaviside.hpp"
#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "Plato_Solve.hpp"
#include "LinearStress.hpp"
#include "ProjectToNode.hpp"
#include "ApplyWeighting.hpp"
#include "StressDivergence.hpp"
#include "SimplexMechanics.hpp"
#include "ApplyConstraints.hpp"
#include "PlatoMathHelpers.hpp"
#include "ElasticModelFactory.hpp"
#include "InterpolateFromNodal.hpp"
#include "PlatoAbstractProblem.hpp"
#include "ElastomechanicsResidual.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "hyperbolic/Newmark.hpp"
#include "hyperbolic/InertialContent.hpp"
#include "hyperbolic/HyperbolicVectorFunction.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TransientMechanicsResidual :
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexMechanics<SpaceDim>::mNumVoigtTerms;
    using Plato::SimplexMechanics<SpaceDim>::mNumDofsPerCell;
    using Plato::SimplexMechanics<SpaceDim>::mNumDofsPerNode;

    using Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>::mSpatialDomain;
    using Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>::mDataMap;

    using StateScalarType       = typename EvaluationType::StateScalarType;
    using StateDotScalarType    = typename EvaluationType::StateDotScalarType;
    using StateDotDotScalarType = typename EvaluationType::StateDotDotScalarType;
    using ControlScalarType     = typename EvaluationType::ControlScalarType;
    using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
    using ResultScalarType      = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>;
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<SpaceDim>;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<SpaceDim, SpaceDim,       IndicatorFunctionType> mApplyMassWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType>> mBodyLoads;
    std::shared_ptr<CubatureType> mCubatureRule;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>> mBoundaryLoads;

    bool mRayleighDamping;

    Teuchos::RCP<Plato::LinearElasticMaterial<SpaceDim>> mMaterialModel;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    TransientMechanicsResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap,
                               {"Displacement X", "Displacement Y", "Displacement Z"},
                               {"Velocity X",     "Velocity Y",     "Velocity Z"    }),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyMassWeighting   (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mCubatureRule         (std::make_shared<CubatureType>()),
        mBoundaryLoads        (nullptr)
    /**************************************************************************/
    {

        // create material model and get stiffness
        //
        Plato::ElasticModelFactory<SpaceDim> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        mRayleighDamping = (mMaterialModel->getRayleighA() != 0.0)
                        || (mMaterialModel->getRayleighB() != 0.0);

        // parse body loads
        //
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType>>
                         (aProblemParams.sublist("Body Loads"));
        }

        // parse boundary Conditions
        //
        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>>
                             (aProblemParams.sublist("Natural Boundary Conditions"));
        }

        auto tResidualParams = aProblemParams.sublist("Hyperbolic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0,
              Plato::Scalar aCurrentTime = 0.0
    ) const override
    /**************************************************************************/
    {
        if ( mRayleighDamping )
        {
             evaluateWithDamping(aState, aStateDot, aStateDotDot, aControl, aConfig, aResult, aTimeStep, aCurrentTime);
        }
        else
        {
             evaluateWithoutDamping(aState, aStateDot, aStateDotDot, aControl, aConfig, aResult, aTimeStep, aCurrentTime);
        }
    }

    /**************************************************************************/
    void
    evaluateWithoutDamping(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0,
              Plato::Scalar aCurrentTime = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using StrainScalarType =
          typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> tComputeGradient;
      Plato::Strain<SpaceDim>                 tComputeVoigtStrain;
      Plato::LinearStress<SpaceDim>           tComputeVoigtStress(mMaterialModel);
      Plato::StressDivergence<SpaceDim>       tComputeStressDivergence;
      Plato::InertialContent<SpaceDim>        tInertialContent(mMaterialModel);

      Plato::ProjectToNode<SpaceDim, mNumDofsPerNode>        tProjectInertialContent;

      Plato::InterpolateFromNodal<SpaceDim, mNumDofsPerNode, /*offset=*/0, SpaceDim> tInterpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        tStrain("strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient",tNumCells,mNumNodesPerCell,SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tStress("stress",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<StateDotDotScalarType>
        tAccelerationGP("acceleration at Gauss point", tNumCells, SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tInertialContentGP("Inertial content at Gauss point", tNumCells, SpaceDim);

      auto tBasisFunctions = mCubatureRule->getBasisFunctions();

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto& tApplyStressWeighting = mApplyStressWeighting;
      auto& tApplyMassWeighting = mApplyMassWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute strain
        tComputeVoigtStrain(aCellOrdinal, tStrain, aState, tGradient);

        // compute stress
        tComputeVoigtStress(aCellOrdinal, tStress, tStrain);

        // apply weighting
        tApplyStressWeighting(aCellOrdinal, tStress, aControl);

        // compute stress divergence
        tComputeStressDivergence(aCellOrdinal, aResult, tStress, tGradient, tCellVolume);

        // compute accelerations at gausspoints
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aStateDotDot, tAccelerationGP);

        // compute inertia at gausspoints
        tInertialContent(aCellOrdinal, tInertialContentGP, tAccelerationGP);

        // apply weighting
        tApplyMassWeighting(aCellOrdinal, tInertialContentGP, aControl);

        // project to nodes
        tProjectInertialContent(aCellOrdinal, tCellVolume, tBasisFunctions, tInertialContentGP, aResult);

      }, "Compute Residual");

     if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, tStress, "stress", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, tStress, "strain", mSpatialDomain);

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aState, aControl, aResult, -1.0 );
      }
    }
    /**************************************************************************/
    void
    evaluateWithDamping(
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0,
              Plato::Scalar aCurrentTime = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using StrainScalarType =
          typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      using VelGradScalarType =
          typename Plato::fad_type_t<Plato::SimplexMechanics<EvaluationType::SpatialDim>, StateDotScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<SpaceDim> tComputeGradient;
      Plato::Strain<SpaceDim>                 tComputeVoigtStrain;
      Plato::LinearStress<SpaceDim>           tComputeVoigtStress(mMaterialModel);
      Plato::StressDivergence<SpaceDim>       tComputeStressDivergence;
      Plato::InertialContent<SpaceDim>        tInertialContent(mMaterialModel);

      Plato::ProjectToNode<SpaceDim, mNumDofsPerNode>        tProjectInertialContent;

      Plato::InterpolateFromNodal<SpaceDim, mNumDofsPerNode, /*offset=*/0, SpaceDim> tInterpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        tStrain("strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<VelGradScalarType>
        tVelGrad("velocity gradient", tNumCells, mNumVoigtTerms);

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient", tNumCells, mNumNodesPerCell, SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tStress("stress", tNumCells, mNumVoigtTerms);

      Plato::ScalarMultiVectorT<StateDotDotScalarType>
        tAccelerationGP("acceleration at Gauss point", tNumCells, SpaceDim);

      Plato::ScalarMultiVectorT<StateDotScalarType>
        tVelocityGP("velocity at Gauss point", tNumCells, SpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tInertialContentGP("Inertial content at Gauss point", tNumCells, SpaceDim);

      auto tBasisFunctions = mCubatureRule->getBasisFunctions();

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto& tApplyStressWeighting = mApplyStressWeighting;
      auto& tApplyMassWeighting = mApplyMassWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute strain
        tComputeVoigtStrain(aCellOrdinal, tStrain, aState, tGradient);

        // compute velocity gradient
        tComputeVoigtStrain(aCellOrdinal, tVelGrad, aStateDot, tGradient);

        // compute stress
        tComputeVoigtStress(aCellOrdinal, tStress, tStrain, tVelGrad);

        // apply weighting
        tApplyStressWeighting(aCellOrdinal, tStress, aControl);

        // compute stress divergence
        tComputeStressDivergence(aCellOrdinal, aResult, tStress, tGradient, tCellVolume);

        // compute accelerations at gausspoints
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aStateDotDot, tAccelerationGP);

        // compute velocities at gausspoints
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aStateDot, tVelocityGP);

        // compute inertia at gausspoints
        tInertialContent(aCellOrdinal, tInertialContentGP, tVelocityGP, tAccelerationGP);

        // apply weighting
        tApplyMassWeighting(aCellOrdinal, tInertialContentGP, aControl);

        // project to nodes
        tProjectInertialContent(aCellOrdinal, tCellVolume, tBasisFunctions, tInertialContentGP, aResult);

      }, "Compute Residual");

     if( std::count(mPlottable.begin(),mPlottable.end(),"stress") ) toMap(mDataMap, tStress, "stress", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"strain") ) toMap(mDataMap, tStress, "strain", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"velgrad") ) toMap(mDataMap, tStress, "velgrad", mSpatialDomain);

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aState, aControl, aResult, -1.0 );
      }
    }
    /**************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                                & aSpatialModel,
        const Plato::ScalarMultiVectorT< StateScalarType       > & aState,
        const Plato::ScalarMultiVectorT< StateDotScalarType    > & aStateDot,
        const Plato::ScalarMultiVectorT< StateDotDotScalarType > & aStateDotDot,
        const Plato::ScalarMultiVectorT< ControlScalarType     > & aControl,
        const Plato::ScalarArray3DT    < ConfigScalarType      > & aConfig,
              Plato::ScalarMultiVectorT< ResultScalarType      > & aResult,
              Plato::Scalar aTimeStep = 0.0,
              Plato::Scalar aCurrentTime = 0.0
    ) const override
    /**************************************************************************/
    {
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult, -1.0, aCurrentTime );
        }
    }
};

#endif

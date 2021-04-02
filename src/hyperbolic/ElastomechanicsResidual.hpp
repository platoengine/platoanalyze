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
#include "hyperbolic/HyperbolicLinearStress.hpp"

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TransientMechanicsResidual :
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    using PhysicsType = typename Plato::SimplexMechanics<mSpaceDim>;

    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using PhysicsType::mNumVoigtTerms;
    using PhysicsType::mNumDofsPerCell;
    using PhysicsType::mNumDofsPerNode;

    using Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>::mSpatialDomain;
    using Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>::mDataMap;

    using StateScalarType       = typename EvaluationType::StateScalarType;
    using StateDotScalarType    = typename EvaluationType::StateDotScalarType;
    using StateDotDotScalarType = typename EvaluationType::StateDotDotScalarType;
    using ControlScalarType     = typename EvaluationType::ControlScalarType;
    using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
    using ResultScalarType      = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = Plato::Hyperbolic::AbstractVectorFunction<EvaluationType>;
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<mSpaceDim>;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<mSpaceDim, mSpaceDim,       IndicatorFunctionType> mApplyMassWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, PhysicsType>> mBodyLoads;
    std::shared_ptr<CubatureType> mCubatureRule;
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim,mNumDofsPerNode>> mBoundaryLoads;

    bool mRayleighDamping;

    Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> mMaterialModel;

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
        Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        mRayleighDamping = (mMaterialModel->getRayleighA() != 0.0)
                        || (mMaterialModel->getRayleighB() != 0.0);

        // parse body loads
        //
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, PhysicsType>>
                         (aProblemParams.sublist("Body Loads"));
        }

        // parse boundary Conditions
        //
        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<mSpaceDim,mNumDofsPerNode>>
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
      using SimplexPhysics = typename Plato::SimplexMechanics<mSpaceDim>;
      using StrainScalarType =
          typename Plato::fad_type_t<SimplexPhysics, StateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
      Plato::Strain<mSpaceDim>                 tComputeVoigtStrain;
      // Note one can use the HyperbolicLinearStress to call the
      // original LinearStress operator (sans VelGrad) or the operator
      // with VelGrad as below in evaluateWithDamping. That is the
      // HyperbolicLinearStress contains both operator interfaces.
      Plato::LinearStress<EvaluationType,
                          SimplexPhysics>      tComputeVoigtStress(mMaterialModel);
      // OR
      // Plato::Hyperbolic::HyperbolicLinearStress
      //          <EvaluationType, SimplexPhysics>     tComputeVoigtStress(mMaterialModel);
      Plato::StressDivergence<mSpaceDim>       tComputeStressDivergence;
      Plato::InertialContent<mSpaceDim>        tInertialContent(mMaterialModel);

      Plato::ProjectToNode<mSpaceDim, mNumDofsPerNode>        tProjectInertialContent;

      Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode, /*offset=*/0, mSpaceDim> tInterpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        tStrain("strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient",tNumCells,mNumNodesPerCell,mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tStress("stress",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<StateDotDotScalarType>
        tAccelerationGP("acceleration at Gauss point", tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tInertialContentGP("Inertial content at Gauss point", tNumCells, mSpaceDim);

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
      using SimplexPhysics = typename Plato::SimplexMechanics<mSpaceDim>;
      using StrainScalarType =
          typename Plato::fad_type_t<SimplexPhysics, StateScalarType, ConfigScalarType>;
      using VelGradScalarType =
          typename Plato::fad_type_t<SimplexPhysics, StateDotScalarType, ConfigScalarType>;
      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
      Plato::Strain<mSpaceDim>                 tComputeVoigtStrain;
      Plato::Hyperbolic::HyperbolicLinearStress
          <EvaluationType, SimplexPhysics>     tComputeVoigtStress(mMaterialModel);
      Plato::StressDivergence<mSpaceDim>       tComputeStressDivergence;
      Plato::InertialContent<mSpaceDim>        tInertialContent(mMaterialModel);

      Plato::ProjectToNode<mSpaceDim, mNumDofsPerNode>        tProjectInertialContent;

      Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode, /*offset=*/0, mSpaceDim> tInterpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        tStrain("strain",tNumCells,mNumVoigtTerms);

      Plato::ScalarMultiVectorT<VelGradScalarType>
        tVelGrad("velocity gradient", tNumCells, mNumVoigtTerms);

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tStress("stress", tNumCells, mNumVoigtTerms);

      Plato::ScalarMultiVectorT<StateDotDotScalarType>
        tAccelerationGP("acceleration at Gauss point", tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<StateDotScalarType>
        tVelocityGP("velocity at Gauss point", tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType>
        tInertialContentGP("Inertial content at Gauss point", tNumCells, mSpaceDim);

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

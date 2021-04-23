#ifndef STABILIZED_ELASTOSTATIC_RESIDUAL_HPP
#define STABILIZED_ELASTOSTATIC_RESIDUAL_HPP

#include "PlatoUtilities.hpp"

#include <memory>

#include "Kinetics.hpp"
#include "PlatoTypes.hpp"
#include "Kinematics.hpp"
#include "FluxDivergence.hpp"
#include "SimplexFadTypes.hpp"
#include "StressDivergence.hpp"
#include "PressureDivergence.hpp"
#include "ProjectToNode.hpp"
#include "Projection.hpp"
#include "AbstractVectorFunctionVMS.hpp"
#include "ApplyWeighting.hpp"
#include "CellForcing.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ElasticModelFactory.hpp"
#include "NaturalBCs.hpp"
#include "BodyLoads.hpp"
#include "SimplexStabilizedMechanics.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Stabilized elastostatic residual (reference: M. Chiumenti et al. (2004))
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class StabilizedElastostaticResidual :
        public Plato::SimplexStabilizedMechanics<EvaluationType::SpatialDim>,
        public Plato::AbstractVectorFunctionVMS<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    static constexpr Plato::OrdinalType mNumMechDims  = mSpaceDim; /*!< number of mechanical degrees of freedom */
    static constexpr Plato::OrdinalType mNumPressDims = 1; /*!< number of pressure degrees of freedom */
    static constexpr Plato::OrdinalType mMechDofOffset = 0; /*!< mechanical degrees of freedom offset */
    static constexpr Plato::OrdinalType mPressDofOffset = mSpaceDim; /*!< pressure degree of freedom offset */

    using PhysicsType = typename Plato::SimplexStabilizedMechanics<mSpaceDim>;

    using Plato::SimplexStabilizedMechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of Voigt terms */
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */
    using PhysicsType::mNumDofsPerNode; /*!< number of nodes per node */
    using PhysicsType::mNumDofsPerCell; /*!< number of nodes per cell */

    using Plato::AbstractVectorFunctionVMS<EvaluationType>::mSpatialDomain; /*!< mesh metadata */
    using Plato::AbstractVectorFunctionVMS<EvaluationType>::mDataMap; /*!< output data map */

    using StateScalarType     = typename EvaluationType::StateScalarType; /*!< State Automatic Differentiation (AD) type */
    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType; /*!< Node State AD type */
    using ControlScalarType   = typename EvaluationType::ControlScalarType; /*!< Control AD type */
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType; /*!< Configuration AD type */
    using ResultScalarType    = typename EvaluationType::ResultScalarType; /*!< Result AD type */

    using FunctionBaseType = Plato::AbstractVectorFunctionVMS<EvaluationType>;
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>;

    IndicatorFunctionType mIndicatorFunction; /*!< material penalty function */
    Plato::ApplyWeighting<mSpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyTensorWeighting; /*!< apply penalty to tensor function */
    Plato::ApplyWeighting<mSpaceDim, mSpaceDim,      IndicatorFunctionType> mApplyVectorWeighting; /*!< apply penalty to vector function */
    Plato::ApplyWeighting<mSpaceDim, 1 /* number of pressure dofs per node */, IndicatorFunctionType> mApplyScalarWeighting; /*!< apply penalty to scalar function */

    std::shared_ptr<Plato::BodyLoads<EvaluationType, PhysicsType>> mBodyLoads; /*!< body loads interface */
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim, mNumMechDims, mNumDofsPerNode, mMechDofOffset>> mBoundaryLoads; /*!< boundary loads interface */

    std::shared_ptr<CubatureType> mCubatureRule; /*!< cubature/integration rule interface */
    Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> mMaterialModel; /*!< material model interface */

    std::vector<std::string> mPlotTable; /*!< array with output data identifiers */

private:
    /******************************************************************************//**
     * \brief initialize material, loads and output data
     * \param [in] aProblemParams input XML data
    **********************************************************************************/
    void initialize(Teuchos::ParameterList& aProblemParams)
    {
        // create material model and get stiffness
        //
        Plato::ElasticModelFactory<mSpaceDim> tMaterialFactory(aProblemParams);
        mMaterialModel = tMaterialFactory.create(mSpatialDomain.getMaterialName());
  

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
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<mSpaceDim, mNumMechDims, mNumDofsPerNode, mMechDofOffset>>
                                (aProblemParams.sublist("Mechanical Natural Boundary Conditions"));
        }
  
        auto tResidualParams = aProblemParams.sublist("Elliptic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
          mPlotTable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh metadata
     * \param [in] aMeshSets side-sets metadata
     * \param [in] aDataMap output data map
     * \param [in] aProblemParams input XML data
     * \param [in] aPenaltyParams penalty function input XML data
    **********************************************************************************/
    StabilizedElastostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType      (aSpatialDomain, aDataMap),
        mIndicatorFunction    (aPenaltyParams),
        mApplyTensorWeighting (mIndicatorFunction),
        mApplyVectorWeighting (mIndicatorFunction),
        mApplyScalarWeighting (mIndicatorFunction),
        mBodyLoads            (nullptr),
        mBoundaryLoads        (nullptr),
        mCubatureRule         (std::make_shared<CubatureType>())
    {
        this->initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Evaluate stabilized elastostatics residual
     * \param [in] aStateWS state, {disp_x, disp_y, disp_z, pressure}, workset
     * \param [in] aPressGradWS pressure gradient workset
     * \param [in] aControlWS control workset
     * \param [in] aConfigWS configuration workset
     * \param [in/out] aResultWS result, e.g. residual, workset
     * \param [in] aTimeStep time step
    **********************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT <NodeStateScalarType> & aPressGradWS,
        const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
              Plato::ScalarMultiVectorT <ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType =
      typename Plato::fad_type_t<Plato::SimplexStabilizedMechanics
                <EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset <mSpaceDim> tComputeGradient;
      Plato::StabilizedKinematics   <mSpaceDim> tKinematics;
      Plato::StabilizedKinetics     <mSpaceDim> tKinetics(mMaterialModel);

      Plato::InterpolateFromNodal   <mSpaceDim, mNumDofsPerNode, mPressDofOffset>  tInterpolatePressureFromNodal;
      Plato::InterpolateFromNodal   <mSpaceDim, mSpaceDim, 0 /* dof offset */, mSpaceDim> tInterpolatePGradFromNodal;
      
      Plato::PressureDivergence     <mSpaceDim, mNumDofsPerNode>                  tPressureDivergence;
      Plato::StressDivergence       <mSpaceDim, mNumDofsPerNode, mMechDofOffset>  tStressDivergence;
      Plato::FluxDivergence         <mSpaceDim, mNumDofsPerNode, mPressDofOffset> tStabilizedDivergence;

      Plato::ProjectToNode          <mSpaceDim, mNumDofsPerNode, mPressDofOffset> tProjectVolumeStrain;

      Plato::ScalarVectorT      <ResultScalarType>    tVolStrain      ("volume strain",      tNumCells);
      Plato::ScalarVectorT      <ResultScalarType>    tPressure       ("GP pressure",        tNumCells);
      Plato::ScalarVectorT      <ConfigScalarType>    tCellVolume     ("cell weight",        tNumCells);
      Plato::ScalarMultiVectorT <NodeStateScalarType> tProjectedPGrad ("projected p grad",   tNumCells, mSpaceDim);
      Plato::ScalarMultiVectorT <ResultScalarType>    tCellStab       ("cell stabilization", tNumCells, mSpaceDim);
      Plato::ScalarMultiVectorT <GradScalarType>      tPGrad          ("pressure grad",      tNumCells, mSpaceDim);
      Plato::ScalarMultiVectorT <ResultScalarType>    tDevStress      ("deviatoric stress",  tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT <GradScalarType>      tDGrad          ("displacement grad",  tNumCells, mNumVoigtTerms);
      Plato::ScalarArray3DT     <ConfigScalarType>    tGradient       ("gradient",           tNumCells, mNumNodesPerCell, mSpaceDim);

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto tBasisFunctions   = mCubatureRule->getBasisFunctions();

      auto& tApplyTensorWeighting = mApplyTensorWeighting;
      auto& tApplyVectorWeighting = mApplyVectorWeighting;
      auto& tApplyScalarWeighting = mApplyScalarWeighting;

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
      {
        // compute gradient operator and cell volume
        //
        tComputeGradient(aCellOrdinal, tGradient, aConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute symmetric gradient of displacement, pressure gradient, and temperature gradient
        //
        tKinematics(aCellOrdinal, tDGrad, tPGrad, aStateWS, tGradient);

        // interpolate projected PGrad, pressure, and temperature to gauss point
        //
        tInterpolatePGradFromNodal        ( aCellOrdinal, tBasisFunctions, aPressGradWS, tProjectedPGrad );
        tInterpolatePressureFromNodal     ( aCellOrdinal, tBasisFunctions, aStateWS, tPressure       );

        // compute the constitutive response
        //
        tKinetics(aCellOrdinal,    tCellVolume,
                  tProjectedPGrad, tPressure,
                  tDGrad,          tPGrad,
                  tDevStress,      tVolStrain,  tCellStab);

        // apply weighting
        //
        tApplyTensorWeighting (aCellOrdinal, tDevStress, aControlWS);
        tApplyVectorWeighting (aCellOrdinal, tCellStab,  aControlWS);
        tApplyScalarWeighting (aCellOrdinal, tPressure,  aControlWS);
        tApplyScalarWeighting (aCellOrdinal, tVolStrain, aControlWS);
    
        // compute divergence
        //
        tStressDivergence    (aCellOrdinal, aResultWS,  tDevStress, tGradient, tCellVolume);
        tPressureDivergence  (aCellOrdinal, aResultWS,  tPressure,  tGradient, tCellVolume);
        tStabilizedDivergence(aCellOrdinal, aResultWS,  tCellStab,  tGradient, tCellVolume, -1.0);

        tProjectVolumeStrain (aCellOrdinal, tCellVolume, tBasisFunctions, tVolStrain, aResultWS);

      }, "Cauchy stress");

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aStateWS, aControlWS, aResultWS, -1.0 );
      }

      if(std::count(mPlotTable.begin(), mPlotTable.end(), "pressure"))
      {
          toMap(mDataMap, tPressure, "pressure", mSpatialDomain);
      }
      if( std::count(mPlotTable.begin(),mPlotTable.end(), "deviatoric stress" ) )
      {
          toMap(mDataMap, tDevStress, "deviatoric stress", mSpatialDomain);
      }
    }
    /******************************************************************************//**
     * \brief Evaluate stabilized elastostatics boundary terms residual
     * \param [in] aStateWS state, {disp_x, disp_y, disp_z, pressure}, workset
     * \param [in] aPressGradWS pressure gradient workset
     * \param [in] aControlWS control workset
     * \param [in] aConfigWS configuration workset
     * \param [in/out] aResultWS result, e.g. residual, workset
     * \param [in] aTimeStep time step
    **********************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                             & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType>     & aStateWS,
        const Plato::ScalarMultiVectorT <NodeStateScalarType> & aPressGradWS,
        const Plato::ScalarMultiVectorT <ControlScalarType>   & aControlWS,
        const Plato::ScalarArray3DT     <ConfigScalarType>    & aConfigWS,
              Plato::ScalarMultiVectorT <ResultScalarType>    & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
      if( mBoundaryLoads != nullptr )
      {
          mBoundaryLoads->get( aSpatialModel, aStateWS, aControlWS, aConfigWS, aResultWS, -1.0 );
      }
    }
};
// class ElastostaticResidual

} // namespace Plato
#endif

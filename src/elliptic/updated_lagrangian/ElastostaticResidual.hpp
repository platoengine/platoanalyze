#pragma once

#include <memory>

#include "elliptic/updated_lagrangian/AbstractVectorFunction.hpp"
#include "elliptic/updated_lagrangian/SimplexMechanics.hpp"
#include "elliptic/updated_lagrangian/EllipticUpLagSimplexFadTypes.hpp"

#include "PlatoTypes.hpp"
#include "Strain.hpp"
#include "elliptic/updated_lagrangian/EllipticUpLagLinearStress.hpp"
#include "StressDivergence.hpp"
#include "ApplyWeighting.hpp"
#include "CellForcing.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"
#include "ToMap.hpp"
#include "VonMisesYieldFunction.hpp"

#include "ElasticModelFactory.hpp"
#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"

#include "elliptic/updated_lagrangian/ExpInstMacros.hpp"

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

/******************************************************************************//**
 * \brief Elastostatic vector function interface
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * \tparam IndicatorFunctionType penalty function used for density-based methods
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ElastostaticResidual :
        public Plato::Elliptic::UpdatedLagrangian::SimplexMechanics<EvaluationType::SpatialDim>,
        public Plato::Elliptic::UpdatedLagrangian::AbstractVectorFunction<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    using PhysicsType = typename Plato::Elliptic::UpdatedLagrangian::SimplexMechanics<mSpaceDim>;

    using PhysicsType::mNumVoigtTerms;
    using PhysicsType::mNumDofsPerNode;
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;

    using Plato::Elliptic::UpdatedLagrangian::AbstractVectorFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::UpdatedLagrangian::AbstractVectorFunction<EvaluationType>::mDataMap;

    using GlobalStateScalarType = typename EvaluationType::GlobalStateScalarType;
    using LocalStateScalarType  = typename EvaluationType::LocalStateScalarType;
    using ControlScalarType     = typename EvaluationType::ControlScalarType;
    using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
    using ResultScalarType      = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = Plato::Elliptic::UpdatedLagrangian::AbstractVectorFunction<EvaluationType>;
    using CubatureType  = Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>;
    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting;

    std::shared_ptr<Plato::BodyLoads<EvaluationType, PhysicsType>> mBodyLoads;
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim,mNumDofsPerNode>> mBoundaryLoads;
    std::shared_ptr<Plato::CellForcing<mNumVoigtTerms>> mCellForcing;
    std::shared_ptr<CubatureType> mCubatureRule;

    Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> mMaterialModel;

    std::vector<std::string> mPlotTable;

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap Plato Analyze database
     * \param [in] aProblemParams input parameters for overall problem
     * \param [in] aPenaltyParams input parameters for penalty function
    **********************************************************************************/
    ElastostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams
    ) :
        FunctionBaseType   (aSpatialDomain, aDataMap),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction),
        mBodyLoads         (nullptr),
        mBoundaryLoads     (nullptr),
        mCellForcing       (nullptr),
        mCubatureRule      (std::make_shared<CubatureType>())
    {
        // create material model and get stiffness
        //
        Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        // parse body loads
        // 
        if(aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType, PhysicsType>>(aProblemParams.sublist("Body Loads"));
        }
  
        // parse boundary Conditions
        // 
        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<mSpaceDim,mNumDofsPerNode>>(aProblemParams.sublist("Natural Boundary Conditions"));
        }
        // parse cell problem forcing
        //
        if(aProblemParams.isSublist("Cell Problem Forcing"))
        {
            Plato::OrdinalType tColumnIndex = aProblemParams.sublist("Cell Problem Forcing").get<Plato::OrdinalType>("Column Index");
            mCellForcing = std::make_shared<Plato::CellForcing<mNumVoigtTerms>>(mMaterialModel->getStiffnessMatrix(), tColumnIndex);
        }

        auto tResidualParams = aProblemParams.sublist("Updated Lagrangian Elliptic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
          mPlotTable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }

    }

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override
    {
      Plato::ScalarMultiVector tDisplacements = aSolutions.get("State");
      Plato::Solutions tSolutionsOutput(aSolutions.physics(), aSolutions.pde());
      tSolutionsOutput.set("Displacement", tDisplacements);
      tSolutionsOutput.setNumDofs("Displacement", 3);
      return tSolutionsOutput;
    }
    
    /******************************************************************************//**
     * \brief Evaluate vector function
     *
     * \param [in] aGlobalState 2D array with global state variables (C,DOF)
     * \param [in] aLocalState 2D array with local state variables (C, NS)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     *
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
     * NS = number of local states per cell
    **********************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <GlobalStateScalarType> & aGlobalState,
        const Plato::ScalarMultiVectorT <LocalStateScalarType>  & aLocalState,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
      using SimplexPhysics = typename Plato::SimplexMechanics<mSpaceDim>;
      using StrainScalarType =
        typename Plato::fad_type_t<Plato::Elliptic::UpdatedLagrangian::SimplexMechanics<EvaluationType::SpatialDim>, GlobalStateScalarType, ConfigScalarType>;

      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
      Plato::Strain<mSpaceDim>                 tComputeVoigtStrainIncrement;
      Plato::Elliptic::UpdatedLagrangian::EllipticUpLagLinearStress<EvaluationType,
                          SimplexPhysics>      tComputeVoigtStress(mMaterialModel);
      Plato::StressDivergence<mSpaceDim>       tComputeStressDivergence;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<StrainScalarType>
        tStrainIncrement("strain increment", tNumCells, mNumVoigtTerms);
    
      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);
    
      Plato::ScalarMultiVectorT<ResultScalarType>
        tStress("stress", tNumCells, mNumVoigtTerms);
    
      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute strain increment
        tComputeVoigtStrainIncrement(aCellOrdinal, tStrainIncrement, aGlobalState, tGradient);
    
        // compute stress
        tComputeVoigtStress(aCellOrdinal, tStress, tStrainIncrement, aLocalState);
      }, "Cauchy stress");

      if( mCellForcing != nullptr )
      {
          mCellForcing->add( tStress );
      }

      auto& tApplyWeighting = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        // apply weighting
        tApplyWeighting(aCellOrdinal, tStress, aControl);
    
        // compute stress divergence
        tComputeStressDivergence(aCellOrdinal, aResult, tStress, tGradient, tCellVolume);
      }, "Apply weighting and compute divergence");

      if( mBodyLoads != nullptr )
      {
          mBodyLoads->get( mSpatialDomain, aGlobalState, aControl, aResult, -1.0 );
      }

      // Required in DataMap for lagrangian update
      Plato::toMap(mDataMap, tStrainIncrement, "strain increment", mSpatialDomain);

      if(std::count(mPlotTable.begin(), mPlotTable.end(), "stress")) { Plato::toMap(mDataMap, tStress, "stress", mSpatialDomain); }
      if(std::count(mPlotTable.begin(), mPlotTable.end(), "Vonmises")) { this->outputVonMises(tStress, mSpatialDomain); }
    }
    /******************************************************************************//**
     * \brief Evaluate vector function
     *
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aGlobalState 2D array with state variables (C,DOF)
     * \param [in] aLocalState 2D array with local state variables (C, NS)
     * \param [in] aControl 2D array with control variables (C,N)
     * \param [in] aConfig 3D array with control variables (C,N,D)
     * \param [in] aResult 1D array with control variables (C,DOF)
     * \param [in] aTimeStep current time step
     *
     * Nomenclature: C = number of cells, DOF = number of degrees of freedom per cell
     * N = number of nodes per cell, D = spatial dimensions
     * NS = number of local states
    **********************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                               & aSpatialModel,
        const Plato::ScalarMultiVectorT <GlobalStateScalarType> & aGlobalState,
        const Plato::ScalarMultiVectorT <LocalStateScalarType>  & aLocalState,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aGlobalState, aControl, aConfig, aResult, -1.0 );
        }
    }


    /**********************************************************************//**
     * \brief Compute Von Mises stress field and copy data into output data map
     * \param [in] aCauchyStress Cauchy stress tensor
    **************************************************************************/
    void
    outputVonMises(
        const Plato::ScalarMultiVectorT<ResultScalarType> & aCauchyStress,
        const Plato::SpatialDomain                        & aSpatialDomain
    ) const
    {
            auto tNumCells = aSpatialDomain.numCells();
            Plato::VonMisesYieldFunction<mSpaceDim> tComputeVonMises;
            Plato::ScalarVectorT<ResultScalarType> tVonMises("Von Mises", tNumCells);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                tComputeVonMises(aCellOrdinal, aCauchyStress, tVonMises);
            }, "Compute VonMises Stress");

            Plato::toMap(mDataMap, tVonMises, "Vonmises", aSpatialDomain);
    }
};
// class ElastostaticResidual

} // namespace UpdatedLagrangian

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC(Plato::Elliptic::UpdatedLagrangian::ElastostaticResidual, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC(Plato::Elliptic::UpdatedLagrangian::ElastostaticResidual, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC(Plato::Elliptic::UpdatedLagrangian::ElastostaticResidual, Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 3)
#endif

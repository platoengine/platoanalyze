#ifndef HEAT_EQUATION_RESIDUAL_HPP
#define HEAT_EQUATION_RESIDUAL_HPP

#include "SimplexThermal.hpp"
#include "ApplyWeighting.hpp"
#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "ThermalContent.hpp"
#include "FluxDivergence.hpp"
#include "PlatoMathHelpers.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

#include "ThermalConductivityMaterial.hpp"
#include "ThermalMassMaterial.hpp"
#include "parabolic/AbstractVectorFunction.hpp"
#include "ImplicitFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "ProjectToNode.hpp"
#include "ApplyWeighting.hpp"
#include "NaturalBCs.hpp"

#include "parabolic/ParabolicSimplexFadTypes.hpp"
#include "parabolic/ExpInstMacros.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class HeatEquationResidual : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::Parabolic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType SpaceDim = EvaluationType::SpatialDim;

    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerNode;

    using Plato::Parabolic::AbstractVectorFunction<EvaluationType>::mSpatialDomain;
    using Plato::Parabolic::AbstractVectorFunction<EvaluationType>::mDataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using StateDotScalarType  = typename EvaluationType::StateDotScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim,SpaceDim,IndicatorFunctionType> mApplyFluxWeighting;
    Plato::ApplyWeighting<SpaceDim,mNumDofsPerNode,IndicatorFunctionType> mApplyMassWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<SpaceDim>> mCubatureRule;
    std::shared_ptr<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>> mBoundaryLoads;

    Teuchos::RCP<Plato::MaterialModel<SpaceDim>> mThermalMassMaterialModel;
    Teuchos::RCP<Plato::MaterialModel<SpaceDim>> mThermalConductivityMaterialModel;


  public:
    /**************************************************************************/
    HeatEquationResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & problemParams,
              Teuchos::ParameterList & penaltyParams
    ) :
     Plato::Parabolic::AbstractVectorFunction<EvaluationType>(aSpatialDomain, aDataMap, {"Temperature"}),
     mIndicatorFunction  (penaltyParams),
     mApplyFluxWeighting (mIndicatorFunction),
     mApplyMassWeighting (mIndicatorFunction),
     mCubatureRule       (std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpaceDim>>()),
     mBoundaryLoads      (nullptr)
    /**************************************************************************/
    {
        {
            Plato::ThermalConductionModelFactory<SpaceDim> mmfactory(problemParams);
            mThermalConductivityMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
        }

        {
            Plato::ThermalMassModelFactory<SpaceDim> mmfactory(problemParams);
            mThermalMassMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
        }

      // parse boundary Conditions
      // 
      if(problemParams.isSublist("Natural Boundary Conditions"))
      {
          mBoundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>>(problemParams.sublist("Natural Boundary Conditions"));
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
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarArray3DT<ConfigScalarType> tGradient ("gradient", tNumCells, mNumNodesPerCell, SpaceDim);

      Plato::ScalarMultiVectorT<GradScalarType  > tGrad ("temperature gradient", tNumCells, SpaceDim);
      Plato::ScalarMultiVectorT<ResultScalarType> tFlux ("thermal flux",         tNumCells, SpaceDim);

      Plato::ScalarVectorT<ConfigScalarType>   tCellVolume        ("cell weight",                 tNumCells);
      Plato::ScalarVectorT<StateScalarType >   tTemperature       ("Gauss point temperature",     tNumCells);
      Plato::ScalarVectorT<StateDotScalarType> tTemperatureRate   ("Gauss point temperature dot", tNumCells);
      Plato::ScalarVectorT<ResultScalarType>   tThermalEnergyRate ("Gauss point energy rate",     tNumCells);

      // create a bunch of functors:
      Plato::ComputeGradientWorkset<SpaceDim>  tComputeGradient;

      Plato::ScalarGrad<SpaceDim>            tScalarGrad;
      Plato::ThermalFlux<SpaceDim>           tThermalFlux(mThermalConductivityMaterialModel);
      Plato::FluxDivergence<SpaceDim>        tFluxDivergence;

      Plato::InterpolateFromNodal <SpaceDim, mNumDofsPerNode> tInterpolateFromNodal;
      Plato::ProjectToNode        <SpaceDim, mNumDofsPerNode> tProjectThermalEnergyRate;

      Plato::ThermalContent<SpaceDim> tThermalContent(mThermalMassMaterialModel);
      
      auto tBasisFunctions = mCubatureRule->getBasisFunctions();
    
      auto& tApplyFluxWeighting  = mApplyFluxWeighting;
      auto& tApplyMassWeighting  = mApplyMassWeighting;
      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
      {
    
        tComputeGradient(tCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(tCellOrdinal) *= tQuadratureWeight;
    
        // compute temperature gradient
        //
        tScalarGrad(tCellOrdinal, tGrad, aState, tGradient);
    
        // compute flux
        //
        tThermalFlux(tCellOrdinal, tFlux, tGrad);
    
        // apply weighting
        //
        tApplyFluxWeighting(tCellOrdinal, tFlux, aControl);

        // compute stress divergence
        //
        tFluxDivergence(tCellOrdinal, aResult, tFlux, tGradient, tCellVolume);

        // add capacitance terms
        
        // compute temperature at gausspoints
        //
        tInterpolateFromNodal(tCellOrdinal, tBasisFunctions, aStateDot, tTemperatureRate);
        tInterpolateFromNodal(tCellOrdinal, tBasisFunctions, aState,    tTemperature    );

        // compute the time rate of internal thermal energy
        //
        tThermalContent(tCellOrdinal, tThermalEnergyRate, tTemperatureRate, tTemperature);

        // apply weighting
        //
        tApplyMassWeighting(tCellOrdinal, tThermalEnergyRate, aControl);

        // project to nodes
        //
        tProjectThermalEnergyRate(tCellOrdinal, tCellVolume, tBasisFunctions, tThermalEnergyRate, aResult);

      },"flux divergence");

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
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult);
        }
    }
};
// class HeatEquationResidual

} // namespace Parabolic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_PARABOLIC_EXPL_DEC(Plato::Parabolic::HeatEquationResidual, Plato::SimplexThermal, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_PARABOLIC_EXPL_DEC(Plato::Parabolic::HeatEquationResidual, Plato::SimplexThermal, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_PARABOLIC_EXPL_DEC(Plato::Parabolic::HeatEquationResidual, Plato::SimplexThermal, 3)
#endif

#endif

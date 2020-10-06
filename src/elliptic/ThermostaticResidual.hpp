#ifndef THERMOSTATIC_RESIDUAL_HPP
#define THERMOSTATIC_RESIDUAL_HPP

#include "SimplexThermal.hpp"
#include "ApplyWeighting.hpp"
#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "FluxDivergence.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoMathHelpers.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "ToMap.hpp"
#include "InterpolateFromNodal.hpp"

#include "ThermalConductivityMaterial.hpp"
#include "elliptic/AbstractVectorFunction.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyWeighting.hpp"
#include "NaturalBCs.hpp"
#include "SimplexFadTypes.hpp"

#include "ExpInstMacros.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class ThermostaticResidual : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::Elliptic::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermal<mSpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermal<mSpaceDim>::mNumDofsPerNode;

    using Plato::Elliptic::AbstractVectorFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractVectorFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mQuadratureWeight;

    IndicatorFunctionType mIndicatorFunction;
    ApplyWeighting<mSpaceDim,mSpaceDim,IndicatorFunctionType> mApplyWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>> mCubatureRule;
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim,mNumDofsPerNode>> mBoundaryLoads;

    Teuchos::RCP<Plato::MaterialModel<mSpaceDim>> mMaterialModel;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    ThermostaticResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & penaltyParams
    ) :
        Plato::Elliptic::AbstractVectorFunction<EvaluationType>(aSpatialDomain, aDataMap),
        mIndicatorFunction(penaltyParams),
        mApplyWeighting(mIndicatorFunction),
        mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>>()),
        mBoundaryLoads(nullptr)
    /**************************************************************************/
    {
        Plato::ThermalConductionModelFactory<mSpaceDim> tMaterialFactory(aProblemParams);
        mMaterialModel = tMaterialFactory.create(aSpatialDomain.getMaterialName());

      // parse boundary Conditions
      // 
      if(aProblemParams.isSublist("Natural Boundary Conditions"))
      {
          mBoundaryLoads = std::make_shared<Plato::NaturalBCs<mSpaceDim,mNumDofsPerNode>>(aProblemParams.sublist("Natural Boundary Conditions"));
      }

        auto tResidualParams = aProblemParams.sublist("Elliptic");
        if( tResidualParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlottable = tResidualParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Kokkos::View<GradScalarType**, Plato::Layout, Plato::MemSpace>
        tGrad("temperature gradient",tNumCells,mSpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("gradient",tNumCells,mNumNodesPerCell,mSpaceDim);

      Kokkos::View<ResultScalarType**, Plato::Layout, Plato::MemSpace>
        tFlux("thermal flux",tNumCells,mSpaceDim);

      Plato::ScalarVectorT<StateScalarType> 
        tTemperature("Gauss point temperature", tNumCells);


      // create a bunch of functors:
      Plato::ComputeGradientWorkset<mSpaceDim>  tComputeGradient;

      Plato::ScalarGrad<mSpaceDim>            tScalarGrad;
      Plato::ThermalFlux<mSpaceDim>           tThermalFlux(mMaterialModel);
      Plato::FluxDivergence<mSpaceDim>        tFluxDivergence;

      Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode> tInterpolateFromNodal;

    
      auto& tApplyWeighting  = mApplyWeighting;
      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto tBasisFunctions    = mCubatureRule->getBasisFunctions();

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
      {
    
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;
    
        // compute temperature gradient
        //
        tScalarGrad(aCellOrdinal, tGrad, aState, tGradient);
    
        // compute flux
        //
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aState, tTemperature);
        tThermalFlux(aCellOrdinal, tFlux, tGrad, tTemperature);
    
        // apply weighting
        //
        tApplyWeighting(aCellOrdinal, tFlux, aControl);
    
        // compute stress divergence
        //
        tFluxDivergence(aCellOrdinal, aResult, tFlux, tGradient, tCellVolume, -1.0);
        
      },"flux divergence");

     if( std::count(mPlottable.begin(),mPlottable.end(),"tgrad") ) toMap(mDataMap, tGrad, "tgrad", mSpatialDomain);
     if( std::count(mPlottable.begin(),mPlottable.end(),"flux" ) ) toMap(mDataMap, tFlux, "flux" , mSpatialDomain);
    }

    /**************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        if( mBoundaryLoads != nullptr )
        {
            mBoundaryLoads->get(aSpatialModel, aState, aControl, aConfig, aResult,  1.0 );
        }
    }
};
// class ThermostaticResidual

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC(Plato::Elliptic::ThermostaticResidual, Plato::SimplexThermal, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC(Plato::Elliptic::ThermostaticResidual, Plato::SimplexThermal, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC(Plato::Elliptic::ThermostaticResidual, Plato::SimplexThermal, 3)
#endif

#endif

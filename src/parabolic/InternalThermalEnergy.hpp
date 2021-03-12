#ifndef PARABOLIC_INTERNAL_THERMAL_ENERGY_HPP
#define PARABOLIC_INTERNAL_THERMAL_ENERGY_HPP

#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "SimplexThermal.hpp"
#include "ImplicitFunctors.hpp"
#include "ThermalConductivityMaterial.hpp"
#include "parabolic/ParabolicSimplexFadTypes.hpp"
#include "parabolic/AbstractScalarFunction.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

#include "parabolic/ExpInstMacros.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************//**
 * @brief Internal energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalThermalEnergy : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::Parabolic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermal<mSpaceDim>::mNumDofsPerCell;

    using Plato::Parabolic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Parabolic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType    = typename EvaluationType::StateScalarType;
    using StateDotScalarType = typename EvaluationType::StateDotScalarType;
    using ControlScalarType  = typename EvaluationType::ControlScalarType;
    using ConfigScalarType   = typename EvaluationType::ConfigScalarType;
    using ResultScalarType   = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = Plato::Parabolic::AbstractScalarFunction<EvaluationType>;
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<mSpaceDim>;

    IndicatorFunctionType mIndicatorFunction;
    ApplyWeighting<mSpaceDim,mSpaceDim,IndicatorFunctionType> mApplyWeighting;

    std::shared_ptr<CubatureType> mCubatureRule;
    Teuchos::RCP<Plato::MaterialModel<mSpaceDim>> mThermalConductivityMaterialModel;

  public:
    /**************************************************************************/
    InternalThermalEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
    ) :
        FunctionBaseType   (aSpatialDomain, aDataMap, aFunctionName),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction),
        mCubatureRule      (std::make_shared<CubatureType>())
    /**************************************************************************/
    {
      Plato::ThermalConductionModelFactory<mSpaceDim> mmfactory(aProblemParams);
      mThermalConductivityMaterialModel = mmfactory.create(aSpatialDomain.getMaterialName());
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>    & aState,
        const Plato::ScalarMultiVectorT <StateDotScalarType> & aStateDot,
        const Plato::ScalarMultiVectorT <ControlScalarType>  & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>   & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>   & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      Plato::ComputeGradientWorkset<mSpaceDim> computeGradient;
      Plato::ScalarGrad<mSpaceDim>             scalarGrad;
      Plato::ThermalFlux<mSpaceDim>            thermalFlux(mThermalConductivityMaterialModel);
      Plato::ScalarProduct<mSpaceDim>          scalarProduct;

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight", tNumCells);

      Kokkos::View<GradScalarType**, Plato::Layout, Plato::MemSpace>
        tgrad("temperature gradient", tNumCells,mSpaceDim);

      Kokkos::View<ConfigScalarType***, Plato::Layout, Plato::MemSpace>
        gradient("gradient", tNumCells,mNumNodesPerCell,mSpaceDim);

      Kokkos::View<ResultScalarType**, Plato::Layout, Plato::MemSpace>
        tflux("thermal flux", tNumCells,mSpaceDim);

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto applyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0, tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
      {
        computeGradient(aCellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute temperature gradient
        //
        scalarGrad(aCellOrdinal, tgrad, aState, gradient);

        // compute flux
        //
        thermalFlux(aCellOrdinal, tflux, tgrad);

        // apply weighting
        //
        applyWeighting(aCellOrdinal, tflux, aControl);
    
        // compute element internal energy (inner product of tgrad and weighted tflux)
        //
        scalarProduct(aCellOrdinal, aResult, tflux, tgrad, cellVolume, -1.0);

      },"energy gradient");
    }
};
// class InternalThermalEnergy

} // namespace Parabolic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_PARABOLIC_EXPL_DEC(Plato::Parabolic::InternalThermalEnergy, Plato::SimplexThermal, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_PARABOLIC_EXPL_DEC(Plato::Parabolic::InternalThermalEnergy, Plato::SimplexThermal, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_PARABOLIC_EXPL_DEC(Plato::Parabolic::InternalThermalEnergy, Plato::SimplexThermal, 3)
#endif

#endif

#ifndef FLUX_P_NORM_HPPS
#define FLUX_P_NORM_HPP

#include "elliptic/AbstractScalarFunction.hpp"

#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "VectorPNorm.hpp"
#include "SimplexThermal.hpp"
#include "ApplyWeighting.hpp"
#include "SimplexFadTypes.hpp"
#include "ThermalConductivityMaterial.hpp"
#include "ImplicitFunctors.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "ExpInstMacros.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class FluxPNorm : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerCell;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim,SpaceDim,IndicatorFunctionType> mApplyWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<SpaceDim>> mCubatureRule;
    Teuchos::RCP<Plato::MaterialModel<SpaceDim>> mMaterialModel;

    Plato::OrdinalType mExponent;

  public:
    /**************************************************************************/
    FluxPNorm(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aProblemParams, 
              Teuchos::ParameterList & aPenaltyParams,
              std::string              aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aFunctionName),
        mIndicatorFunction(aPenaltyParams),
        mApplyWeighting(mIndicatorFunction),
        mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpaceDim>>())
    /**************************************************************************/
    {
      Plato::ThermalConductionModelFactory<SpaceDim> mmfactory(aProblemParams);
      mMaterialModel = mmfactory.create(mSpatialDomain.getMaterialName());

      auto params = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);

      mExponent = params.get<Plato::Scalar>("Exponent");
    }

    /**************************************************************************/
    void evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto numCells = mSpatialDomain.numCells();

      Plato::ComputeGradientWorkset<SpaceDim> computeGradient;
      Plato::ScalarGrad<SpaceDim>             scalarGrad;
      Plato::ThermalFlux<SpaceDim>            thermalFlux(mMaterialModel);
      Plato::VectorPNorm<SpaceDim>            vectorPNorm;

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>,StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Kokkos::View<GradScalarType**, Plato::Layout, Plato::MemSpace>
        tgrad("temperature gradient",numCells,SpaceDim);

      Kokkos::View<ConfigScalarType***, Plato::Layout, Plato::MemSpace>
        gradient("gradient",numCells,mNumNodesPerCell,SpaceDim);

      Kokkos::View<ResultScalarType**, Plato::Layout, Plato::MemSpace>
        tflux("thermal flux",numCells,SpaceDim);

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto& applyWeighting  = mApplyWeighting;
      auto exponent         = mExponent;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
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
    
        // compute vector p-norm of flux
        //
        vectorPNorm(aCellOrdinal, aResult, tflux, exponent, cellVolume);

      },"Flux P-norm");
    }

    /**************************************************************************/
    void
    postEvaluate( 
      Plato::ScalarVector resultVector,
      Plato::Scalar       resultScalar)
    /**************************************************************************/
    {
      auto scale = pow(resultScalar,(1.0-mExponent)/mExponent)/mExponent;
      auto numEntries = resultVector.size();
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numEntries), LAMBDA_EXPRESSION(int entryOrdinal)
      {
        resultVector(entryOrdinal) *= scale;
      },"scale vector");
    }

    /**************************************************************************/
    void
    postEvaluate( Plato::Scalar& resultValue )
    /**************************************************************************/
    {
      resultValue = pow(resultValue, 1.0/mExponent);
    }
};
// class FluxPNorm

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC(Plato::Elliptic::FluxPNorm, Plato::SimplexThermal, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC(Plato::Elliptic::FluxPNorm, Plato::SimplexThermal, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC(Plato::Elliptic::FluxPNorm, Plato::SimplexThermal, 3)
#endif

#endif

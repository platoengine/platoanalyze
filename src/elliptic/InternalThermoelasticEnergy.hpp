#ifndef INTERNAL_THERMOELASTIC_ENERGY_HPP
#define INTERNAL_THERMOELASTIC_ENERGY_HPP

#include "elliptic/AbstractScalarFunction.hpp"
#include "elliptic/EllipticSimplexFadTypes.hpp"

#include "SimplexThermomechanics.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "TMKinematics.hpp"
#include "TMKinetics.hpp"
#include "ImplicitFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ThermoelasticMaterial.hpp"
#include "ToMap.hpp"
#include "ExpInstMacros.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * @brief Compute internal thermo-elastic energy criterion, given by
 *                  /f$ f(z) = u^{T}K_u(z)u + T^{T}K_t(z)T /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalThermoelasticEnergy : 
  public Plato::SimplexThermomechanics<EvaluationType::SpatialDim>,
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;
    static constexpr Plato::OrdinalType TDofOffset = mSpaceDim;
    
    using Plato::SimplexThermomechanics<mSpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermomechanics<mSpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermomechanics<mSpaceDim>::mNumDofsPerNode;

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Teuchos::RCP<Plato::MaterialModel<mSpaceDim>> mMaterialModel;
    
    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<mSpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyStressWeighting;
    Plato::ApplyWeighting<mSpaceDim, mSpaceDim,      IndicatorFunctionType> mApplyFluxWeighting;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    std::vector<std::string> mPlottable;

  public:
    /**************************************************************************/
    InternalThermoelasticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aFunctionName),
        mIndicatorFunction    (aPenaltyParams),
        mApplyStressWeighting (mIndicatorFunction),
        mApplyFluxWeighting   (mIndicatorFunction),
        mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    /**************************************************************************/
    {
        Teuchos::ParameterList tProblemParams(aProblemParams);

        auto tMaterialName = aSpatialDomain.getMaterialName();

        if( aProblemParams.isSublist("Material Models") == false )
        {
            THROWERR("Required input list ('Material Models') is missing.");
        }

        if( aProblemParams.sublist("Material Models").isSublist(tMaterialName) == false )
        {
            std::stringstream ss;
            ss << "Specified material model ('" << tMaterialName << "') is not defined";
            THROWERR(ss.str());
        }

        auto& tParams = aProblemParams.sublist(aFunctionName);
        if( tParams.get<bool>("Include Thermal Strain", true) == false )
        {
           auto tMaterialParams = tProblemParams.sublist("Material Models").sublist(tMaterialName);
           tMaterialParams.sublist("Cubic Linear Thermoelastic").set("a11",0.0);
           tMaterialParams.sublist("Cubic Linear Thermoelastic").set("a22",0.0);
           tMaterialParams.sublist("Cubic Linear Thermoelastic").set("a33",0.0);
        }

        Plato::ThermoelasticModelFactory<mSpaceDim> mmfactory(tProblemParams);
        mMaterialModel = mmfactory.create(tMaterialName);

        if( tProblemParams.isType<Teuchos::Array<std::string>>("Plottable") )
        {
            mPlottable = tProblemParams.get<Teuchos::Array<std::string>>("Plottable").toVector();
        }
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>  & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>  & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermomechanics<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ComputeGradientWorkset<mSpaceDim> computeGradient;
      Plato::TMKinematics<mSpaceDim>                  kinematics;
      Plato::TMKinetics<mSpaceDim>                    kinetics(mMaterialModel);

      Plato::ScalarProduct<mNumVoigtTerms>          mechanicalScalarProduct;
      Plato::ScalarProduct<mSpaceDim>                 thermalScalarProduct;

      Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode, TDofOffset> interpolateFromNodal;

      Plato::ScalarVectorT<ConfigScalarType> cellVolume("cell weight",tNumCells);

      Plato::ScalarMultiVectorT<GradScalarType>   strain("strain", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<GradScalarType>   tgrad ("tgrad",  tNumCells, mSpaceDim);

      Plato::ScalarMultiVectorT<ResultScalarType> stress("stress", tNumCells, mNumVoigtTerms);
      Plato::ScalarMultiVectorT<ResultScalarType> flux  ("flux",   tNumCells, mSpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>   gradient("gradient",tNumCells,mNumNodesPerCell,mSpaceDim);

      Plato::ScalarVectorT<StateScalarType> temperature("Gauss point temperature", tNumCells);

      auto quadratureWeight = mCubatureRule->getCubWeight();
      auto basisFunctions   = mCubatureRule->getBasisFunctions();

      auto& applyStressWeighting = mApplyStressWeighting;
      auto& applyFluxWeighting   = mApplyFluxWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        computeGradient(aCellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(aCellOrdinal) *= quadratureWeight;

        // compute strain and temperature gradient
        //
        kinematics(aCellOrdinal, strain, tgrad, aState, gradient);

        // compute stress and thermal flux
        //
        interpolateFromNodal(aCellOrdinal, basisFunctions, aState, temperature);
        kinetics(aCellOrdinal, stress, flux, strain, tgrad, temperature);

        // apply weighting
        //
        applyStressWeighting(aCellOrdinal, stress, aControl);
        applyFluxWeighting  (aCellOrdinal, flux,   aControl);

        // compute element internal energy (inner product of strain and weighted stress)
        //
        mechanicalScalarProduct(aCellOrdinal, aResult, stress, strain, cellVolume);
        thermalScalarProduct   (aCellOrdinal, aResult, flux,   tgrad,  cellVolume);

      },"energy gradient");
    }
};

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC(Plato::Elliptic::InternalThermoelasticEnergy, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC(Plato::Elliptic::InternalThermoelasticEnergy, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC(Plato::Elliptic::InternalThermoelasticEnergy, Plato::SimplexThermomechanics, 3)
#endif

#endif

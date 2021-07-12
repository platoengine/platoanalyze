#pragma once

#include "elliptic/updated_lagrangian/EllipticUpLagSimplexFadTypes.hpp"
#include "elliptic/updated_lagrangian/AbstractScalarFunction.hpp"
#include "elliptic/updated_lagrangian/SimplexMechanics.hpp"
#include "elliptic/updated_lagrangian/ExpInstMacros.hpp"

#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "Strain.hpp"
#include "LinearStress.hpp"
#include "TensorPNorm.hpp"
#include "ElasticModelFactory.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ImplicitFunctors.hpp"
#include "ToMap.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

/******************************************************************************//**
 * @brief Internal energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class StressPNorm : 
  public Plato::Elliptic::UpdatedLagrangian::SimplexMechanics<EvaluationType::SpatialDim>,
  public Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;
    
    using Plato::Elliptic::UpdatedLagrangian::SimplexMechanics<mSpaceDim>::mNumVoigtTerms;
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;

    using Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<EvaluationType>::mDataMap;

    using GlobalStateScalarType = typename EvaluationType::GlobalStateScalarType;
    using LocalStateScalarType  = typename EvaluationType::LocalStateScalarType;
    using ControlScalarType     = typename EvaluationType::ControlScalarType;
    using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
    using ResultScalarType      = typename EvaluationType::ResultScalarType;

    IndicatorFunctionType mIndicatorFunction; /*!< penalty function */
    Plato::ApplyWeighting<mSpaceDim,mNumVoigtTerms,IndicatorFunctionType> mApplyWeighting; /*!< apply penalty function */

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    std::vector<std::string> mPlottable; /*!< database of output field names */

    Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> mMaterialModel;

    Teuchos::RCP<TensorNormBase<mNumVoigtTerms,EvaluationType>> mNorm;

  public:
    /******************************************************************************//**
     * @brief Constructor
     * @param aSpatialDomain Plato Analyze spatial domain
     * @param aProblemParams input database for overall problem
     * @param aPenaltyParams input database for penalty function
    **********************************************************************************/
    StressPNorm(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aPenaltyParams,
              std::string            & aFunctionName
    ) :
        Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aFunctionName),
        mIndicatorFunction (aPenaltyParams),
        mApplyWeighting    (mIndicatorFunction),
        mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    {
        Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        if(aProblemParams.isType < Teuchos::Array < std::string >> ("Plottable"))
        {
            mPlottable = aProblemParams.get < Teuchos::Array < std::string >> ("Plottable").toVector();
        }

        auto params = aProblemParams.sublist("Criteria").get<Teuchos::ParameterList>(aFunctionName);

        TensorNormFactory<mNumVoigtTerms, EvaluationType> tNormFactory;
        mNorm = tNormFactory.create(params);
    }

    /******************************************************************************//**
     * @brief Evaluate internal elastic energy function
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [out] aResult 1D container of cell criterion values
     * @param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <GlobalStateScalarType> & aGlobalState,
        const Plato::ScalarMultiVectorT <LocalStateScalarType>  & aLocalState,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep = 0.0) const
    {
        auto tNumCells = mSpatialDomain.numCells();

        Plato::Strain<mSpaceDim> tComputeVoigtStrainIncrement;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::LinearStress<mSpaceDim> tComputeVoigtStress(mMaterialModel);

      using StrainScalarType = 
        typename Plato::fad_type_t<Plato::Elliptic::UpdatedLagrangian::SimplexMechanics
           <EvaluationType::SpatialDim>, GlobalStateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Kokkos::View<StrainScalarType**, Plato::Layout, Plato::MemSpace>
        tStrainIncrement("strain increment", tNumCells, mNumVoigtTerms);

      Kokkos::View<ConfigScalarType***, Plato::Layout, Plato::MemSpace>
        tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

      Kokkos::View<ResultScalarType**, Plato::Layout, Plato::MemSpace>
        tStress("stress", tNumCells, mNumVoigtTerms);

      auto tQuadratureWeight = mCubatureRule->getCubWeight();

      auto tApplyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute strain increment
        //
        tComputeVoigtStrainIncrement(aCellOrdinal, tStrainIncrement, aGlobalState, tGradient);

        // compute stress
        //
        tComputeVoigtStress(aCellOrdinal, tStress, tStrainIncrement, aLocalState);

        // apply weighting
        //
        tApplyWeighting(aCellOrdinal, tStress, aControl);
    
      },"Compute Stress");

      mNorm->evaluate(aResult, tStress, aControl, tCellVolume);

    }
};
// class StressPNorm

} // namespace UpdatedLagrangian

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC(Plato::Elliptic::UpdatedLagrangian::StressPNorm, 
                              Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC(Plato::Elliptic::UpdatedLagrangian::StressPNorm,
                              Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_ELLIPTIC_UPLAG_EXPL_DEC(Plato::Elliptic::UpdatedLagrangian::StressPNorm,
                              Plato::Elliptic::UpdatedLagrangian::SimplexMechanics, 3)
#endif

#ifndef HYPERBOLIC_INTERNAL_ELASTIC_ENERGY_HPP
#define HYPERBOLIC_INTERNAL_ELASTIC_ENERGY_HPP

//#include "SimplexFadTypes.hpp"
#include "SpatialModel.hpp"
#include "SimplexMechanics.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "Strain.hpp"
#include "LinearStress.hpp"
#include "ElasticModelFactory.hpp"
#include "ImplicitFunctors.hpp"
#include "ToMap.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "hyperbolic/HyperbolicAbstractScalarFunction.hpp"

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************//**
 * \brief Internal energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalElasticEnergy :
  public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
  public Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of Voigt terms */
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */
    using Plato::SimplexMechanics<mSpaceDim>::mNumDofsPerCell; /*!< number of degree of freedom per cell */

    using Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>::mSpatialDomain; /*!< mesh database */
    using Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>::mDataMap; /*!< Plato Analyze database */

    using StateScalarType       = typename EvaluationType::StateScalarType;
    using StateDotScalarType    = typename EvaluationType::StateDotScalarType;
    using StateDotDotScalarType = typename EvaluationType::StateDotDotScalarType;
    using ControlScalarType     = typename EvaluationType::ControlScalarType;
    using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
    using ResultScalarType      = typename EvaluationType::ResultScalarType;

    using FunctionBaseType = Plato::Hyperbolic::AbstractScalarFunction<EvaluationType>;
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<mSpaceDim>;

    std::shared_ptr<CubatureType> mCubatureRule;

    IndicatorFunctionType mIndicatorFunction; /*!< penalty function */
    Plato::ApplyWeighting<mSpaceDim, mNumVoigtTerms, IndicatorFunctionType> mApplyWeighting; /*!< apply penalty function */

    std::vector<std::string> mPlottable; /*!< database of output field names */

    Teuchos::RCP<Plato::LinearElasticMaterial<mSpaceDim>> mMaterialModel;


  public:
    /******************************************************************************//**
     * \brief Constructor
     * \param aSpatialDomain Plato Analyze spatial domain
     * \param aDataMap Plato Analyze data map
     * \param aProblemParams input database for overall problem
     * \param aFunctionParams input database for function
    **********************************************************************************/
    InternalElasticEnergy(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              Teuchos::ParameterList & aFunctionParams,
        const std::string            & aFunctionName
    ) :
        FunctionBaseType   (aSpatialDomain, aDataMap, aFunctionName),
        mCubatureRule      (std::make_shared<CubatureType>()),
        mIndicatorFunction (aFunctionParams.sublist("Penalty Function")),
        mApplyWeighting    (mIndicatorFunction)
    {
        Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aProblemParams);
        mMaterialModel = tMaterialModelFactory.create(aSpatialDomain.getMaterialName());

        if(aProblemParams.isType < Teuchos::Array < std::string >> ("Plottable"))
        {
            mPlottable = aProblemParams.get < Teuchos::Array < std::string >> ("Plottable").toVector();
        }
    }

    /******************************************************************************//**
     * \brief Evaluate internal elastic energy function
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>       & aState,
        const Plato::ScalarMultiVectorT <StateDotScalarType>    & aStateDot,
        const Plato::ScalarMultiVectorT <StateDotDotScalarType> & aStateDotDot,
        const Plato::ScalarMultiVectorT <ControlScalarType>     & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>      & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>      & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    {
        using SimplexPhysics = typename Plato::SimplexMechanics<mSpaceDim>;
        using StrainScalarType =
            typename Plato::fad_type_t<SimplexPhysics, StateScalarType, ConfigScalarType>;

        auto tNumCells = mSpatialDomain.numCells();

        Plato::Strain<mSpaceDim>                 tComputeVoigtStrain;
        Plato::ScalarProduct<mNumVoigtTerms>     tComputeScalarProduct;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::LinearStress<EvaluationType,
                            SimplexPhysics>      tComputeVoigtStress(mMaterialModel);

        Plato::ScalarVectorT      <ConfigScalarType> tCellVolume ("cell weight",tNumCells);
        Plato::ScalarArray3DT     <ConfigScalarType> tGradient   ("gradient",tNumCells,mNumNodesPerCell,mSpaceDim);
        Plato::ScalarMultiVectorT <StrainScalarType> tStrain     ("strain",tNumCells,mNumVoigtTerms);
        Plato::ScalarMultiVectorT <ResultScalarType> tStress     ("stress",tNumCells,mNumVoigtTerms);

        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tApplyWeighting   = mApplyWeighting;
        Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;

            // compute strain
            //
            tComputeVoigtStrain(aCellOrdinal, tStrain, aState, tGradient);

            // compute stress
            //
            tComputeVoigtStress(aCellOrdinal, tStress, tStrain);

            // apply weighting
            //
            tApplyWeighting(aCellOrdinal, tStress, aControl);

            // compute element internal energy (inner product of strain and weighted stress)
            //
            tComputeScalarProduct(aCellOrdinal, aResult, tStress, tStrain, tCellVolume);

        },"energy gradient");

        if( std::count(mPlottable.begin(),mPlottable.end(),"strain") )
            toMap(mDataMap, tStrain, "strain", mSpatialDomain);
        if( std::count(mPlottable.begin(),mPlottable.end(),"stress") )
            toMap(mDataMap, tStress, "stress", mSpatialDomain);
    }
}; // class InternalElasticEnergy

} // namespace Hyperbolic

} // namespace Plato
#endif

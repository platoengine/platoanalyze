#pragma once

#include <algorithm>
#include <memory>

#include "PlatoStaticsTypes.hpp"
#include "Simp.hpp"
#include "WorksetBase.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoMathHelpers.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ImplicitFunctors.hpp"
#include "AbstractLocalMeasure.hpp"
#include "BLAS1.hpp"


namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Volume integral criterion of field quantites (primarily for use with VolumeAverageCriterion)
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysicsT>
class VolumeIntegralCriterion :
        public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    static constexpr Plato::OrdinalType mNumVoigtTerms = SimplexPhysicsT::mNumVoigtTerms; /*!< number of Voigt terms */
    static constexpr Plato::OrdinalType mNumNodesPerCell = SimplexPhysicsT::mNumNodesPerCell; /*!< number of nodes per cell/element */

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain; /*!< mesh database */
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap; /*!< PLATO Engine output database */

    using StateT   = typename EvaluationType::StateScalarType;   /*!< state variables automatic differentiation type */
    using ConfigT  = typename EvaluationType::ConfigScalarType;  /*!< configuration variables automatic differentiation type */
    using ResultT  = typename EvaluationType::ResultScalarType;  /*!< result variables automatic differentiation type */
    using ControlT = typename EvaluationType::ControlScalarType; /*!< control variables automatic differentiation type */

    using Residual = typename Plato::ResidualTypes<SimplexPhysicsT>;

    using FunctionBaseType = Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    Plato::ScalarVector mSpatialWeights; /*!< spatially varying weights */

    Plato::Scalar mPenalty;        /*!< penalty parameter in SIMP model */
    Plato::Scalar mMinErsatzValue; /*!< minimum ersatz material value in SIMP model */

    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType, SimplexPhysicsT>> mLocalMeasure; /*!< Volume averaged quantity with evaluation type */

private:
    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams)
    {
        this->readInputs(aInputParams);
    }

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void readInputs(Teuchos::ParameterList & aInputParams)
    {
        Teuchos::ParameterList & tParams = aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
        auto tPenaltyParams = tParams.sublist("Penalty Function");
        std::string tPenaltyType = tPenaltyParams.get<std::string>("Type", "SIMP");
        if (tPenaltyType != "SIMP")
        {
            THROWERR("A penalty function type other than SIMP is not yet implemented for the VolumeIntegralCriterion.")
        }
        mPenalty        = tParams.get<Plato::Scalar>("Exponent", 3.0);
        mMinErsatzValue = tParams.get<Plato::Scalar>("Minimum Value", 1e-9);
    }

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aPlatoDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aFuncName user defined function name
     **********************************************************************************/
    VolumeIntegralCriterion(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFuncName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aFuncName),
        mPenalty(3),
        mMinErsatzValue(1.0e-9)
    {
        this->initialize(aInputParams);

        auto tNumCells = mSpatialDomain.numCells();
        Kokkos::resize(mSpatialWeights, tNumCells);
        Plato::blas1::fill(static_cast<Plato::Scalar>(1.0), mSpatialWeights);
printf("Warning: SIMP is still hard coded in the VolumeIntegralCriterion.hpp ");
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    VolumeIntegralCriterion(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, "Volume Integral Criterion"),
        mPenalty(3),
        mMinErsatzValue(0.0),
        mLocalMeasure(nullptr)
    {

    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~VolumeIntegralCriterion()
    {
    }

    /******************************************************************************//**
     * \brief Set volume integrated quanitity
     * \param [in] aInputEvaluationType evaluation type volume integrated quanitity
    **********************************************************************************/
    void setVolumeIntegratedQuantity(const std::shared_ptr<AbstractLocalMeasure<EvaluationType,SimplexPhysicsT>> & aInput)
    {
        mLocalMeasure = aInput;
    }

    /******************************************************************************//**
     * \brief Set spatial weights
     * \param [in] aInput scalar vector of spatial weights
    **********************************************************************************/
    void setSpatialWeights(Plato::ScalarVector & aInput) override
    {
        Kokkos::resize(mSpatialWeights, aInput.size());
        Plato::blas1::copy(aInput, mSpatialWeights);
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void
    updateProblem(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D     & aConfigWS
    ) override
    {
        // Perhaps update penalty exponent?
        WARNING("Penalty exponents not yet updated in VolumeIntegralCriterion.")
    }

    /******************************************************************************//**
     * \brief Evaluate volume average criterion
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void evaluate(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();

        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;

        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);

        // ****** COMPUTE VOLUME AVERAGED QUANTITIES AND STORE ON DEVICE ******
        Plato::ScalarVectorT<ResultT> tVolumeIntegratedQuantity("volume integrated quantity", tNumCells);
        (*mLocalMeasure)(aStateWS, aConfigWS, tVolumeIntegratedQuantity);
        
        // ****** ALLOCATE TEMPORARY ARRAYS ON DEVICE ******
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tConfigurationGradient("configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
        auto tQuadratureWeight = tCubatureRule.getCubWeight();
        auto tSpatialWeights  = mSpatialWeights;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tConfigurationGradient, aConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;

            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControlWS);
            ControlT tPenaltyFunctionValue = tSIMP(tDensity);
            
            aResultWS(aCellOrdinal) = tPenaltyFunctionValue * tVolumeIntegratedQuantity(aCellOrdinal) * tCellVolume(aCellOrdinal)
                                    * tSpatialWeights(aCellOrdinal);
        },"Compute Volume Integral Criterion");

    }

};
// class VolumeIntegralCriterion

}
//namespace Elliptic

}
//namespace Plato
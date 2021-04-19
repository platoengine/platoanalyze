#pragma once

#include <algorithm>
#include <memory>

#include "Simp.hpp"
#include "ToMap.hpp"
#include "BLAS1.hpp"
#include "WorksetBase.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoMathHelpers.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ExpInstMacros.hpp"
#include "AbstractLocalMeasure.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Augmented Lagrangian local constraint criterion tailored for general problems
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysicsT>
class AugLagStressCriterionQuadratic :
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

    Plato::Scalar mPenalty; /*!< penalty parameter in SIMP model */
    Plato::Scalar mLocalMeasureLimit; /*!< local measure limit/upper bound */
    Plato::Scalar mAugLagPenalty; /*!< augmented Lagrangian penalty */
    Plato::Scalar mMinErsatzValue; /*!< minimum ersatz material value in SIMP model */
    Plato::Scalar mAugLagPenaltyUpperBound; /*!< upper bound on augmented Lagrangian penalty */
    Plato::Scalar mInitialLagrangeMultipliersValue; /*!< initial value for Lagrange multipliers */
    Plato::Scalar mAugLagPenaltyExpansionMultiplier; /*!< expansion parameter for augmented Lagrangian penalty */

    Plato::ScalarVector mLagrangeMultipliers; /*!< Lagrange multipliers */

    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType,SimplexPhysicsT>> mLocalMeasureEvaluationType; /*!< Local measure with evaluation type */
    std::shared_ptr<Plato::AbstractLocalMeasure<Residual,SimplexPhysicsT>>       mLocalMeasurePODType; /*!< Local measure with POD type */

private:
    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams)
    {
        this->readInputs(aInputParams);

        Plato::blas1::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void readInputs(Teuchos::ParameterList & aInputParams)
    {
        Teuchos::ParameterList & tParams = aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
        mPenalty = tParams.get<Plato::Scalar>("SIMP penalty", 3.0);
        mLocalMeasureLimit = tParams.get<Plato::Scalar>("Local Measure Limit", 1.0);
        mAugLagPenalty = tParams.get<Plato::Scalar>("Initial Penalty", 0.25);
        mMinErsatzValue = tParams.get<Plato::Scalar>("Min. Ersatz Material", 1e-9);
        mAugLagPenaltyUpperBound = tParams.get<Plato::Scalar>("Penalty Upper Bound", 500.0);
        mInitialLagrangeMultipliersValue = tParams.get<Plato::Scalar>("Initial Lagrange Multiplier", 0.01);
        mAugLagPenaltyExpansionMultiplier = tParams.get<Plato::Scalar>("Penalty Expansion Multiplier", 1.5);
    }

    /******************************************************************************//**
     * \brief Update Augmented Lagrangian penalty
    **********************************************************************************/
    void updateAugLagPenaltyMultipliers()
    {
        mAugLagPenalty = mAugLagPenaltyExpansionMultiplier * mAugLagPenalty;
        mAugLagPenalty = std::min(mAugLagPenalty, mAugLagPenaltyUpperBound);
    }

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aPlatoDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aFuncName user defined function name
     **********************************************************************************/
    AugLagStressCriterionQuadratic(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFuncName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aFuncName),
        mPenalty(3),
        mLocalMeasureLimit(1),
        mAugLagPenalty(0.1),
        mMinErsatzValue(0.0),
        mAugLagPenaltyUpperBound(100),
        mInitialLagrangeMultipliersValue(0.01),
        mAugLagPenaltyExpansionMultiplier(1.05),
        mLagrangeMultipliers("Lagrange Multipliers", aSpatialDomain.Mesh.nelems())
    {
        this->initialize(aInputParams);
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    AugLagStressCriterionQuadratic(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, "Local Constraint Quadratic"),
        mPenalty(3),
        mLocalMeasureLimit(1),
        mAugLagPenalty(0.1),
        mMinErsatzValue(0.0),
        mAugLagPenaltyUpperBound(100),
        mInitialLagrangeMultipliersValue(0.01),
        mAugLagPenaltyExpansionMultiplier(1.05),
        mLagrangeMultipliers("Lagrange Multipliers", aSpatialDomain.Mesh.nelems()),
        mLocalMeasureEvaluationType(nullptr),
        mLocalMeasurePODType(nullptr)
    {
        Plato::blas1::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~AugLagStressCriterionQuadratic()
    {
    }

    /******************************************************************************//**
     * \brief Return augmented Lagrangian penalty multiplier
     * \return augmented Lagrangian penalty multiplier
    **********************************************************************************/
    Plato::Scalar getAugLagPenalty() const
    {
        return (mAugLagPenalty);
    }

    /******************************************************************************//**
     * \brief Return Lagrange multipliers
     * \return 1D view of Lagrange multipliers
    **********************************************************************************/
    Plato::ScalarVector getLagrangeMultipliers() const
    {
        return (mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * \brief Set local measure function
     * \param [in] aInputEvaluationType evaluation type local measure
     * \param [in] aInputPODType pod type local measure
    **********************************************************************************/
    void setLocalMeasure(const std::shared_ptr<AbstractLocalMeasure<EvaluationType,SimplexPhysicsT>> & aInputEvaluationType,
                         const std::shared_ptr<AbstractLocalMeasure<Residual,SimplexPhysicsT>> & aInputPODType)
    {
        mLocalMeasureEvaluationType = aInputEvaluationType;
        mLocalMeasurePODType        = aInputPODType;
    }

    /******************************************************************************//**
     * \brief Set local constraint limit/upper bound
     * \param [in] aInput local constraint limit
    **********************************************************************************/
    void setLocalMeasureValueLimit(const Plato::Scalar & aInput)
    {
        mLocalMeasureLimit = aInput;
    }

    /******************************************************************************//**
     * \brief Set augmented Lagrangian function penalty multiplier
     * \param [in] aInput penalty multiplier
     **********************************************************************************/
    void setAugLagPenalty(const Plato::Scalar & aInput)
    {
        mAugLagPenalty = aInput;
    }

    /******************************************************************************//**
     * \brief Set Lagrange multipliers
     * \param [in] aInput Lagrange multipliers
     **********************************************************************************/
    void setLagrangeMultipliers(const Plato::ScalarVector & aInput)
    {
        assert(aInput.size() == mLagrangeMultipliers.size());
        Plato::blas1::copy(aInput, mLagrangeMultipliers);
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
        this->updateLagrangeMultipliers(aStateWS, aControlWS, aConfigWS);
        this->updateAugLagPenaltyMultipliers();
    }

    /******************************************************************************//**
     * \brief Evaluate augmented Lagrangian local constraint criterion
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

        using StrainT = typename Plato::fad_type_t<SimplexPhysicsT, StateT, ConfigT>;

        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);

        // ****** COMPUTE LOCAL MEASURE VALUES AND STORE ON DEVICE ******
        Plato::ScalarVectorT<ResultT> tLocalMeasureValue("local measure value", tNumCells);
        (*mLocalMeasureEvaluationType)(aStateWS, aConfigWS, tLocalMeasureValue);
        
        // ****** ALLOCATE TEMPORARY ARRAYS ON DEVICE ******
        Plato::ScalarVectorT<ResultT> tConstraintValue("constraint", tNumCells);
        Plato::ScalarVectorT<ResultT> tTrialConstraintValue("trial constraint", tNumCells);
        Plato::ScalarVectorT<ResultT> tTrueConstraintValue("true constraint", tNumCells);
        
        Plato::ScalarVectorT<ResultT> tLocalMeasureValueOverLimit("local measure over limit", tNumCells);
        Plato::ScalarVectorT<ResultT> tLocalMeasureValueOverLimitMinusOne("local measure over limit minus one", tNumCells);
        Plato::ScalarVectorT<ResultT> tOutputPenalizedLocalMeasure("output penalized local measure", tNumCells);

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tLocalMeasureValueLimit = mLocalMeasureLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tLagrangeMultipliers = mLagrangeMultipliers;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
        auto tBasisFunc = tCubatureRule.getBasisFunctions();
        Plato::Scalar tLagrangianMultiplier = static_cast<Plato::Scalar>(1.0 / tNumCells);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tLocalMeasureValueOverLimit(aCellOrdinal) = tLocalMeasureValue(aCellOrdinal) / tLocalMeasureValueLimit;
            tLocalMeasureValueOverLimitMinusOne(aCellOrdinal) = tLocalMeasureValueOverLimit(aCellOrdinal) - static_cast<Plato::Scalar>(1.0);
            tConstraintValue(aCellOrdinal) = ( //pow(tLocalMeasureValueOverLimitMinusOne(aCellOrdinal), 4) +
                                               pow(tLocalMeasureValueOverLimitMinusOne(aCellOrdinal), 2) );

            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControlWS);
            ControlT tMaterialPenalty = tSIMP(tDensity);
            tOutputPenalizedLocalMeasure(aCellOrdinal) = tMaterialPenalty * tLocalMeasureValue(aCellOrdinal);
            tTrialConstraintValue(aCellOrdinal) = tMaterialPenalty * tConstraintValue(aCellOrdinal);
            tTrueConstraintValue(aCellOrdinal) = tLocalMeasureValueOverLimit(aCellOrdinal) > static_cast<ResultT>(1.0) ?
                                                     tTrialConstraintValue(aCellOrdinal) : static_cast<ResultT>(0.0);

            // Compute constraint contribution to augmented Lagrangian function
            aResultWS(aCellOrdinal) = tLagrangianMultiplier * ( ( tLagrangeMultipliers(aCellOrdinal) *
                    tTrueConstraintValue(aCellOrdinal) ) + ( static_cast<Plato::Scalar>(0.5) * tAugLagPenalty *
                    tTrueConstraintValue(aCellOrdinal) * tTrueConstraintValue(aCellOrdinal) ) );
        },"Compute Quadratic Augmented Lagrangian Function Without Objective");

         Plato::toMap(mDataMap, tOutputPenalizedLocalMeasure, mLocalMeasureEvaluationType->getName(), mSpatialDomain);
    }

    /******************************************************************************//**
     * \brief Update Lagrange multipliers
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void
    updateLagrangeMultipliers(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D     & aConfigWS
    )
    {
        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();

        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);

        // ****** COMPUTE LOCAL MEASURE VALUES AND STORE ON DEVICE ******
        Plato::ScalarVector tLocalMeasureValue("local measure value", tNumCells);
        (*mLocalMeasurePODType)(aStateWS, aConfigWS, tLocalMeasureValue);
        
        // ****** ALLOCATE TEMPORARY ARRAYS ON DEVICE ******
        Plato::ScalarVector tConstraintValue("constraint residual", tNumCells);
        Plato::ScalarVector tTrialConstraintValue("trial constraint", tNumCells);
        Plato::ScalarVector tTrueConstraintValue("true constraint", tNumCells);
        
        Plato::ScalarVector tLocalMeasureValueOverLimit("local measure over limit", tNumCells);
        Plato::ScalarVector tLocalMeasureValueOverLimitMinusOne("local measure over limit minus one", tNumCells);

        Plato::ScalarVector tTrialMultiplier("trial multiplier", tNumCells);

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tLocalMeasureValueLimit = mLocalMeasureLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tLagrangeMultipliers = mLagrangeMultipliers;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
        auto tCubWeight = tCubatureRule.getCubWeight();
        auto tBasisFunc = tCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            // Compute local constraint residual
            tLocalMeasureValueOverLimit(aCellOrdinal) = tLocalMeasureValue(aCellOrdinal) / tLocalMeasureValueLimit;
            tLocalMeasureValueOverLimitMinusOne(aCellOrdinal) = tLocalMeasureValueOverLimit(aCellOrdinal) - static_cast<Plato::Scalar>(1.0);
            tConstraintValue(aCellOrdinal) = ( //pow(tLocalMeasureValueOverLimitMinusOne(aCellOrdinal), 4) +
                                               pow(tLocalMeasureValueOverLimitMinusOne(aCellOrdinal), 2) );

            Plato::Scalar tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControlWS);
            Plato::Scalar tMaterialPenalty = tSIMP(tDensity);
            tTrialConstraintValue(aCellOrdinal) = tMaterialPenalty * tConstraintValue(aCellOrdinal);
            tTrueConstraintValue(aCellOrdinal) = tLocalMeasureValueOverLimit(aCellOrdinal) > static_cast<Plato::Scalar>(1.0) ?
                                                       tTrialConstraintValue(aCellOrdinal) : static_cast<Plato::Scalar>(0.0);

            // Compute Lagrange multiplier
            tTrialMultiplier(aCellOrdinal) = tLagrangeMultipliers(aCellOrdinal) + 
                                           ( tAugLagPenalty * tTrueConstraintValue(aCellOrdinal) );
            tLagrangeMultipliers(aCellOrdinal) = (tTrialMultiplier(aCellOrdinal) < static_cast<Plato::Scalar>(0.0)) ?
                                                 static_cast<Plato::Scalar>(0.0) : tTrialMultiplier(aCellOrdinal);
        },"Update Multipliers");
    }
};
// class AugLagStressCriterionQuadratic

}
//namespace Plato
#include "SimplexMechanics.hpp"
#include "SimplexThermomechanics.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC2(Plato::AugLagStressCriterionQuadratic, Plato::SimplexMechanics, 1)
PLATO_EXPL_DEC2(Plato::AugLagStressCriterionQuadratic, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC2(Plato::AugLagStressCriterionQuadratic, Plato::SimplexMechanics, 2)
PLATO_EXPL_DEC2(Plato::AugLagStressCriterionQuadratic, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC2(Plato::AugLagStressCriterionQuadratic, Plato::SimplexMechanics, 3)
PLATO_EXPL_DEC2(Plato::AugLagStressCriterionQuadratic, Plato::SimplexThermomechanics, 3)
#endif

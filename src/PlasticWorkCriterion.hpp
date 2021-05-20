/*
 * PlasticWorkCriterion.hpp
 *
 *  Created on: Feb 29, 2020
 */

#pragma once

#include "Simp.hpp"
#include "Strain.hpp"
#include "SimplexFadTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "SimplexPlasticity.hpp"
#include "SimplexThermoPlasticity.hpp"
#include "ComputeStabilizedCauchyStress.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "J2PlasticityUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ThermoPlasticityUtilities.hpp"
#include "IsotropicMaterialUtilities.hpp"
#include "DoubleDotProduct2ndOrderTensor.hpp"
#include "AbstractLocalScalarFunctionInc.hpp"
#include "InterpolateFromNodal.hpp"
#include "ComputeStabilizedCauchyStress.hpp"
#include "ComputeDeviatoricStress.hpp"
#include "TimeData.hpp"
#include "ExpInstMacros.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Evaluate the plastic work criterion using a trapezoid rule.  The plastic
 *   work criterion is given by:
 *
 *  \f$ f(\phi,u_{k},u_{k-1},c_{k},c_{k-1}) = \frac{1}{2}\int_{\Omega}\sigma_{k} :
 *       \left( \epsilon_{k}^{p} - \epsilon_{k-1}^{p} \right) d\Omega  \f$
 *
 * where \f$ \phi \f$ are the control variables, \f$ u \f$ are the global state
 * variables, \f$ c \f$ are the local state variables, \f$ \epsilon^p \f$ is the
 * plastic strain tensor, and \f$ \sigma \f$ is the Cauchy stress tensor.  The
 * \f$ k-th \f$ index denotes the time stpe index.
 *
 * \tparam EvaluationType      evaluation type for scalar function, determines
 *                             which AD type is active
 * \tparam SimplexPhysicsType  simplex physics type, determines values of
 *                             physics-based static parameters
*******************************************************************************/
template<typename EvaluationType, typename SimplexPhysicsType>
class PlasticWorkCriterion : public Plato::AbstractLocalScalarFunctionInc<EvaluationType>
{
// private member data
private:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    static constexpr auto mNumStressTerms       = SimplexPhysicsType::mNumStressTerms;  /*!< number of stress/strain components */
    static constexpr auto mNumNodesPerCell      = SimplexPhysicsType::mNumNodesPerCell; /*!< number nodes per cell */
    static constexpr auto mNumGlobalDofsPerNode = SimplexPhysicsType::mNumDofsPerNode;  /*!< number global degrees of freedom per node */
    static constexpr auto mPressureDofOffset = SimplexPhysicsType::mPressureDofOffset;  /*!< pressure dofs offset */

    using ResultT = typename EvaluationType::ResultScalarType;                 /*!< result variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType;                 /*!< config variables automatic differentiation type */
    using ControlT = typename EvaluationType::ControlScalarType;               /*!< control variables automatic differentiation type */
    using LocalStateT = typename EvaluationType::LocalStateScalarType;         /*!< local state variables automatic differentiation type */
    using GlobalStateT = typename EvaluationType::StateScalarType;             /*!< global state variables automatic differentiation type */
    using PrevLocalStateT = typename EvaluationType::PrevLocalStateScalarType; /*!< local state variables automatic differentiation type */
    using PrevGlobalStateT = typename EvaluationType::PrevStateScalarType;     /*!< global state variables automatic differentiation type */

    using FunctionBaseType = Plato::AbstractLocalScalarFunctionInc<EvaluationType>;
    using Plato::AbstractLocalScalarFunctionInc<EvaluationType>::mSpatialDomain;

    Plato::Scalar mBulkModulus;              /*!< elastic bulk modulus */
    Plato::Scalar mShearModulus;             /*!< elastic shear modulus */

    Plato::Scalar mPressureScaling;             /*!< pressure scaling */

    Plato::Scalar mThermalExpansionCoefficient;    /*!< Thermal Expansivity */
    Plato::Scalar mReferenceTemperature;           /*!< thermal reference temperature */
    Plato::Scalar mTemperatureScaling;             /*!< temperature scaling */

    Plato::Scalar mPenaltySIMP;                /*!< SIMP penalty for elastic properties */
    Plato::Scalar mMinErsatz;                  /*!< SIMP min ersatz stiffness for elastic properties */
    Plato::Scalar mUpperBoundOnPenaltySIMP;    /*!< continuation parameter: upper bound on SIMP penalty for elastic properties */
    Plato::Scalar mAdditiveContinuationParam;  /*!< continuation parameter: multiplier on SIMP penalty for elastic properties */

    Plato::LinearTetCubRuleDegreeOne<mSpaceDim> mCubatureRule;  /*!< simplex linear cubature rule */

// public access functions
public:
    /***************************************************************************//**
     * \brief Constructor for plastic work criterion
     *
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap     PLATO Analyze output data map side sets database
     * \param [in] aInputParams input parameters from XML file
     * \param [in] aName        scalar function name
    *******************************************************************************/
    PlasticWorkCriterion(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aName),
        mBulkModulus(-1.0),
        mShearModulus(-1.0),
        mThermalExpansionCoefficient(0.0),
        mReferenceTemperature(0.0),
        mPressureScaling(1.0),
        mTemperatureScaling(1.0),
        mPenaltySIMP(3),
        mMinErsatz(1e-9),
        mUpperBoundOnPenaltySIMP(4),
        mAdditiveContinuationParam(0.1),
        mCubatureRule()
    {
        this->parsePenaltyModelParams(aInputParams);
        this->parseMaterialProperties(aInputParams);
    }

    /***************************************************************************//**
     * \brief Constructor of maximize total work criterion
     *
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap     PLATO Analyze output data map side sets database
     * \param [in] aName        scalar function name
    *******************************************************************************/
    PlasticWorkCriterion(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap,
              std::string            aName = ""
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aName),
        mBulkModulus(1.0),
        mShearModulus(1.0),
        mThermalExpansionCoefficient(0.0),
        mReferenceTemperature(0.0),
        mPenaltySIMP(3),
        mMinErsatz(1e-9),
        mUpperBoundOnPenaltySIMP(4),
        mAdditiveContinuationParam(0.1),
        mCubatureRule()
    {
    }

    /***************************************************************************//**
     * \brief Destructor of maximize total work criterion
    *******************************************************************************/
    virtual ~PlasticWorkCriterion(){}

    /***************************************************************************//**
     * \brief Evaluates plastic work criterion. FAD type determines output/result value.
     *
     * \param [in] aCurrentGlobalState  current global states
     * \param [in] aPreviousGlobalState previous global states
     * \param [in] aCurrentLocalState   current local states
     * \param [in] aPreviousLocalState  previous global states
     * \param [in] aControls            control variables
     * \param [in] aConfig              configuration variables
     * \param [in] aResult              output container
     * \param [in] aTimeData            time data object
    *******************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<GlobalStateT> &aCurrentGlobalState,
                  const Plato::ScalarMultiVectorT<PrevGlobalStateT> &aPreviousGlobalState,
                  const Plato::ScalarMultiVectorT<LocalStateT> &aCurrentLocalState,
                  const Plato::ScalarMultiVectorT<PrevLocalStateT> &aPreviousLocalState,
                  const Plato::ScalarMultiVectorT<ControlT> &aControls,
                  const Plato::ScalarArray3DT<ConfigT> &aConfig,
                  const Plato::ScalarVectorT<ResultT> &aResult,
                  const Plato::TimeData &aTimeData)
    {
        using TotalStrainT   = typename Plato::fad_type_t<SimplexPhysicsType, GlobalStateT, ConfigT>;
        using ElasticStrainT = typename Plato::fad_type_t<SimplexPhysicsType, LocalStateT, ConfigT, GlobalStateT>;

        using PreviousTotalStrainT   = typename Plato::fad_type_t<SimplexPhysicsType, PrevGlobalStateT, ConfigT>;
        using PreviousElasticStrainT = typename Plato::fad_type_t<SimplexPhysicsType, PrevLocalStateT, ConfigT, PrevGlobalStateT>;

        // allocate functors used to evaluate criterion
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::ComputeDeviatoricStress<mSpaceDim> tComputeDeviatoricStress;
        Plato::ComputeStabilizedCauchyStress<mSpaceDim> tComputeCauchyStress;
        Plato::InterpolateFromNodal<mSpaceDim, mNumGlobalDofsPerNode, mPressureDofOffset> tInterpolatePressureFromNodal;
        Plato::J2PlasticityUtilities<mSpaceDim>  tJ2PlasticityUtils;
        Plato::Strain<mSpaceDim, mNumGlobalDofsPerNode> tComputeTotalStrain;
        Plato::ThermoPlasticityUtilities<mSpaceDim, SimplexPhysicsType> tThermoPlasticityUtils(mThermalExpansionCoefficient, mReferenceTemperature,
                                                                                               mTemperatureScaling);
        Plato::MSIMP tPenaltyFunction(mPenaltySIMP, mMinErsatz);

        // allocate local containers used to evaluate criterion
        auto tNumCells = mSpatialDomain.numCells();

        Plato::ScalarMultiVectorT<PreviousTotalStrainT>   tPreviousTotalStrain("previous total strain",tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<PreviousElasticStrainT> tPreviousElasticStrain("previous elastic strain", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ResultT>                tPreviousCauchyStress("previous cauchy stress", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ResultT>                tPreviousDeviatoricStress("previous deviatoric stress", tNumCells, mNumStressTerms);
        Plato::ScalarVectorT<PrevGlobalStateT>            tPreviousPressure("previous pressure", tNumCells);

        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarMultiVectorT<ResultT> tCurrentCauchyStress("current cauchy stress", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ResultT> tPlasticStrainMisfit("plastic strain misfit", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<TotalStrainT> tCurrentTotalStrain("current total strain",tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ElasticStrainT> tCurrentElasticStrain("current elastic strain", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ResultT>        tCurrentDeviatoricStress("current deviatoric stress", tNumCells, mNumStressTerms);
        Plato::ScalarVectorT<GlobalStateT>        tCurrentPressure("current pressure", tNumCells);

        Plato::ScalarArray3DT<ConfigT> tConfigurationGradient("configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        Plato::ScalarMultiVectorT<ResultT> tAverageCauchyStress("average cauchy stress", tNumCells, mNumStressTerms);

        auto tNumStressTerms = mNumStressTerms;
        auto tPressureScaling = mPressureScaling;
        auto tElasticShearModulus = mShearModulus;

        constexpr Plato::Scalar tOneHalf = 0.5;

        auto tQuadratureWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
        {
            // compute configuration gradients
            tComputeGradient(aCellOrdinal, tConfigurationGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;

            tInterpolatePressureFromNodal(aCellOrdinal, tBasisFunctions, aPreviousGlobalState, tPreviousPressure);
            tInterpolatePressureFromNodal(aCellOrdinal, tBasisFunctions, aCurrentGlobalState, tCurrentPressure);
            tPreviousPressure(aCellOrdinal) *= tPressureScaling;
            tCurrentPressure(aCellOrdinal) *= tPressureScaling;

            // compute plastic strain misfit
            tJ2PlasticityUtils.computePlasticStrainMisfit(aCellOrdinal, aCurrentLocalState, aPreviousLocalState, tPlasticStrainMisfit);

            // compute previous elastic strain
            tComputeTotalStrain(aCellOrdinal, tPreviousTotalStrain, aPreviousGlobalState, tConfigurationGradient);
            tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aPreviousGlobalState, aPreviousLocalState,
                                                        tBasisFunctions, tPreviousTotalStrain, tPreviousElasticStrain);

            // compute current elastic strain
            tComputeTotalStrain(aCellOrdinal, tCurrentTotalStrain, aCurrentGlobalState, tConfigurationGradient);
            tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aCurrentGlobalState, aCurrentLocalState,
                                                        tBasisFunctions, tCurrentTotalStrain, tCurrentElasticStrain);

            // compute cell penalty and penalized elastic properties
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControls);
            ControlT tElasticPropertiesPenalty = tPenaltyFunction(tDensity);
            ControlT tPenalizedShearModulus = tElasticPropertiesPenalty * tElasticShearModulus;

            tComputeDeviatoricStress(aCellOrdinal, tPenalizedShearModulus, tPreviousElasticStrain, tPreviousDeviatoricStress);
            tComputeDeviatoricStress(aCellOrdinal, tPenalizedShearModulus, tCurrentElasticStrain, tCurrentDeviatoricStress);

            tComputeCauchyStress(aCellOrdinal, tPreviousPressure, tPreviousDeviatoricStress, tPreviousCauchyStress);
            tComputeCauchyStress(aCellOrdinal, tCurrentPressure, tCurrentDeviatoricStress, tCurrentCauchyStress);

            for (Plato::OrdinalType tIndex = 0; tIndex < tNumStressTerms; ++tIndex)
                tAverageCauchyStress(aCellOrdinal, tIndex) = tOneHalf * 
                                                             (tCurrentCauchyStress(aCellOrdinal, tIndex) + tPreviousCauchyStress(aCellOrdinal, tIndex));

            // compute double dot product (strain tensor shear terms already have factor of 2)
            aResult(aCellOrdinal) = 0.0;
            for (Plato::OrdinalType tIndex = 0; tIndex < tNumStressTerms; ++tIndex)
                aResult(aCellOrdinal) += tAverageCauchyStress(aCellOrdinal, tIndex) * tPlasticStrainMisfit(aCellOrdinal, tIndex);
            aResult(aCellOrdinal) *= tCellVolume(aCellOrdinal);
        }, "plastic work criterion");
    }

    /******************************************************************************//**
     * \brief Update physics-based data within a frequency of optimization iterations
     * \param [in] aGlobalState global state variables
     * \param [in] aLocalState  local state variables
     * \param [in] aControl     control variables, e.g. design variables
     * \param [in] aTimeData    time data object
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aGlobalState,
                       const Plato::ScalarMultiVector & aLocalState,
                       const Plato::ScalarVector & aControl,
                       const Plato::TimeData & aTimeData) override
    {
        // update SIMP penalty parameter
        auto tPreviousPenaltySIMP = mPenaltySIMP;
        auto tSuggestedPenaltySIMP = tPreviousPenaltySIMP + mAdditiveContinuationParam;
        mPenaltySIMP = tSuggestedPenaltySIMP >= mUpperBoundOnPenaltySIMP ? mUpperBoundOnPenaltySIMP : tSuggestedPenaltySIMP;
        std::ostringstream tMsg;
        tMsg << "Plastic Work Criterion: New penalty parameter is set to '" << mPenaltySIMP
                << "'. Previous penalty parameter was '" << tPreviousPenaltySIMP << "'.\n";
        REPORT(tMsg.str().c_str())
    }

private:
    /**********************************************************************//**
     * \brief Parse elastic material properties
     * \param [in] aInputParams input XML data, i.e. parameter list
    **************************************************************************/
    void parsePenaltyModelParams(Teuchos::ParameterList &aInputParams)
    {
        auto tFunctionName = this->getName();
        if(aInputParams.sublist("Criteria").isSublist(tFunctionName) == true)
        {
            Teuchos::ParameterList tFunctionParams = aInputParams.sublist("Criteria").sublist(tFunctionName);
            Teuchos::ParameterList tInputData = tFunctionParams.sublist("Penalty Function");
            mPenaltySIMP = tInputData.get<Plato::Scalar>("Exponent", 3.0);
            mMinErsatz = tInputData.get<Plato::Scalar>("Minimum Value", 1e-9);
            mAdditiveContinuationParam = tInputData.get<Plato::Scalar>("Additive Continuation", 0.1);
            mUpperBoundOnPenaltySIMP = tInputData.get<Plato::Scalar>("Penalty Exponent Upper Bound", 4.0);
        }
        else
        {
            const auto tError = std::string("UNKNOWN USER DEFINED SCALAR FUNCTION SUBLIST '")
                    + tFunctionName + "'. USER DEFINED SCALAR FUNCTION SUBLIST '" + tFunctionName
                    + "' IS NOT DEFINED IN THE INPUT FILE.";
            THROWERR(tError)
        }
    }

    /**********************************************************************//**
     * \brief Parse elastic material properties
     * \param [in] aProblemParams input XML data, i.e. parameter list
    **************************************************************************/
    void parseMaterialProperties(Teuchos::ParameterList &aProblemParams)
    {
        if(aProblemParams.isSublist("Material Models"))
        {
            auto tMaterialName = mSpatialDomain.getMaterialName();
            Teuchos::ParameterList tMaterialsList = aProblemParams.sublist("Material Models");
            mPressureScaling    = tMaterialsList.get<Plato::Scalar>("Pressure Scaling", 1.0);
            mTemperatureScaling = tMaterialsList.get<Plato::Scalar>("Temperature Scaling", 1.0);
            Teuchos::ParameterList tMaterialList  = tMaterialsList.sublist(tMaterialName);
            this->parseIsotropicMaterialProperties(tMaterialList);
        }
        else
        {
            THROWERR("'Material Models' SUBLIST IS NOT DEFINED.")
        }
    }

    /**********************************************************************//**
     * \brief Parse isotropic material properties
     * \param [in] aProblemParams input XML data, i.e. parameter list
    **************************************************************************/
    void parseIsotropicMaterialProperties(Teuchos::ParameterList &aMaterialParams)
    {
        if (aMaterialParams.isSublist("Isotropic Linear Elastic"))
        {
            auto tElasticSubList = aMaterialParams.sublist("Isotropic Linear Elastic");
            auto tPoissonsRatio = Plato::parse_poissons_ratio(tElasticSubList);
            auto tElasticModulus = Plato::parse_elastic_modulus(tElasticSubList);
            mBulkModulus = Plato::compute_bulk_modulus(tElasticModulus, tPoissonsRatio);
            mShearModulus = Plato::compute_shear_modulus(tElasticModulus, tPoissonsRatio);
        }
        else if (aMaterialParams.isSublist("Isotropic Linear Thermoelastic"))
        {
            auto tThermoelasticSubList = aMaterialParams.sublist("Isotropic Linear Thermoelastic");
            auto tPoissonsRatio = Plato::parse_poissons_ratio(tThermoelasticSubList);
            auto tElasticModulus = Plato::parse_elastic_modulus(tThermoelasticSubList);
            mBulkModulus = Plato::compute_bulk_modulus(tElasticModulus, tPoissonsRatio);
            mShearModulus = Plato::compute_shear_modulus(tElasticModulus, tPoissonsRatio);

            mThermalExpansionCoefficient = tThermoelasticSubList.get<Plato::Scalar>("Thermal Expansivity");
            mReferenceTemperature        = tThermoelasticSubList.get<Plato::Scalar>("Reference Temperature");
        }
        else
        {
            THROWERR("'Isotropic Linear Elastic' or 'Isotropic Linear Thermoelastic' sublist of 'Material Model' is not defined.")
        }
    }
};
// class PlasticWorkCriterion

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_INC_VMS(Plato::PlasticWorkCriterion, Plato::SimplexPlasticity, 2)
PLATO_EXPL_DEC_INC_VMS(Plato::PlasticWorkCriterion, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_INC_VMS(Plato::PlasticWorkCriterion, Plato::SimplexPlasticity, 3)
PLATO_EXPL_DEC_INC_VMS(Plato::PlasticWorkCriterion, Plato::SimplexThermoPlasticity, 3)
#endif

}
// namespace Plato

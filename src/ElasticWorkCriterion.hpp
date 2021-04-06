/*
 * ElasticWorkCriterion.hpp
 *
 *  Created on: Mar 7, 2020
 */

#pragma once

#include "Simp.hpp"
#include "Strain.hpp"
#include "SimplexFadTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "SimplexPlasticity.hpp"
#include "SimplexThermoPlasticity.hpp"
#include "ComputeElasticWork.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "ComputeStabilizedCauchyStress.hpp"
#include "ThermoPlasticityUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "IsotropicMaterialUtilities.hpp"
#include "DoubleDotProduct2ndOrderTensor.hpp"
#include "AbstractLocalScalarFunctionInc.hpp"
#include "ComputeDeviatoricStress.hpp"
#include "TimeData.hpp"
#include "ExpInstMacros.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Evaluate the elastic work criterion. The elastic work criterion is given by:
 *
 *  \f$ f(\phi,u_{n},c_{n}) =
 *          \mu\epsilon_{ij}^{d}\epsilon_{ij}^{d} + \kappa\epsilon_{kk}^{2}  \f$
 *
 * where \f$ \phi \f$ are the control variables, \f$ u \f$ are the global state
 * variables, \f$ c \f$ are the local state variables, \f$ \mu \f$ is the shear
 * modulus, \f$ \epsilon_{ij}^d \f$ is the deviatoric strain tensor, \f$ \kappa \f$
 * is the bulk modulus, and \f$ \epsilon_{kk} \f$ is the volumetric strain.  The
 * \f$ n-th \f$ index denotes the time stpe index and \f$ \{i,j,k\}\in(0,D) \f$,
 * where \f$ D \f$ is the spatial dimension.
 *
 * \tparam EvaluationType      evaluation type for scalar function, determines
 *                             which AD type is active
 * \tparam SimplexPhysicsType  simplex physics type, determines values of
 *                             physics-based static parameters
*******************************************************************************/
template<typename EvaluationType, typename SimplexPhysicsType>
class ElasticWorkCriterion : public Plato::AbstractLocalScalarFunctionInc<EvaluationType>
{
// private member data
private:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    static constexpr auto mNumStressTerms = SimplexPhysicsType::mNumStressTerms;        /*!< number of stress/strain components */
    static constexpr auto mNumNodesPerCell = SimplexPhysicsType::mNumNodesPerCell;      /*!< number nodes per cell */
    static constexpr auto mNumGlobalDofsPerNode = SimplexPhysicsType::mNumDofsPerNode;  /*!< number global degrees of freedom per node */
    static constexpr auto mPressureDofOffset = SimplexPhysicsType::mPressureDofOffset;  /*!< pressure dofs offset */

    using ResultT = typename EvaluationType::ResultScalarType;                     /*!< result variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType;                     /*!< config variables automatic differentiation type */
    using ControlT = typename EvaluationType::ControlScalarType;                   /*!< control variables automatic differentiation type */
    using LocalStateT = typename EvaluationType::LocalStateScalarType;             /*!< local state variables automatic differentiation type */
    using GlobalStateT = typename EvaluationType::StateScalarType;                 /*!< global state variables automatic differentiation type */
    using PrevLocalStateT = typename EvaluationType::PrevLocalStateScalarType;     /*!< local state variables automatic differentiation type */
    using PrevGlobalStateT = typename EvaluationType::PrevStateScalarType;         /*!< global state variables automatic differentiation type */

    using FunctionBaseType = Plato::AbstractLocalScalarFunctionInc<EvaluationType>;
    using Plato::AbstractLocalScalarFunctionInc<EvaluationType>::mSpatialDomain;

    Plato::Scalar mBulkModulus;              /*!< elastic bulk modulus */
    Plato::Scalar mShearModulus;             /*!< elastic shear modulus */

    Plato::Scalar mPressureScaling;             /*!< pressure scaling */

    Plato::Scalar mThermalExpansionCoefficient;    /*!< thermal expansion coefficient */
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
     * \brief Constructor for elastic work criterion
     *
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap     PLATO Analyze output data map side sets database
     * \param [in] aInputParams input parameters from XML file
     * \param [in] aName        scalar function name
    *******************************************************************************/
    ElasticWorkCriterion(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aName),
        mBulkModulus(-1.0),
        mShearModulus(-1.0),
        mPressureScaling(1.0),
        mThermalExpansionCoefficient(0.0),
        mReferenceTemperature(0.0),
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
     * \brief Constructor for elastic work criterion
     *
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap     PLATO Analyze output data map side sets database
     * \param [in] aName        scalar function name
    *******************************************************************************/
    ElasticWorkCriterion(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap,
              std::string            aName = ""
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aName),
        mBulkModulus(1.0),
        mShearModulus(1.0),
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
    virtual ~ElasticWorkCriterion(){}


    /***************************************************************************//**
     * \brief Evaluates elastic work criterion. FAD type determines output/result value.
     *
     * \param [in] aCurrentGlobalState  current global states
     * \param [in] aPreviousGlobalState previous global states
     * \param [in] aCurrentLocalState   current local states
     * \param [in] aPreviousLocalState  previous global states
     * \param [in] aControls            control variables
     * \param [in] aConfig              configuration variables
     * \param [in] aResult              output container
     * \param [in] aTimeStep            pseudo time step index
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
        Plato::ComputeDeviatoricStress<mSpaceDim> tComputeDeviatoricStress;
        Plato::ComputeStabilizedCauchyStress<mSpaceDim> tComputeCauchyStress;
        Plato::InterpolateFromNodal<mSpaceDim, mNumGlobalDofsPerNode, mPressureDofOffset> tInterpolatePressureFromNodal;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::Strain<mSpaceDim, mNumGlobalDofsPerNode> tComputeTotalStrain;
        Plato::ThermoPlasticityUtilities<mSpaceDim, SimplexPhysicsType> tThermoPlasticityUtils(mThermalExpansionCoefficient, mReferenceTemperature,
                                                                                               mTemperatureScaling);
        Plato::MSIMP tPenaltyFunction(mPenaltySIMP, mMinErsatz);

        // allocate local containers used to evaluate criterion
        auto tNumCells = mSpatialDomain.numCells();
        
        Plato::ScalarMultiVectorT<PreviousTotalStrainT>   tPreviousTotalStrain("previous total strain",tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<PreviousElasticStrainT> tPreviousElasticStrain("previous elastic strain", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ResultT>                tPreviousDeviatoricStress("previous deviatoric stress", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ResultT>                tPreviousCauchyStress("previous cauchy stress", tNumCells, mNumStressTerms);
        Plato::ScalarVectorT<PrevGlobalStateT>            tPreviousPressure("previous pressure", tNumCells);

        Plato::ScalarMultiVectorT<TotalStrainT>   tCurrentTotalStrain("current total strain",tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ElasticStrainT> tCurrentElasticStrain("current elastic strain", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ResultT>        tCurrentDeviatoricStress("current deviatoric stress", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ResultT>        tCurrentCauchyStress("current cauchy stress", tNumCells, mNumStressTerms);
        Plato::ScalarVectorT<GlobalStateT>        tCurrentPressure("current pressure", tNumCells);

        Plato::ScalarMultiVectorT<ResultT>        tAverageCauchyStress("average cauchy stress", tNumCells, mNumStressTerms);

        Plato::ScalarVectorT<ConfigT>             tCellVolume("cell volume", tNumCells);
        Plato::ScalarMultiVectorT<ResultT>        tElasticStrainMisfit("elastic strain misfit", tNumCells, mNumStressTerms);
        Plato::ScalarArray3DT<ConfigT>            tConfigurationGradient("configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

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

            // compute previous elastic strain
            tComputeTotalStrain(aCellOrdinal, tPreviousTotalStrain, aPreviousGlobalState, tConfigurationGradient);
            tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aPreviousGlobalState, aPreviousLocalState,
                                                        tBasisFunctions, tPreviousTotalStrain, tPreviousElasticStrain);

            // compute current elastic strain
            tComputeTotalStrain(aCellOrdinal, tCurrentTotalStrain, aCurrentGlobalState, tConfigurationGradient);
            tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aCurrentGlobalState, aCurrentLocalState,
                                                        tBasisFunctions, tCurrentTotalStrain, tCurrentElasticStrain);
                                                    
            for (Plato::OrdinalType tIndex = 0; tIndex < tNumStressTerms; ++tIndex)
                tElasticStrainMisfit(aCellOrdinal, tIndex) = tCurrentElasticStrain(aCellOrdinal, tIndex) - tPreviousElasticStrain(aCellOrdinal, tIndex);

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

            // Compute elastic work (strain tensor shear terms already have factor of 2)
            aResult(aCellOrdinal) = 0.0;
            for (Plato::OrdinalType tIndex = 0; tIndex < tNumStressTerms; ++tIndex)
                aResult(aCellOrdinal) += tAverageCauchyStress(aCellOrdinal, tIndex) * tElasticStrainMisfit(aCellOrdinal, tIndex);
            aResult(aCellOrdinal) *= tCellVolume(aCellOrdinal);
        }, "elastic work criterion");
    }

    /******************************************************************************//**
     * \brief Update physics-based data within a frequency of optimization iterations
     * \param [in] aGlobalState global state variables
     * \param [in] aLocalState  local state variables
     * \param [in] aControl     control variables, e.g. design variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aGlobalState,
                       const Plato::ScalarMultiVector & aLocalState,
                       const Plato::ScalarVector & aControl,
                       const Plato::TimeData & aTimeData) override
    {
        auto tPreviousPenaltySIMP = mPenaltySIMP;
        auto tSuggestedPenaltySIMP = tPreviousPenaltySIMP + mAdditiveContinuationParam;
        mPenaltySIMP = tSuggestedPenaltySIMP >= mUpperBoundOnPenaltySIMP ? mUpperBoundOnPenaltySIMP : tSuggestedPenaltySIMP;
        std::ostringstream tMsg;
        tMsg << "Elastic Work Criterion: New penalty parameter is set to '" << mPenaltySIMP
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
            mAdditiveContinuationParam = tInputData.get<Plato::Scalar>("Additive Continuation", 1.1);
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
     * \param [in] aMaterialParams input XML data, i.e. parameter list
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

            mThermalExpansionCoefficient = tThermoelasticSubList.get<Plato::Scalar>("Thermal Expansion Coefficient");
            mReferenceTemperature        = tThermoelasticSubList.get<Plato::Scalar>("Reference Temperature");
        }
        else
        {
            THROWERR("'Isotropic Linear Elastic' or 'Isotropic Linear Thermoelastic' sublist of 'Material Model' is not defined.")
        }
    }
};
// class ElasticWorkCriterion

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_INC_VMS(Plato::ElasticWorkCriterion, Plato::SimplexPlasticity, 2)
PLATO_EXPL_DEC_INC_VMS(Plato::ElasticWorkCriterion, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_INC_VMS(Plato::ElasticWorkCriterion, Plato::SimplexPlasticity, 3)
PLATO_EXPL_DEC_INC_VMS(Plato::ElasticWorkCriterion, Plato::SimplexThermoPlasticity, 3)
#endif

}
// namespace Plato

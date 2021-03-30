/*
 * ThermoplasticityThermalEnergyCriterion.hpp
 *
 *  Created on: Mar 22, 2021
 */

#pragma once

#include "Simp.hpp"
#include "SimplexFadTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "SimplexThermoPlasticity.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "InterpolateGradientFromScalarNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "AbstractLocalScalarFunctionInc.hpp"

#include "ExpInstMacros.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Evaluate the thermoplasticity thermal energy criterion.
 *
 *
 * \tparam EvaluationType      evaluation type for scalar function, determines
 *                             which AD type is active
 * \tparam SimplexPhysicsType  simplex physics type, determines values of
 *                             physics-based static parameters
*******************************************************************************/
template<typename EvaluationType, typename SimplexPhysicsType>
class ThermoplasticityThermalEnergyCriterion : public Plato::AbstractLocalScalarFunctionInc<EvaluationType>
{
// private member data
private:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    static constexpr auto mNumNodesPerCell = SimplexPhysicsType::mNumNodesPerCell;      /*!< number nodes per cell */
    static constexpr auto mNumGlobalDofsPerNode = SimplexPhysicsType::mNumDofsPerNode;  /*!< number global degrees of freedom per node */
    static constexpr auto mTemperatureDofOffset = SimplexPhysicsType::mTemperatureDofOffset;  /*!< temperature dof offset */

    using ResultT = typename EvaluationType::ResultScalarType;                     /*!< result variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType;                     /*!< config variables automatic differentiation type */
    using ControlT = typename EvaluationType::ControlScalarType;                   /*!< control variables automatic differentiation type */
    using LocalStateT = typename EvaluationType::LocalStateScalarType;             /*!< local state variables automatic differentiation type */
    using GlobalStateT = typename EvaluationType::StateScalarType;                 /*!< global state variables automatic differentiation type */
    using PrevLocalStateT = typename EvaluationType::PrevLocalStateScalarType;     /*!< local state variables automatic differentiation type */
    using PrevGlobalStateT = typename EvaluationType::PrevStateScalarType;         /*!< global state variables automatic differentiation type */

    using FunctionBaseType = Plato::AbstractLocalScalarFunctionInc<EvaluationType>;
    using Plato::AbstractLocalScalarFunctionInc<EvaluationType>::mSpatialDomain;

    Plato::Scalar mThermalConductivityCoefficient;    /*!< thermal conductivity coefficient */
    Plato::Scalar mTemperatureScaling;             /*!< temperature scaling */

    Plato::Scalar mPenaltySIMP;                /*!< SIMP penalty for elastic properties */
    Plato::Scalar mMinErsatz;                  /*!< SIMP min ersatz stiffness for elastic properties */
    Plato::Scalar mUpperBoundOnPenaltySIMP;    /*!< continuation parameter: upper bound on SIMP penalty for elastic properties */
    Plato::Scalar mAdditiveContinuationParam;  /*!< continuation parameter: multiplier on SIMP penalty for elastic properties */

    Plato::LinearTetCubRuleDegreeOne<mSpaceDim> mCubatureRule;  /*!< simplex linear cubature rule */

// public access functions
public:
    /***************************************************************************//**
     * \brief Constructor for thermoplasticity thermal energy criterion
     *
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap     PLATO Analyze output data map side sets database
     * \param [in] aInputParams input parameters from XML file
     * \param [in] aName        scalar function name
    *******************************************************************************/
    ThermoplasticityThermalEnergyCriterion(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aName),
        mThermalConductivityCoefficient(0.0),
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
     * \brief Constructor for total work criterion
     *
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap     PLATO Analyze output data map side sets database
     * \param [in] aName        scalar function name
    *******************************************************************************/
    ThermoplasticityThermalEnergyCriterion(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap,
              std::string            aName = ""
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aName),
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
    virtual ~ThermoplasticityThermalEnergyCriterion(){}


    /***************************************************************************//**
     * \brief Evaluates total work criterion. FAD type determines output/result value.
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
                  Plato::Scalar aTimeStep = 0.0)
    {
        using GState_Config_T = typename Plato::fad_type_t<SimplexPhysicsType, GlobalStateT, ConfigT>;
        using GState_Config_Control_T = typename Plato::fad_type_t<SimplexPhysicsType, ControlT, ConfigT, GlobalStateT>;

        // allocate functors used to evaluate criterion
        Plato::InterpolateGradientFromScalarNodal<mSpaceDim, mNumGlobalDofsPerNode, mTemperatureDofOffset> tInterpolateTemperatureGradFromNodal;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::MSIMP tPenaltyFunction(mPenaltySIMP, mMinErsatz);

        // allocate local containers used to evaluate criterion
        auto tNumCells = mSpatialDomain.numCells();
        
        Plato::ScalarMultiVectorT<GState_Config_T> tTemperatureGrad("temperature grad", tNumCells, mSpaceDim);
        Plato::ScalarVectorT<ConfigT>              tCellVolume("cell volume", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tConfigurationGradient("configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        auto tSpaceDim = mSpaceDim;
        auto tTemperatureScalingSquared = mTemperatureScaling * mTemperatureScaling;
        auto tThermalConductivityCoefficient = mThermalConductivityCoefficient;

        constexpr Plato::Scalar tOneHalf = 0.5;

        auto tQuadratureWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
        {
            // compute configuration gradients
            tComputeGradient(aCellOrdinal, tConfigurationGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;

            tInterpolateTemperatureGradFromNodal(aCellOrdinal, tConfigurationGradient, aCurrentGlobalState, tTemperatureGrad);
            
            // compute cell penalty and penalized elastic properties
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControls);
            ControlT tElasticPropertiesPenalty = tPenaltyFunction(tDensity);
            ControlT tPenalizedThermalConductivityCoefficient = tElasticPropertiesPenalty * tThermalConductivityCoefficient;

            aResult(aCellOrdinal) = 0.0;
            for (Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; ++tIndex)
                aResult(aCellOrdinal) += tTemperatureGrad(aCellOrdinal, tIndex) * tTemperatureGrad(aCellOrdinal, tIndex);
            aResult(aCellOrdinal) *= 
                tPenalizedThermalConductivityCoefficient * (tCellVolume(aCellOrdinal) * tTemperatureScalingSquared * tOneHalf);
        }, "thermal energy criterion");
    }

    /******************************************************************************//**
     * \brief Update physics-based data within a frequency of optimization iterations
     * \param [in] aGlobalState global state variables
     * \param [in] aLocalState  local state variables
     * \param [in] aControl     control variables, e.g. design variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aGlobalState,
                       const Plato::ScalarMultiVector & aLocalState,
                       const Plato::ScalarVector & aControl) override
    {
        auto tPreviousPenaltySIMP = mPenaltySIMP;
        auto tSuggestedPenaltySIMP = tPreviousPenaltySIMP + mAdditiveContinuationParam;
        mPenaltySIMP = tSuggestedPenaltySIMP >= mUpperBoundOnPenaltySIMP ? mUpperBoundOnPenaltySIMP : tSuggestedPenaltySIMP;
        std::ostringstream tMsg;
        tMsg << "Thermoplasticity Thermal Energy Criterion: New penalty parameter is set to '" << mPenaltySIMP
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
        if (aMaterialParams.isSublist("Isotropic Linear Thermoelastic"))
        {
            auto tThermoelasticSubList = aMaterialParams.sublist("Isotropic Linear Thermoelastic");
            mThermalConductivityCoefficient = tThermoelasticSubList.get<Plato::Scalar>("Thermal Conductivity Coefficient");
        }
        else
        {
            THROWERR("'Isotropic Linear Thermoelastic' sublist of 'Material Model' is not defined.")
        }
    }
};
// class ThermoplasticityThermalEnergyCriterion

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_INC_VMS(Plato::ThermoplasticityThermalEnergyCriterion, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_INC_VMS(Plato::ThermoplasticityThermalEnergyCriterion, Plato::SimplexThermoPlasticity, 3)
#endif

}
// namespace Plato

/*
 * InfinitesimalStrainThermoPlasticityResidual.hpp
 *
 *  Created on: Jan 20, 2021
 */

#pragma once

#include "Simp.hpp"
#include "ToMap.hpp"
#include "Strain.hpp"
#include "BodyLoads.hpp"
#include "NaturalBCs.hpp"
#include "ScalarGrad.hpp"
#include "ProjectToNode.hpp"
#include "FluxDivergence.hpp"
#include "StressDivergence.hpp"
#include "StrainDivergence.hpp"
#include "SimplexThermoPlasticity.hpp"
#include "PressureDivergence.hpp"
#include "ComputeCauchyStress.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "InterpolateGradientFromScalarNodal.hpp"
#include "ComputeStabilization.hpp"
#include "J2PlasticityUtilities.hpp"
#include "ComputeDeviatoricStress.hpp"
#include "ComputePrincipalStresses.hpp"
#include "ThermoPlasticityUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "IsotropicMaterialUtilities.hpp"
#include "AbstractGlobalVectorFunctionInc.hpp"
#include "ExpInstMacros.hpp"

namespace Plato
{

/***********************************************************************//**
 * \brief Evaluate stabilized infinitesimal strain plasticity residual, defined as
 *
 * \tparam EvaluationType denotes evaluation type for vector function, possible
 *   options are Residual, Jacobian, PartialControl, etc.
 * \tparam SimplexPhysicsType simplex physics type, e.g. SimplexThermoPlasticity. gives
 *   access to static data related to the physics problem.
 *
 * \f$   \langle \nabla{v_h}, s_h \rangle + \langle \nabla\cdot{v_h}, p_h \rangle
 *     - \langle v_h, f \rangle - \langle v_h, b \rangle = 0\ \forall\ v_h \in V_{h,0}
 *       = \{v_h \in V_h | v = 0\ \mbox{in}\ \partial\Omega_{u} \} \f$
 *
 * \f$   \langle q_h, \nabla\cdot{u_h} \rangle - \langle q_h, \frac{1}{K}p_h \rangle
 *     - \sum_{e=1}^{N_{elem}} \tau_e \langle \nabla{q_h} \left[ \nabla{p_h} - \Pi_h
 *       \right] \rangle = 0\ \forall\ q_h \in L_h \subset\ L^2(\Omega) \f$
 *
 * \f$ \langle \nabla{p_h}, \eta_h \rangle - \langle \Pi_h, \eta_h \rangle = 0\
 *     \forall\ \eta_h \in V_h \subset\ H^{1}(\Omega) \f$
 *
***************************************************************************/
template<typename EvaluationType, typename SimplexPhysicsType>
class InfinitesimalStrainThermoPlasticityResidual: public Plato::AbstractGlobalVectorFunctionInc<EvaluationType>
{
// Private member data
private:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim;                      /*!< number of spatial dimensions */
    static constexpr auto mNumStressTerms = SimplexPhysicsType::mNumStressTerms;       /*!< number of stress/strain components */
    static constexpr auto mNumDofsPerCell = SimplexPhysicsType::mNumDofsPerCell;       /*!< number of degrees of freedom (dofs) per cell */
    static constexpr auto mNumNodesPerCell = SimplexPhysicsType::mNumNodesPerCell;     /*!< number nodes per cell */
    static constexpr auto mPressureDofOffset = SimplexPhysicsType::mPressureDofOffset; /*!< pressure dofs offset */
    static constexpr auto mNumGlobalDofsPerNode = SimplexPhysicsType::mNumDofsPerNode; /*!< number of global dofs per node */

    static constexpr auto mNumDisplacementDims = mSpaceDim;                                    /*!< number of displacement degrees of freedom */
    static constexpr auto mDisplacementDofOffset = SimplexPhysicsType::mDisplacementDofOffset; /*!< displacement degrees of freedom offset */
    static constexpr Plato::OrdinalType mNumThermalDims = 1;                           /*!< number of thermal degrees of freedom */
    static constexpr auto mTemperatureDofOffset = SimplexPhysicsType::mTemperatureDofOffset; /*!< temperature dofs offset */

    using Plato::AbstractGlobalVectorFunctionInc<EvaluationType>::mSpatialDomain; /*!< mesh database */
    using Plato::AbstractGlobalVectorFunctionInc<EvaluationType>::mDataMap;       /*!< PLATO Engine output database */

    using GlobalStateT = typename EvaluationType::StateScalarType;             /*!< global state variables automatic differentiation type */
    using PrevGlobalStateT = typename EvaluationType::PrevStateScalarType;     /*!< global state variables automatic differentiation type */
    using LocalStateT = typename EvaluationType::LocalStateScalarType;         /*!< local state variables automatic differentiation type */
    using PrevLocalStateT = typename EvaluationType::PrevLocalStateScalarType; /*!< local state variables automatic differentiation type */
    using NodeStateT = typename EvaluationType::NodeStateScalarType;           /*!< node State AD type */
    using ControlT = typename EvaluationType::ControlScalarType;               /*!< control variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType;                 /*!< config variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType;                 /*!< result variables automatic differentiation type */

    using FunctionBaseType = Plato::AbstractGlobalVectorFunctionInc<EvaluationType>;
    using CubatureType = Plato::LinearTetCubRuleDegreeOne<mSpaceDim>;

    Plato::Scalar mPoissonsRatio;                  /*!< Poisson's ratio */
    Plato::Scalar mElasticModulus;                 /*!< elastic modulus */
    Plato::Scalar mPressureScaling;                /*!< Pressure scaling term */
    Plato::Scalar mTemperatureScaling;                /*!< Temperature scaling term */
    Plato::Scalar mElasticBulkModulus;             /*!< elastic bulk modulus */
    Plato::Scalar mElasticShearModulus;            /*!< elastic shear modulus */
    Plato::Scalar mThermalConductivityCoefficient; /*!< thermal conductivity coefficient */
    Plato::Scalar mThermalExpansionCoefficient;    /*!< thermal expansion coefficient */
    Plato::Scalar mReferenceTemperature;           /*!< thermal reference temperature */

    Plato::Scalar mPenaltySIMP;               /*!< SIMP penalty for elastic properties */
    Plato::Scalar mMinErsatzSIMP;             /*!< SIMP min ersatz stiffness for elastic properties */
    Plato::Scalar mUpperBoundOnPenaltySIMP;   /*!< continuation parameter: upper bound on SIMP penalty for elastic properties */
    Plato::Scalar mAdditiveContinuationParam; /*!< continuation parameter: multiplier on SIMP penalty for elastic properties */

    std::vector<std::string> mPlotTable;           /*!< array of output element data identifiers*/

    std::shared_ptr<Plato::BodyLoads<EvaluationType>> mBodyLoads;                       /*!< body loads interface */
    std::shared_ptr<CubatureType> mCubatureRule;                                        /*!< linear cubature rule */
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim, mNumDisplacementDims, mNumGlobalDofsPerNode, mDisplacementDofOffset>> 
                    mNeumannMechanicalLoads; /*!< Neumann mechanical loads interface */
    std::shared_ptr<Plato::NaturalBCs<mSpaceDim, mNumThermalDims, mNumGlobalDofsPerNode, mTemperatureDofOffset>> 
                    mNeumannThermalLoads; /*!< Neumann thermal loads interface */

// Private access functions
private:
    /***************************************************************************//**
     * \brief initialize material, loads and output data
     * \param [in] aProblemParams input XML data, i.e. parameter list
    *******************************************************************************/
    void initialize(Teuchos::ParameterList &aProblemParams)
    {
        this->parseExternalForces(aProblemParams);
        this->parseOutputDataNames(aProblemParams);
        this->parseMaterialProperties(aProblemParams);
        this->parseMaterialPenaltyInputs(aProblemParams);
    }

    /***************************************************************************//**
     * \brief Parse output data names
     * \param [in] aProblemParams input XML data, i.e. parameter list
    *******************************************************************************/
    void parseOutputDataNames(Teuchos::ParameterList &aProblemParams)
    {
        if(aProblemParams.isSublist("Elliptic"))
        {
            auto tResidualParams = aProblemParams.sublist("Elliptic");
            if (tResidualParams.isType < Teuchos::Array < std::string >> ("Plottable"))
            {
                mPlotTable = tResidualParams.get < Teuchos::Array < std::string >> ("Plottable").toVector();
            }
        }
        else
        {
            THROWERR("Infinitesimal Strain Thermoplasticity Residual: 'Elliptic' sublist is not defined in XML input file.")
        }
    }

    /***************************************************************************//**
     * \brief Parse material penalty inputs
     * \param [in] aProblemParams input XML data, i.e. parameter list
    *******************************************************************************/
    void parseMaterialPenaltyInputs(Teuchos::ParameterList &aProblemParams)
    {
        if(aProblemParams.isSublist("Elliptic"))
        {
            auto tResidualParams = aProblemParams.sublist("Elliptic");
            if(tResidualParams.isSublist("Penalty Function"))
            {
                auto tPenaltyParams = tResidualParams.sublist("Penalty Function");
                mPenaltySIMP = tPenaltyParams.get<Plato::Scalar>("Exponent", 1.0);
                mMinErsatzSIMP = tPenaltyParams.get<Plato::Scalar>("Minimum Value", 1e-9);
                mUpperBoundOnPenaltySIMP = tPenaltyParams.get<Plato::Scalar>("Penalty Exponent Upper Bound", 4.0);
                mAdditiveContinuationParam = tPenaltyParams.get<Plato::Scalar>("Additive Continuation", 0.1);
            }
        }
        else
        {
            THROWERR("Infinitesimal Strain Thermoplasticity Residual: 'Elliptic' sublist is not defined in XML input file.")
        }
    }

    /***********************************************************************//**
     * \brief Parse external forces
     * \param [in] aProblemParams input XML data, i.e. parameter list
    ***************************************************************************/
    void parseExternalForces(Teuchos::ParameterList &aProblemParams)
    {
        // Parse body loads
        if (aProblemParams.isSublist("Body Loads"))
        {
            mBodyLoads = std::make_shared<Plato::BodyLoads<EvaluationType>>(aProblemParams.sublist("Body Loads"));
        }

        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            auto tNaturalBCsParams = aProblemParams.sublist("Natural Boundary Conditions");

            // Parse mechanical Neumann loads
            if(tNaturalBCsParams.isSublist("Mechanical Natural Boundary Conditions"))
            {
                mNeumannMechanicalLoads =
                        std::make_shared<Plato::NaturalBCs<mSpaceDim, mNumDisplacementDims, mNumGlobalDofsPerNode, mDisplacementDofOffset>>
                        (tNaturalBCsParams.sublist("Mechanical Natural Boundary Conditions"));
            }
            else
            {
                REPORT("No 'Mechanical Natural Boundary Conditions' specified.")
            }

            // Parse thermal Neumann loads
            if(tNaturalBCsParams.isSublist("Thermal Natural Boundary Conditions"))
            {
                mNeumannThermalLoads =
                        std::make_shared<Plato::NaturalBCs<mSpaceDim, mNumThermalDims, mNumGlobalDofsPerNode, mTemperatureDofOffset>>
                        (tNaturalBCsParams.sublist("Thermal Natural Boundary Conditions"));
            }
            else
            {
                REPORT("No 'Thermal Natural Boundary Conditions' specified.")
            }
        }

        mDataMap.mScalarValues["LoadControlConstant"] = 1.0;
    }

    /**********************************************************************//**
     * \brief Parse elastic material properties
     * \param [in] aProblemParams input XML data, i.e. parameter list
    **************************************************************************/
    void parseMaterialProperties(Teuchos::ParameterList &aProblemParams)
    {
        if(aProblemParams.isSublist("Material Models"))
        {
            this->parseIsotropicMaterialProperties(aProblemParams);
        }
        else
        {
            THROWERR("Infinitesimal Strain Thermoplasticity Residual: 'Material Models' sublist is not defined.")
        }
    }

    /**********************************************************************//**
     * \brief Parse isotropic material parameters
     * \param [in] aProblemParams input XML data, i.e. parameter list
    **************************************************************************/
    void parseIsotropicMaterialProperties(Teuchos::ParameterList &aProblemParams)
    {
        Teuchos::ParameterList tMaterialsInputs = aProblemParams.sublist("Material Models");

        auto tMaterialName = mSpatialDomain.getMaterialName();
        Teuchos::ParameterList tMaterialInputs = tMaterialsInputs.sublist(tMaterialName);

        mPressureScaling = tMaterialInputs.get<Plato::Scalar>("Pressure Scaling", 1.0);
        mTemperatureScaling = tMaterialInputs.get<Plato::Scalar>("Temperature Scaling", 1.0);
        if (tMaterialInputs.isSublist("Isotropic Linear Thermoelastic"))
        {
            auto tThermoelasticSubList = tMaterialInputs.sublist("Isotropic Linear Thermoelastic");
            mPoissonsRatio = tThermoelasticSubList.get<Plato::Scalar>("Poissons Ratio");
            mElasticModulus = tThermoelasticSubList.get<Plato::Scalar>("Youngs Modulus");
            //mPoissonsRatio = Plato::parse_poissons_ratio(tThermoelasticSubList);
            //mElasticModulus = Plato::parse_elastic_modulus(tThermoelasticSubList);
            mElasticBulkModulus = Plato::compute_bulk_modulus(mElasticModulus, mPoissonsRatio);
            mElasticShearModulus = Plato::compute_shear_modulus(mElasticModulus, mPoissonsRatio);

            mThermalExpansionCoefficient    = tThermoelasticSubList.get<Plato::Scalar>("Thermal Expansion Coefficient");
            mReferenceTemperature           = tThermoelasticSubList.get<Plato::Scalar>("Reference Temperature");
            mThermalConductivityCoefficient = tThermoelasticSubList.get<Plato::Scalar>("Thermal Conductivity Coefficient");
        }
        else
        {
            std::stringstream ss;
            ss << "Infinitesimal Strain Thermoplasticity Residual: 'Isotropic Linear Thermoelastic' sublist of '" << tMaterialName << "' is not defined.";
            THROWERR(ss.str());
        }
    }

    /**********************************************************************//**
     * \brief Copy data to output data map
     * \tparam DataT data type
     * \param [in] aData output data
     * \param [in] aName output data name
    **************************************************************************/
    template<typename DataT>
    void outputData(const DataT & aData, const std::string & aName)
    {
        if(std::count(mPlotTable.begin(), mPlotTable.end(), aName))
        {
             Plato::toMap(mDataMap, aData, aName, mSpatialDomain);
        }
    }

    /************************************************************************//**
     * \brief Add external neumann forces to residual
     * \param [in]     aGlobalState current global state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aControls    design variables
     * \param [in]     aConfig      configuration variables
     * \param [in/out] aResult      residual evaluation
    ****************************************************************************/
    void addExternalForces(
        const Plato::SpatialModel                      & aSpatialModel,
        const Plato::ScalarMultiVectorT <GlobalStateT> & aGlobalState,
        const Plato::ScalarMultiVectorT <ControlT>     & aControl,
        const Plato::ScalarArray3DT     <ConfigT>      & aConfig,
        const Plato::ScalarMultiVectorT <ResultT>      & aResult)
    {
        auto tSearch = mDataMap.mScalarValues.find("LoadControlConstant");
        if(tSearch == mDataMap.mScalarValues.end())
        {
            THROWERR("Infinitesimal Strain Thermoplasticity Residual: 'Load Control Constant' is NOT defined in data map.")
        }

        auto tMultiplier = static_cast<Plato::Scalar>(-1.0) * tSearch->second;
        if( mNeumannMechanicalLoads != nullptr )
        {
            mNeumannMechanicalLoads->get( aSpatialModel, aGlobalState, aControl, aConfig, aResult, tMultiplier );
        }

        tMultiplier = static_cast<Plato::Scalar>(-1.0);
        if( mNeumannThermalLoads != nullptr )
        {
            mNeumannThermalLoads->get( aSpatialModel, aGlobalState, aControl, aConfig, aResult, tMultiplier );
        }
    }

    /************************************************************************//**
     * \brief Add body forces to residual
     * \param [in]     aGlobalState current global state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aControls    design variables
     * \param [in]     aConfig      configuration variables
     * \param [in/out] aResult      residual evaluation
    ****************************************************************************/
    void
    addBodyForces(
        const Plato::ScalarMultiVectorT <GlobalStateT> & aGlobalState,
        const Plato::ScalarMultiVectorT <ControlT>     & aControl,
        const Plato::ScalarArray3DT     <ConfigT>      & aConfig,
        const Plato::ScalarMultiVectorT <ResultT>      & aResult)
    {
        auto tSearch = mDataMap.mScalarValues.find("LoadControlConstant");
        if(tSearch == mDataMap.mScalarValues.end())
        {
            THROWERR("Infinitesimal Strain Thermoplasticity Residual: 'Load Control Constant' is NOT defined in data map.")
        }

        auto tMultiplier = static_cast<Plato::Scalar>(-1.0) * tSearch->second;
        if (mBodyLoads != nullptr)
        {
            mBodyLoads->get( mSpatialDomain, aGlobalState, aControl, aResult, tMultiplier );
        }
    }

    /************************************************************************//**
     * \brief Compute principal stress components
     * \param [in]     aGlobalState current global state ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aControls    design variables
     * \param [in]     aConfig      configuration variables
     * \param [in/out] aResult      residual evaluation
    ****************************************************************************/
    void
    computePrincipalStresses(
        const Plato::ScalarMultiVectorT <GlobalStateT> & aGlobalState,
        const Plato::ScalarMultiVectorT <LocalStateT>  & aLocalState,
        const Plato::ScalarMultiVectorT <ControlT>     & aControl,
        const Plato::ScalarArray3DT     <ConfigT>      & aConfig
    )
    {
        if(std::count(mPlotTable.begin(), mPlotTable.end(), "principal stresses"))
        {
            Plato::ComputePrincipalStresses<EvaluationType, SimplexPhysicsType> tComputePrincipalStresses;
            tComputePrincipalStresses.setBulkModulus(mElasticBulkModulus);
            tComputePrincipalStresses.setShearModulus(mElasticShearModulus);
            tComputePrincipalStresses.setPenaltySIMP(mPenaltySIMP);
            tComputePrincipalStresses.setMinErsatzSIMP(mMinErsatzSIMP);

            const auto tNumCells = aGlobalState.extent(0);
            Plato::ScalarMultiVectorT<ResultT> tPrincipalStresses("principal stresses", tNumCells, mSpaceDim);
            tComputePrincipalStresses(aGlobalState, aLocalState, aControl, aConfig, tPrincipalStresses);
            this->outputData(tPrincipalStresses, "principal stresses");
        }
    }

// Public access functions
public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh          mesh metadata
     * \param [in] aMeshSets      side-sets metadata
     * \param [in] aDataMap       output data map
     * \param [in] aProblemParams input XML data
     * \param [in] aPenaltyParams penalty function input XML data
    *******************************************************************************/
    InfinitesimalStrainThermoPlasticityResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap),
        mPoissonsRatio(-1.0),
        mElasticModulus(-1.0),
        mPressureScaling(1.0),
        mTemperatureScaling(1.0),
        mElasticBulkModulus(-1.0),
        mElasticShearModulus(-1.0),
        mThermalConductivityCoefficient(-1.0),
        mThermalExpansionCoefficient(0.0),
        mReferenceTemperature(0.0),
        mPenaltySIMP(3),
        mMinErsatzSIMP(1e-9),
        mUpperBoundOnPenaltySIMP(4),
        mAdditiveContinuationParam(0.1),
        mBodyLoads(nullptr),
        mCubatureRule(std::make_shared<CubatureType>()),
        mNeumannMechanicalLoads(nullptr),
        mNeumannThermalLoads(nullptr)
    {
        this->initialize(aProblemParams);
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    virtual ~InfinitesimalStrainThermoPlasticityResidual()
    {
    }

    /***************************************************************************//**
     * \brief Set load control multiplier
     * \param [in] aInput load control multiplier
    *******************************************************************************/
    void setLoadControlMultiplier(const Plato::Scalar& aInput)
    {
        mDataMap.mScalarValues["LoadControlConstant"] = aInput;
    }

    /************************************************************************//**
     * \brief Evaluate the stabilized residual equation
     *
     * \param [in]     aCurrentGlobalState    current global state workset ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aPrevGlobalState       previous global state workset ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in]     aCurrentLocalState     current local state workset ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aPrevLocalState        previous local state workset ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in]     aProjectedPressureGrad current pressure gradient workset ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aControls               design variables workset
     * \param [in]     aConfig                configuration workset
     * \param [in/out] aResult                residual workset
     * \param [in]     aTimeStep              current time step (i.e. \f$ \Delta{t}^{n} \f$), default = 0.0
     *
    ****************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <GlobalStateT>     & aCurrentGlobalState,
        const Plato::ScalarMultiVectorT <PrevGlobalStateT> & aPrevGlobalState,
        const Plato::ScalarMultiVectorT <LocalStateT>      & aCurrentLocalState,
        const Plato::ScalarMultiVectorT <PrevLocalStateT>  & aPrevLocalState,
        const Plato::ScalarMultiVectorT <NodeStateT>       & aProjectedPressureGrad,
        const Plato::ScalarMultiVectorT <ControlT>         & aControls,
        const Plato::ScalarArray3DT     <ConfigT>          & aConfig,
        const Plato::ScalarMultiVectorT <ResultT>          & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) override
    {
        auto tNumCells = mSpatialDomain.numCells();
        auto tSpaceDim = mSpaceDim;

        using GradScalarT = typename Plato::fad_type_t<SimplexPhysicsType, GlobalStateT, ConfigT>;
        using ElasticStrainT = typename Plato::fad_type_t<SimplexPhysicsType, LocalStateT, ConfigT, GlobalStateT>;
        using ThermalFluxT = typename Plato::fad_type_t<SimplexPhysicsType, ControlT, ConfigT, GlobalStateT>;

        // Functors used to compute residual-related quantities
        Plato::ScalarGrad<mSpaceDim> tComputeScalarGrad;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::ComputeCauchyStress<mSpaceDim> tComputeCauchyStress;
        Plato::J2PlasticityUtilities<mSpaceDim>  tJ2PlasticityUtils;
        Plato::StrainDivergence <mSpaceDim> tComputeStrainDivergence;
        Plato::ComputeDeviatoricStress<mSpaceDim> tComputeDeviatoricStress;
        Plato::Strain<mSpaceDim, mNumGlobalDofsPerNode> tComputeTotalStrain;
        Plato::ThermoPlasticityUtilities<mSpaceDim, SimplexPhysicsType> tThermoPlasticityUtils(mThermalExpansionCoefficient, mReferenceTemperature);
        Plato::ComputeStabilization<mSpaceDim> tComputeStabilization(mPressureScaling, mElasticShearModulus);
        Plato::InterpolateFromNodal<mSpaceDim, mNumGlobalDofsPerNode, mPressureDofOffset> tInterpolatePressureFromNodal;
        Plato::InterpolateFromNodal<mSpaceDim, mSpaceDim, mDisplacementDofOffset, mSpaceDim> tInterpolatePressGradFromNodal;
        Plato::InterpolateFromNodal<mSpaceDim, mNumGlobalDofsPerNode, mTemperatureDofOffset> tInterpolateTemperatureFromNodal;
        Plato::InterpolateGradientFromScalarNodal<mSpaceDim, mNumGlobalDofsPerNode, mTemperatureDofOffset> tInterpolateTemperatureGradFromNodal;

        // Residual evaulation functors
        Plato::PressureDivergence<mSpaceDim, mNumGlobalDofsPerNode> tPressureDivergence;
        Plato::StressDivergence<mSpaceDim, mNumGlobalDofsPerNode, mDisplacementDofOffset> tStressDivergence;
        Plato::ProjectToNode<mSpaceDim, mNumGlobalDofsPerNode, mPressureDofOffset> tProjectVolumeStrain;
        Plato::FluxDivergence<mSpaceDim, mNumGlobalDofsPerNode, mPressureDofOffset> tStabilizedDivergence;
        Plato::FluxDivergence<mSpaceDim, mNumGlobalDofsPerNode, mTemperatureDofOffset> tThermalFluxDivergence;
        Plato::MSIMP tPenaltyFunction(mPenaltySIMP, mMinErsatzSIMP);

        Plato::ScalarVectorT<ResultT> tPressure("L2 pressure", tNumCells);
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarVectorT<ResultT> tVolumeStrain("volume strain", tNumCells);
        Plato::ScalarVectorT<ResultT> tStrainDivergence("strain divergence", tNumCells);
        Plato::ScalarVectorT<ResultT> tTemperature("temperature", tNumCells);
        Plato::ScalarVectorT<ResultT> tThermalVolumetricStrain("thermal volumetric strain", tNumCells);
        Plato::ScalarMultiVectorT<ResultT> tStabilization("cell stabilization", tNumCells, mSpaceDim);
        Plato::ScalarMultiVectorT<GradScalarT> tPressureGrad("pressure gradient", tNumCells, mSpaceDim);
        Plato::ScalarMultiVectorT<GradScalarT> tTotalStrain("total strain", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ResultT> tDeviatoricStress("deviatoric stress", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ElasticStrainT> tElasticStrain("elastic strain", tNumCells, mNumStressTerms);
        Plato::ScalarArray3DT<ConfigT> tConfigurationGradient("configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);
        Plato::ScalarMultiVectorT<NodeStateT> tProjectedPressureGradGP("projected pressure gradient", tNumCells, mSpaceDim);
        Plato::ScalarMultiVectorT<GradScalarT> tTemperatureGrad("temperature grad", tNumCells, mSpaceDim);
        Plato::ScalarMultiVectorT<ThermalFluxT> tThermalFlux("thermal flux", tNumCells, mSpaceDim);

        // output quantities
        Plato::ScalarMultiVectorT<ResultT> tCauchyStress("cauchy stress", tNumCells, mNumStressTerms);
        Plato::ScalarVectorT<LocalStateT> tAccumPlasticStrain("accumulated plastic strain", tNumCells);
        Plato::ScalarVectorT<LocalStateT> tPlasticMultiplier("plastic multiplier increment", tNumCells);
        Plato::ScalarMultiVectorT<LocalStateT> tPlasticStrain("plastic strain", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<LocalStateT> tBackStress("back-stress stress", tNumCells, mNumStressTerms);

        // Transfer elasticity parameters to device
        auto tNumDofsPerNode = mNumGlobalDofsPerNode;
        auto tPressureScaling = mPressureScaling;
        auto tPressureDofOffset = mPressureDofOffset;
        auto tElasticBulkModulus = mElasticBulkModulus;
        auto tElasticShearModulus = mElasticShearModulus;

        //auto tTemperatureScaling = mTemperatureScaling;
        auto tThermalConductivityCoefficient = mThermalConductivityCoefficient;
        auto tThermalExpansionCoefficient = mThermalExpansionCoefficient;
        auto tReferenceTemperature = mReferenceTemperature;

        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
        {
            // compute configuration gradients
            tComputeGradient(aCellOrdinal, tConfigurationGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;

            // compute thermal quantities
            tInterpolateTemperatureFromNodal(aCellOrdinal, tBasisFunctions, aCurrentGlobalState, tTemperature);
            tInterpolateTemperatureGradFromNodal(aCellOrdinal, tConfigurationGradient, aCurrentGlobalState, tTemperatureGrad);
            // Trace of the isotropic thermal strain tensor which for 2D plane strain and 3D is always 3*thermal_strain
            tThermalVolumetricStrain(aCellOrdinal) = static_cast<ThermalFluxT>(3.0) * tThermalExpansionCoefficient * 
                                                     (tTemperature(aCellOrdinal) - tReferenceTemperature);

            // compute elastic strain, i.e. e_elastic = e_total - e_plastic - e_thermal
            tComputeTotalStrain(aCellOrdinal, tTotalStrain, aCurrentGlobalState, tConfigurationGradient);
            tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aCurrentGlobalState, aCurrentLocalState,
                                                        tBasisFunctions, tTotalStrain, tElasticStrain);

            // compute pressure gradient
            tComputeScalarGrad(aCellOrdinal, tNumDofsPerNode, tPressureDofOffset,
                               aCurrentGlobalState, tConfigurationGradient, tPressureGrad);

            // interpolate projected pressure grad, pressure, and temperature to gauss point
            tInterpolatePressureFromNodal(aCellOrdinal, tBasisFunctions, aCurrentGlobalState, tPressure);
            tInterpolatePressGradFromNodal(aCellOrdinal, tBasisFunctions, aProjectedPressureGrad, tProjectedPressureGradGP);

            // compute cell penalty
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControls);
            ControlT tElasticPropertiesPenalty = tPenaltyFunction(tDensity);

            // compute deviatoric stress and displacement divergence
            ControlT tPenalizedShearModulus = tElasticPropertiesPenalty * tElasticShearModulus;
            tComputeDeviatoricStress(aCellOrdinal, tPenalizedShearModulus, tElasticStrain, tDeviatoricStress);
            tComputeStrainDivergence(aCellOrdinal, tTotalStrain, tStrainDivergence); /*This is actually displacement divergence*/

            // compute volume difference
            tPressure(aCellOrdinal) *= tPressureScaling * tElasticPropertiesPenalty;
            tVolumeStrain(aCellOrdinal) = tPressureScaling * tElasticPropertiesPenalty /* IS THIS RIGHT WITH DOUBLE PENALTY ON PRESSURE? Shouldn't it
                                                                                          just be on the bulk modulus? */
                * (tStrainDivergence(aCellOrdinal) - tThermalVolumetricStrain(aCellOrdinal) - tPressure(aCellOrdinal) / tElasticBulkModulus);

            // compute cell stabilization term
            tComputeStabilization(aCellOrdinal, tCellVolume, tPressureGrad, tProjectedPressureGradGP, tStabilization);
            Plato::apply_penalty<mSpaceDim>(aCellOrdinal, tElasticPropertiesPenalty, tStabilization);

            // compute the thermal flux
            ControlT tPenalizedThermalConductivityCoefficient = tElasticPropertiesPenalty * tThermalConductivityCoefficient;
            for (Plato::OrdinalType tDimIndex = 0; tDimIndex < tSpaceDim; ++tDimIndex)
                tThermalFlux(aCellOrdinal, tDimIndex) = static_cast<ThermalFluxT>(-1.0) * tPenalizedThermalConductivityCoefficient *
                                                        tTemperatureGrad(aCellOrdinal, tDimIndex);

            // compute residual
            tStressDivergence (aCellOrdinal, aResult, tDeviatoricStress, tConfigurationGradient, tCellVolume);
            tThermalFluxDivergence(aCellOrdinal, aResult, tThermalFlux, tConfigurationGradient, tCellVolume, -1.0);
            tPressureDivergence (aCellOrdinal, aResult, tPressure, tConfigurationGradient, tCellVolume);
            tStabilizedDivergence (aCellOrdinal, aResult, tStabilization, tConfigurationGradient, tCellVolume, -1.0);
            tProjectVolumeStrain (aCellOrdinal, tCellVolume, tBasisFunctions, tVolumeStrain, aResult);

            // prepare output data
            ControlT tPenalizedBulkModulus = tElasticPropertiesPenalty * tElasticBulkModulus;
            tComputeCauchyStress(aCellOrdinal, tPenalizedBulkModulus, tPenalizedShearModulus, tElasticStrain, tCauchyStress);
            tJ2PlasticityUtils.getPlasticMultiplierIncrement(aCellOrdinal, aCurrentLocalState, tPlasticMultiplier);
            tJ2PlasticityUtils.getAccumulatedPlasticStrain(aCellOrdinal, aCurrentLocalState, tAccumPlasticStrain);
            tJ2PlasticityUtils.getPlasticStrainTensor(aCellOrdinal, aCurrentLocalState, tPlasticStrain);
            tJ2PlasticityUtils.getBackstressTensor(aCellOrdinal, aCurrentLocalState, tBackStress);
        }, "stabilized infinitesimal strain thermoplasticity residual");

        this->addBodyForces(aCurrentGlobalState, aControls, aConfig, aResult);

        // set current output data
        this->computePrincipalStresses(aCurrentGlobalState, aCurrentLocalState, aControls, aConfig);
        this->outputData(tPlasticMultiplier, "plastic multiplier increment");
        this->outputData(tAccumPlasticStrain, "accumulated plastic strain");
        this->outputData(tDeviatoricStress, "deviatoric stress");
        this->outputData(tElasticStrain, "elastic strain");
        this->outputData(tPlasticStrain, "plastic strain");
        this->outputData(tCauchyStress, "cauchy stress");
        this->outputData(tBackStress, "backstress");
        this->outputData(tThermalFlux, "thermal flux");
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
                       Plato::Scalar aTimeStep = 0.0) override
    {
        // update SIMP penalty parameter
        auto tPreviousPenaltySIMP = mPenaltySIMP;
        auto tSuggestedPenaltySIMP = tPreviousPenaltySIMP + mAdditiveContinuationParam;
        mPenaltySIMP = tSuggestedPenaltySIMP >= mUpperBoundOnPenaltySIMP ? mUpperBoundOnPenaltySIMP : tSuggestedPenaltySIMP;
        std::ostringstream tMsg;
        tMsg << "Infinitesimal Strain Thermoplasticity Residual: New penalty parameter is set to '" << mPenaltySIMP
                << "'. Previous penalty parameter was '" << tPreviousPenaltySIMP << "'.\n";
        REPORT(tMsg.str().c_str())
    }
    /************************************************************************//**
     * \brief Evaluate the stabilized residual equation
     *
     * \param [in]     aCurrentGlobalState    current global state workset ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aPrevGlobalState       previous global state workset ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in]     aCurrentLocalState     current local state workset ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aPrevLocalState        previous local state workset ( i.e. state at the n-th minus one time interval (\f$ t^{n-1} \f$) )
     * \param [in]     aProjectedPressureGrad current pressure gradient workset ( i.e. state at the n-th time interval (\f$ t^{n} \f$) )
     * \param [in]     aControls               design variables workset
     * \param [in]     aConfig                configuration workset
     * \param [in/out] aResult                residual workset
     * \param [in]     aTimeStep              current time step (i.e. \f$ \Delta{t}^{n} \f$), default = 0.0
     *
    ****************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                          & aSpatialModel,
        const Plato::ScalarMultiVectorT <GlobalStateT>     & aCurrentGlobalState,
        const Plato::ScalarMultiVectorT <PrevGlobalStateT> & aPrevGlobalState,
        const Plato::ScalarMultiVectorT <LocalStateT>      & aCurrentLocalState,
        const Plato::ScalarMultiVectorT <PrevLocalStateT>  & aPrevLocalState,
        const Plato::ScalarMultiVectorT <NodeStateT>       & aProjectedPressureGrad,
        const Plato::ScalarMultiVectorT <ControlT>         & aControls,
        const Plato::ScalarArray3DT     <ConfigT>          & aConfig,
        const Plato::ScalarMultiVectorT <ResultT>          & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) override
    {
        this->addExternalForces(aSpatialModel, aCurrentGlobalState, aControls, aConfig, aResult);
    }
};
// class InfinitesimalStrainThermoPlasticityResidual

}
// namespace Plato

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_INC_VMS(Plato::InfinitesimalStrainThermoPlasticityResidual, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_INC_VMS(Plato::InfinitesimalStrainThermoPlasticityResidual, Plato::SimplexThermoPlasticity, 3)
#endif
/*
 * BasicLocalScalarFunction.hpp
 *
 *  Created on: Mar 1, 2020
 */

#pragma once

#include "BLAS2.hpp"
#include "WorksetBase.hpp"
#include "SimplexFadTypes.hpp"
#include "LocalScalarFunctionInc.hpp"
#include "InfinitesimalStrainPlasticity.hpp"
#include "AbstractLocalScalarFunctionInc.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Interface for the evaluation of path-dependent scalar functions,
 * including sensitivities, of the form:
 *
 *            \f$ \alpha * F(z,u_{i},u_{i-1},c_i,c_{i-1}) \f$,
 *
 * where \f$ z \f$ denotes the control variables, \f$ u \f$ are the global states
 * and \f$ c \f$ are the local states.  The \f$ i \f$ index denotes the time step
 * index; thus, time steps \f$ i \f$ and \f$ i-1 \f$ are the current and previous
 * time steps, respectively.
*******************************************************************************/
template<typename PhysicsT>
class BasicLocalScalarFunction : public Plato::LocalScalarFunctionInc
{
// private member data
private:
    using Residual        = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;       /*!< automatic differentiation (AD) type for the residual */
    using GradientX       = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX;      /*!< AD type for the configuration */
    using GradientZ       = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ;      /*!< AD type for the controls */
    using LocalJacobian   = typename Plato::Evaluation<typename PhysicsT::SimplexT>::LocalJacobian;  /*!< AD type for the current local states */
    using LocalJacobianP  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::LocalJacobianP; /*!< AD type for the previous local states */
    using GlobalJacobian  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian;       /*!< AD type for the current global states */
    using GlobalJacobianP = typename Plato::Evaluation<typename PhysicsT::SimplexT>::JacobianP;      /*!< AD type for the previous global states */

    static constexpr auto mNumControl = PhysicsT::SimplexT::mNumControl;                   /*!< number of control fields, i.e. vectors, number of materials */
    static constexpr auto mNumSpatialDims = PhysicsT::SimplexT::mNumSpatialDims;           /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell = PhysicsT::SimplexT::mNumNodesPerCell;         /*!< number of nodes per cell (i.e. element) */
    static constexpr auto mNumGlobalDofsPerNode = PhysicsT::SimplexT::mNumDofsPerNode;     /*!< number of global degrees of freedom per node */
    static constexpr auto mNumGlobalDofsPerCell = PhysicsT::SimplexT::mNumDofsPerCell;     /*!< number of global degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumLocalDofsPerCell = PhysicsT::SimplexT::mNumLocalDofsPerCell; /*!< number of local degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumNodeStatePerNode = PhysicsT::SimplexT::mNumNodeStatePerNode; /*!< number of pressure gradient degrees of freedom per node */
    static constexpr auto mNumNodeStatePerCell = PhysicsT::SimplexT::mNumNodeStatePerCell; /*!< number of pressure gradient degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;      /*!< number of configuration (i.e. coordinates) degrees of freedom per cell (i.e. element) */

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap;                   /*!< output data map */
    Plato::Scalar mMultiplier;                  /*!< scalar function multipliers */
    std::string mFunctionName;                  /*!< user defined function name */
    Plato::WorksetBase<PhysicsT> mWorksetBase;  /*!< assembly routine interface */

    using ValueFunction           = std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<Residual>>;
    using GradientZFunction       = std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<GradientZ>>;
    using GradientXFunction       = std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<GradientX>>;
    using LocalJacobianPFunction  = std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<LocalJacobianP>>;
    using GlobalJacobianPFunction = std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<GlobalJacobianP>>;
    using LocalJacobianFunction   = std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<LocalJacobian>>;
    using GlobalJacobianFunction  = std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<GlobalJacobian>>;

    std::map<std::string, ValueFunction>           mValueFunctions;
    std::map<std::string, GradientZFunction>       mGradientZFunctions;
    std::map<std::string, GradientXFunction>       mGradientXFunctions;
    std::map<std::string, LocalJacobianPFunction>  mLocalJacobianPFunctions;
    std::map<std::string, GlobalJacobianPFunction> mGlobalJacobianPFunctions;
    std::map<std::string, LocalJacobianFunction>   mLocalJacobianFunctions;
    std::map<std::string, GlobalJacobianFunction>  mGlobalJacobianFunctions;

// public access functions
public:
    /******************************************************************************//**
     * /brief Path-dependent physics-based scalar function constructor
     * /param [in] aSpatialDomain Plato Analyze spatial domain
     * /param [in] aDataMap Plato Analyze output data map
     * /param [in] aInputParams input parameters database
     * /param [in] aName user defined function name
    **********************************************************************************/
    BasicLocalScalarFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) :
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mMultiplier   (1.0),
        mFunctionName (aName),
        mWorksetBase  (aSpatialModel.Mesh)
    {
        this->initialize(aInputParams);
    }

    /******************************************************************************//**
     * /brief Path-dependent physics-based scalar function constructor
     * /param [in] aMesh mesh database
     * /param [in] aDataMap PLATO Analyze output data map
     * /param [in] aName user defined function name
    **********************************************************************************/
    BasicLocalScalarFunction(
        const Plato::SpatialModel  & aSpatialModel,
              Plato::DataMap       & aDataMap,
              std::string            aName = ""
    ) :
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mMultiplier   (1.0),
        mFunctionName (aName),
        mWorksetBase  (aSpatialModel.Mesh)
    {
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    virtual ~BasicLocalScalarFunction(){}

    /***************************************************************************//**
     * \brief Return scalar function name
     * \return user defined function name
    *******************************************************************************/
    void setScalarFunctionMultiplier(const Plato::Scalar & aInput)
    {
        mMultiplier = aInput;
    }

    /***************************************************************************//**
     * \brief Return scalar function name
     * \return user defined function name
    *******************************************************************************/
    decltype(mFunctionName) name() const override
    {
        return (mFunctionName);
    }

//TODO what are these for?
#ifdef COMPILE_DEAD_CODE
    /******************************************************************************//**
     * \brief Append path-dependent scalar function interface for value function
     * \param [in] path-dependent scalar function interface
    **********************************************************************************/
    void appendCriterionValue(const std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<Residual>>& aInput)
    {
        mScalarFuncValue = aInput;
    }

    /******************************************************************************//**
     * \brief Append path-dependent scalar function interface for partial derivative
     * with respect to current global states
     * \param [in] path-dependent scalar function interface
    **********************************************************************************/
    void appendPartialCurrentGlobalState(const std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<GlobalJacobian>>& aInput)
    {
        mPartialCurrentGlobalStates = aInput;
    }

    /******************************************************************************//**
     * \brief Append path-dependent scalar function interface for partial derivative
     * with respect to previous global states
     * \param [in] path-dependent scalar function interface
    **********************************************************************************/
    void appendPartialPreviousGlobalState(const std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<GlobalJacobianP>>& aInput)
    {
        mPartialPrevGlobalStates = aInput;
    }

    /******************************************************************************//**
     * \brief Append path-dependent scalar function interface for partial derivative
     * with respect to current local states
     * \param [in] path-dependent scalar function interface
    **********************************************************************************/
    void appendPartialCurrentLocalStates(const std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<LocalJacobian>>& aInput)
    {
        mPartialCurrentLocalStates = aInput;
    }

    /******************************************************************************//**
     * \brief Append path-dependent scalar function interface for partial derivative
     * with respect to previous local states
     * \param [in] path-dependent scalar function interface
    **********************************************************************************/
    void appendPartialPreviousLocalStates(const std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<LocalJacobianP>>& aInput)
    {
        mPartialPrevLocalStates = aInput;
    }

    /******************************************************************************//**
     * \brief Append path-dependent scalar function interface for partial derivative
     * with respect to controls
     * \param [in] path-dependent scalar function interface
    **********************************************************************************/
    void appendPartialControls(const std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<GradientZ>>& aInput)
    {
        mPartialControls = aInput;
    }

    /******************************************************************************//**
     * \brief Append path-dependent scalar function interface for partial derivative
     * with respect to configuration
     * \param [in] path-dependent scalar function interface
    **********************************************************************************/
    void appendPartialConfiguration(const std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<GradientX>>& aInput)
    {
        mPartialConfiguration = aInput;
    }
#endif

    /***************************************************************************//**
     * \brief Return function value
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeStep             current time step increment
     * \return function value
    *******************************************************************************/
    Plato::Scalar
    value(
        const Plato::ScalarVector & aCurrentGlobalState,
        const Plato::ScalarVector & aPreviousGlobalState,
        const Plato::ScalarVector & aCurrentLocalState,
        const Plato::ScalarVector & aPreviousLocalState,
        const Plato::ScalarVector & aControls,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        Plato::Scalar tCriterionValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // set workset of current global states
            using CurrentGlobalStateScalar = typename Residual::StateScalarType;
            Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS, tDomain);

            // set workset of previous global states
            using PreviousGlobalStateScalar = typename Residual::PrevStateScalarType;
            Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS, tDomain);

            // set workset of current local states
            using CurrentLocalStateScalar = typename Residual::LocalStateScalarType;
            Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS, tDomain);

            // set workset of previous local states
            using PreviousLocalStateScalar = typename Residual::PrevLocalStateScalarType;
            Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS, tDomain);

            // workset control
            using ControlScalar = typename Residual::ControlScalarType;
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControls, tControlWS, tDomain);

            // workset config
            using ConfigScalar = typename Residual::ConfigScalarType;
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // create result view
            using ResultScalar = typename Residual::ResultScalarType;
            Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

            // evaluate function
            mValueFunctions.at(tName)->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                                tCurrentLocalStateWS, tPreviousLocalStateWS,
                                                tControlWS, tConfigWS, tResultWS, aTimeStep);

            // sum across elements
            tCriterionValue += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
        }
        tCriterionValue = mMultiplier * tCriterionValue;

        auto tName = mSpatialModel.Domains.front().getDomainName();
        mDataMap.mScalarValues[mValueFunctions.at(tName)->getName()] = tCriterionValue;

        return (tCriterionValue);
    }

    /***************************************************************************//**
     * \brief Return workset with partial derivative wrt design variables
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeStep             current time step increment
     * \return workset with partial workset derivative wrt design variables
    *******************************************************************************/
    Plato::ScalarMultiVector
    gradient_z(
        const Plato::ScalarVector & aCurrentGlobalState,
        const Plato::ScalarVector & aPreviousGlobalState,
        const Plato::ScalarVector & aCurrentLocalState,
        const Plato::ScalarVector & aPreviousLocalState,
        const Plato::ScalarVector & aControls,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        auto tTotalNumCells = mWorksetBase.numCells();
        Plato::ScalarMultiVector tCriterionPartialWrtControl("criterion partial wrt control", tTotalNumCells, mNumNodesPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // set workset of current global states
            using CurrentGlobalStateScalar = typename GradientZ::StateScalarType;
            Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS, tDomain);

            // set workset of previous global states
            using PreviousGlobalStateScalar = typename GradientZ::PrevStateScalarType;
            Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS, tDomain);

            // set workset of current local states
            using CurrentLocalStateScalar = typename GradientZ::LocalStateScalarType;
            Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS, tDomain);

            // set workset of previous local states
            using PreviousLocalStateScalar = typename GradientZ::PrevLocalStateScalarType;
            Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS, tDomain);

            // workset control
            using ControlScalar = typename GradientZ::ControlScalarType;
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControls, tControlWS, tDomain);

            // workset config
            using ConfigScalar = typename GradientZ::ConfigScalarType;
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // create result view
            using ResultScalar = typename GradientZ::ResultScalarType;
            Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

            // evaluate function
            mGradientZFunctions.at(tName)->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                                    tCurrentLocalStateWS, tPreviousLocalStateWS,
                                                    tControlWS, tConfigWS, tResultWS, aTimeStep);

            // convert AD types to POD types
            Plato::transform_ad_type_to_pod_2Dview<mNumNodesPerCell>(tDomain, tResultWS, tCriterionPartialWrtControl);
        }
        Plato::blas2::scale(mMultiplier, tCriterionPartialWrtControl);

        return tCriterionPartialWrtControl;
    }

    /***************************************************************************//**
     * \brief Return workset with partial derivative wrt current global states
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeStep             current time step increment
     * \return workset with partial derivative wrt current global states
    *******************************************************************************/
    Plato::ScalarMultiVector gradient_u(
        const Plato::ScalarVector & aCurrentGlobalState,
        const Plato::ScalarVector & aPreviousGlobalState,
        const Plato::ScalarVector & aCurrentLocalState,
        const Plato::ScalarVector & aPreviousLocalState,
        const Plato::ScalarVector & aControls,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        auto tTotalNumCells = mWorksetBase.numCells();
        Plato::ScalarMultiVector tCriterionPartialWrtGlobalStates("criterion partial wrt global states", tTotalNumCells, mNumGlobalDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // set workset of current global states
            using CurrentGlobalStateScalar = typename GlobalJacobian::StateScalarType;
            Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS, tDomain);

            // set workset of previous global states
            using PreviousGlobalStateScalar = typename GlobalJacobian::PrevStateScalarType;
            Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS, tDomain);

            // set workset of current local states
            using CurrentLocalStateScalar = typename GlobalJacobian::LocalStateScalarType;
            Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS, tDomain);

            // set workset of previous local states
            using PreviousLocalStateScalar = typename GlobalJacobian::PrevLocalStateScalarType;
            Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS, tDomain);

            // workset control
            using ControlScalar = typename GlobalJacobian::ControlScalarType;
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControls, tControlWS, tDomain);

            // workset config
            using ConfigScalar = typename GlobalJacobian::ConfigScalarType;
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // create result view
            using ResultScalar = typename GlobalJacobian::ResultScalarType;
            Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

            // evaluate function
            mGlobalJacobianFunctions.at(tName)->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                                         tCurrentLocalStateWS, tPreviousLocalStateWS,
                                                         tControlWS, tConfigWS, tResultWS, aTimeStep);

            // convert AD types to POD types
            Plato::transform_ad_type_to_pod_2Dview<mNumGlobalDofsPerCell>(tDomain, tResultWS, tCriterionPartialWrtGlobalStates);
        }
        Plato::blas2::scale(mMultiplier, tCriterionPartialWrtGlobalStates);

        return (tCriterionPartialWrtGlobalStates);
    }

    /***************************************************************************//**
     * \brief Return workset with partial derivative wrt previous global states
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeStep             current time step increment
     * \return workset with partial derivative wrt previous global states
    *******************************************************************************/
    Plato::ScalarMultiVector
    gradient_up(
        const Plato::ScalarVector & aCurrentGlobalState,
        const Plato::ScalarVector & aPreviousGlobalState,
        const Plato::ScalarVector & aCurrentLocalState,
        const Plato::ScalarVector & aPreviousLocalState,
        const Plato::ScalarVector & aControls,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        auto tTotalNumCells = mWorksetBase.numCells();
        Plato::ScalarMultiVector tCriterionPartialWrtPrevGlobalState("partial wrt previous global states", tTotalNumCells, mNumGlobalDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // set workset of current global states
            using CurrentGlobalStateScalar = typename GlobalJacobianP::StateScalarType;
            Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS, tDomain);

            // set workset of previous global states
            using PreviousGlobalStateScalar = typename GlobalJacobianP::PrevStateScalarType;
            Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS, tDomain);

            // set workset of current local states
            using CurrentLocalStateScalar = typename GlobalJacobianP::LocalStateScalarType;
            Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS, tDomain);

            // set workset of previous local states
            using PreviousLocalStateScalar = typename GlobalJacobianP::PrevLocalStateScalarType;
            Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS, tDomain);

            // workset control
            using ControlScalar = typename GlobalJacobianP::ControlScalarType;
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControls, tControlWS, tDomain);

            // workset config
            using ConfigScalar = typename GlobalJacobianP::ConfigScalarType;
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // create result view
            using ResultScalar = typename GlobalJacobianP::ResultScalarType;
            Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

            // evaluate function
            mGlobalJacobianPFunctions.at(tName)->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                                          tCurrentLocalStateWS, tPreviousLocalStateWS,
                                                          tControlWS, tConfigWS, tResultWS, aTimeStep);

            // convert AD types to POD types
            Plato::transform_ad_type_to_pod_2Dview<mNumGlobalDofsPerCell>(tDomain, tResultWS, tCriterionPartialWrtPrevGlobalState);
        }
        Plato::blas2::scale(mMultiplier, tCriterionPartialWrtPrevGlobalState);

        return (tCriterionPartialWrtPrevGlobalState);
    }

    /***************************************************************************//**
     * \brief Return workset with partial derivative wrt current local states
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeStep             current time step increment
     * \return workset with partial derivative wrt current local states
    *******************************************************************************/
    Plato::ScalarMultiVector
    gradient_c(
        const Plato::ScalarVector & aCurrentGlobalState,
        const Plato::ScalarVector & aPreviousGlobalState,
        const Plato::ScalarVector & aCurrentLocalState,
        const Plato::ScalarVector & aPreviousLocalState,
        const Plato::ScalarVector & aControls,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        auto tTotalNumCells = mWorksetBase.numCells();
        Plato::ScalarMultiVector tCriterionPartialWrtLocalStates("criterion partial wrt local states", tTotalNumCells, mNumLocalDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // set workset of current global states
            using CurrentGlobalStateScalar = typename LocalJacobian::StateScalarType;
            Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS, tDomain);

            // set workset of previous global states
            using PreviousGlobalStateScalar = typename LocalJacobian::PrevStateScalarType;
            Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS, tDomain);

            // set workset of current local states
            using CurrentLocalStateScalar = typename LocalJacobian::LocalStateScalarType;
            Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS, tDomain);

            // set workset of previous local states
            using PreviousLocalStateScalar = typename LocalJacobian::PrevLocalStateScalarType;
            Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS, tDomain);

            // workset control
            using ControlScalar = typename LocalJacobian::ControlScalarType;
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControls, tControlWS, tDomain);

            // workset config
            using ConfigScalar = typename LocalJacobian::ConfigScalarType;
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // create result view
            using ResultScalar = typename LocalJacobian::ResultScalarType;
            Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

            // evaluate function
            mLocalJacobianFunctions.at(tName)->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                                        tCurrentLocalStateWS, tPreviousLocalStateWS,
                                                        tControlWS, tConfigWS, tResultWS, aTimeStep);

            // convert AD types to POD types
            Plato::transform_ad_type_to_pod_2Dview<mNumLocalDofsPerCell>(tDomain, tResultWS, tCriterionPartialWrtLocalStates);
        }
        Plato::blas2::scale(mMultiplier, tCriterionPartialWrtLocalStates);

        return tCriterionPartialWrtLocalStates;
    }

    /***************************************************************************//**
     * \brief Return workset with partial derivative wrt previous local states
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeStep             current time step increment
     * \return workset with partial derivative wrt previous local states
    *******************************************************************************/
    Plato::ScalarMultiVector
    gradient_cp(
        const Plato::ScalarVector & aCurrentGlobalState,
        const Plato::ScalarVector & aPreviousGlobalState,
        const Plato::ScalarVector & aCurrentLocalState,
        const Plato::ScalarVector & aPreviousLocalState,
        const Plato::ScalarVector & aControls,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        auto tTotalNumCells = mWorksetBase.numCells();
        Plato::ScalarMultiVector tCriterionPartialWrtPrevLocalStates("partial wrt previous local states", tTotalNumCells, mNumLocalDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // set workset of current global states
            using CurrentGlobalStateScalar = typename LocalJacobianP::StateScalarType;
            Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS, tDomain);

            // set workset of previous global states
            using PreviousGlobalStateScalar = typename LocalJacobianP::PrevStateScalarType;
            Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS, tDomain);

            // set workset of current local states
            using CurrentLocalStateScalar = typename LocalJacobianP::LocalStateScalarType;
            Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS, tDomain);

            // set workset of previous local states
            using PreviousLocalStateScalar = typename LocalJacobianP::PrevLocalStateScalarType;
            Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS, tDomain);

            // workset control
            using ControlScalar = typename LocalJacobianP::ControlScalarType;
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControls, tControlWS, tDomain);

            // workset config
            using ConfigScalar = typename LocalJacobianP::ConfigScalarType;
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // create result view
            using ResultScalar = typename LocalJacobianP::ResultScalarType;
            Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

            // evaluate function
            mLocalJacobianPFunctions.at(tName)->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                                         tCurrentLocalStateWS, tPreviousLocalStateWS,
                                                         tControlWS, tConfigWS, tResultWS, aTimeStep);

            // convert AD types to POD types
            Plato::transform_ad_type_to_pod_2Dview<mNumLocalDofsPerCell>(tDomain, tResultWS, tCriterionPartialWrtPrevLocalStates);
        }
        Plato::blas2::scale(mMultiplier, tCriterionPartialWrtPrevLocalStates);

        return tCriterionPartialWrtPrevLocalStates;
    }

    /***************************************************************************//**
     * \brief Return workset with partial derivative wrt configuration variables
     * \param [in] aCurrentGlobalState   global states at time step i (i.e. current)
     * \param [in] aPreviousGlobalState  global states at time step i-1 (i.e. previous)
     * \param [in] aCurrentLocalState    local states at time step i (i.e. current)
     * \param [in] aPreviousLocalState   local states at time step i-1 (i.e. previous)
     * \param [in] aControls             set of controls, i.e. design variables
     * \param [in] aTimeStep             current time step increment
     * \return workset with partial derivative wrt configuration variables
     *******************************************************************************/
    Plato::ScalarMultiVector
    gradient_x(
        const Plato::ScalarVector & aCurrentGlobalState,
        const Plato::ScalarVector & aPreviousGlobalState,
        const Plato::ScalarVector & aCurrentLocalState,
        const Plato::ScalarVector & aPreviousLocalState,
        const Plato::ScalarVector & aControls,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        auto tTotalNumCells = mWorksetBase.numCells();
        Plato::ScalarMultiVector tCriterionPartialWrtConfiguration("criterion partial wrt configuration", tTotalNumCells, mNumConfigDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // set workset of current global states
            using CurrentGlobalStateScalar = typename GradientX::StateScalarType;
            Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS, tDomain);

            // set workset of previous global states
            using PreviousGlobalStateScalar = typename GradientX::PrevStateScalarType;
            Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS, tDomain);

            // set workset of current local states
            using CurrentLocalStateScalar = typename GradientX::LocalStateScalarType;
            Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS, tDomain);

            // set workset of previous local states
            using PreviousLocalStateScalar = typename GradientX::PrevLocalStateScalarType;
            Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS, tDomain);

            // workset control
            using ControlScalar = typename GradientX::ControlScalarType;
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControls, tControlWS, tDomain);

            // workset config
            using ConfigScalar = typename GradientX::ConfigScalarType;
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // create result view
            using ResultScalar = typename GradientX::ResultScalarType;
            Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

            // evaluate function
            mGradientXFunctions.at(tName)->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                                    tCurrentLocalStateWS, tPreviousLocalStateWS,
                                                    tControlWS, tConfigWS, tResultWS, aTimeStep);

            // convert AD types to POD types
            Plato::transform_ad_type_to_pod_2Dview<mNumSpatialDims>(tDomain, tResultWS, tCriterionPartialWrtConfiguration);
        }
        Plato::blas2::scale(mMultiplier, tCriterionPartialWrtConfiguration);

        return tCriterionPartialWrtConfiguration;
    }

    /***************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aGlobalStates global states for all time steps
     * \param [in] aLocalStates  local states for all time steps
     * \param [in] aControls     current controls, i.e. design variables
     * \param [in] aTimeStep current time step increment
    *******************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aGlobalStates,
                       const Plato::ScalarMultiVector & aLocalStates,
                       const Plato::ScalarVector & aControls,
                       Plato::Scalar aTimeStep = 0.0) const override
    {
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();
            mValueFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls);
            mGradientZFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls);
            mGradientXFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls);
            mLocalJacobianPFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls);
            mGlobalJacobianPFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls);
            mLocalJacobianFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls);
            mGlobalJacobianFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls);
        }
    }

private:
    /******************************************************************************//**
     * \brief Initialization of Physics Scalar Function
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void
    initialize(
        Teuchos::ParameterList & aInputParams
    )
    {
        if(aInputParams.isSublist(mFunctionName) == false)
        {
            const auto tError = std::string("UNKNOWN USER DEFINED SCALAR FUNCTION SUBLIST '")
                    + mFunctionName + "'. USER DEFINED SCALAR FUNCTION SUBLIST '" + mFunctionName
                    + "' IS NOT DEFINED IN THE INPUT FILE.";
            THROWERR(tError)
        }

        auto tInputData = aInputParams.sublist(mFunctionName);

        auto tFunctionType = tInputData.get<std::string>("Scalar Function Type", "UNDEFINED");

        mMultiplier = tInputData.get<Plato::Scalar>("Multiplier", 1.0);

        typename PhysicsT::FunctionFactory tFactory;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mValueFunctions[tName]           = tFactory.template createLocalScalarFunctionInc<Residual>       (tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            mGradientXFunctions[tName]       = tFactory.template createLocalScalarFunctionInc<GradientX>      (tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            mGradientZFunctions[tName]       = tFactory.template createLocalScalarFunctionInc<GradientZ>      (tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            mLocalJacobianFunctions[tName]   = tFactory.template createLocalScalarFunctionInc<LocalJacobian>  (tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            mLocalJacobianPFunctions[tName]  = tFactory.template createLocalScalarFunctionInc<LocalJacobianP> (tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            mGlobalJacobianFunctions[tName]  = tFactory.template createLocalScalarFunctionInc<GlobalJacobian> (tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            mGlobalJacobianPFunctions[tName] = tFactory.template createLocalScalarFunctionInc<GlobalJacobianP>(tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
        }
    }
};
// class BasicLocalScalarFunction

}
// namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::BasicLocalScalarFunction<Plato::InfinitesimalStrainPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::BasicLocalScalarFunction<Plato::InfinitesimalStrainPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::BasicLocalScalarFunction<Plato::InfinitesimalStrainPlasticity<3>>;
#endif

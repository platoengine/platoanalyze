/*
 * BasicLocalScalarFunctionInc.hpp
 *
 *  Created on: Mar 1, 2020
 */

#pragma once

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
class BasicLocalScalarFunctionInc : public Plato::LocalScalarFunctionInc
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

    Plato::DataMap& mDataMap;                   /*!< output data map */
    Plato::Scalar mMultiplier;                  /*!< scalar function multipliers */
    std::string mFunctionName;                  /*!< user defined function name */
    Plato::WorksetBase<PhysicsT> mWorksetBase;  /*!< assembly routine interface */

    std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<Residual>>        mScalarFuncValue;            /*!< value function */
    std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<GradientZ>>       mPartialControls;            /*!< partial derivative wrt controls */
    std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<GradientX>>       mPartialConfiguration;       /*!< partial derivative wrt configuration */
    std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<LocalJacobianP>>  mPartialPrevLocalStates;     /*!< partial derivative wrt previous local states */
    std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<GlobalJacobianP>> mPartialPrevGlobalStates;    /*!< partial derivative wrt previous global states */
    std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<LocalJacobian>>   mPartialCurrentLocalStates;  /*!< partial derivative wrt current local states */
    std::shared_ptr<Plato::AbstractLocalScalarFunctionInc<GlobalJacobian>>  mPartialCurrentGlobalStates; /*!< partial derivative wrt current global states */

// public access functions
public:
    /******************************************************************************//**
     * /brief Path-dependent physics-based scalar function constructor
     * /param [in] aMesh mesh database
     * /param [in] aMeshSets side sets database
     * /param [in] aDataMap PLATO Analyze output data map
     * /param [in] aInputParams input parameters database
     * /param [in] aName user defined function name
    **********************************************************************************/
    BasicLocalScalarFunctionInc(Omega_h::Mesh& aMesh,
                                Omega_h::MeshSets& aMeshSets,
                                Plato::DataMap & aDataMap,
                                Teuchos::ParameterList& aInputParams,
                                std::string& aName) :
            mDataMap(aDataMap),
            mMultiplier(1.0),
            mFunctionName(aName),
            mWorksetBase(aMesh)
    {
        this->initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * /brief Path-dependent physics-based scalar function constructor
     * /param [in] aMesh mesh database
     * /param [in] aDataMap PLATO Analyze output data map
     * /param [in] aName user defined function name
    **********************************************************************************/
    BasicLocalScalarFunctionInc(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap, std::string aName = "") :
            mDataMap(aDataMap),
            mMultiplier(1.0),
            mFunctionName(aName),
            mWorksetBase(aMesh)
    {
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    virtual ~BasicLocalScalarFunctionInc(){}

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
    Plato::Scalar value(const Plato::ScalarVector & aCurrentGlobalState,
                        const Plato::ScalarVector & aPreviousGlobalState,
                        const Plato::ScalarVector & aCurrentLocalState,
                        const Plato::ScalarVector & aPreviousLocalState,
                        const Plato::ScalarVector & aControls,
                        Plato::Scalar aTimeStep = 0.0) const override
    {
        auto tNumCells = mScalarFuncValue->getMesh().nelems();

        // set workset of current global states
        using CurrentGlobalStateScalar = typename Residual::StateScalarType;
        Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // set workset of previous global states
        using PreviousGlobalStateScalar = typename Residual::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS);

        // set workset of current local states
        using CurrentLocalStateScalar = typename Residual::LocalStateScalarType;
        Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // set workset of previous local states
        using PreviousLocalStateScalar = typename Residual::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS);

        // workset control
        using ControlScalar = typename Residual::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // workset config
        using ConfigScalar = typename Residual::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // create result view
        using ResultScalar = typename Residual::ResultScalarType;
        Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

        // evaluate function
        mScalarFuncValue->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                   tCurrentLocalStateWS, tPreviousLocalStateWS,
                                   tControlWS, tConfigWS, tResultWS, aTimeStep);

        // sum across elements
        auto tCriterionValue = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
        tCriterionValue = mMultiplier * tCriterionValue;
        mDataMap.mScalarValues[mScalarFuncValue->getName()] = tCriterionValue;

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
    Plato::ScalarMultiVector gradient_z(const Plato::ScalarVector & aCurrentGlobalState,
                                        const Plato::ScalarVector & aPreviousGlobalState,
                                        const Plato::ScalarVector & aCurrentLocalState,
                                        const Plato::ScalarVector & aPreviousLocalState,
                                        const Plato::ScalarVector & aControls,
                                        Plato::Scalar aTimeStep = 0.0) const override
    {
        auto tNumCells = mScalarFuncValue->getMesh().nelems();

        // set workset of current global states
        using CurrentGlobalStateScalar = typename GradientZ::StateScalarType;
        Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // set workset of previous global states
        using PreviousGlobalStateScalar = typename GradientZ::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS);

        // set workset of current local states
        using CurrentLocalStateScalar = typename GradientZ::LocalStateScalarType;
        Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // set workset of previous local states
        using PreviousLocalStateScalar = typename GradientZ::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS);

        // workset control
        using ControlScalar = typename GradientZ::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // workset config
        using ConfigScalar = typename GradientZ::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // create result view
        using ResultScalar = typename GradientZ::ResultScalarType;
        Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

        // evaluate function
        mPartialControls->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                   tCurrentLocalStateWS, tPreviousLocalStateWS,
                                   tControlWS, tConfigWS, tResultWS, aTimeStep);

        // convert AD types to POD types
        Plato::ScalarMultiVector tCriterionPartialWrtControl("criterion partial wrt control", tNumCells, mNumNodesPerCell);
        Plato::transform_ad_type_to_pod_2Dview<mNumNodesPerCell>(tResultWS, tCriterionPartialWrtControl);
        Plato::scale_array_2D(mMultiplier, tCriterionPartialWrtControl);

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
    Plato::ScalarMultiVector gradient_u(const Plato::ScalarVector & aCurrentGlobalState,
                                        const Plato::ScalarVector & aPreviousGlobalState,
                                        const Plato::ScalarVector & aCurrentLocalState,
                                        const Plato::ScalarVector & aPreviousLocalState,
                                        const Plato::ScalarVector & aControls,
                                        Plato::Scalar aTimeStep = 0.0) const override
    {
        auto tNumCells = mScalarFuncValue->getMesh().nelems();

        // set workset of current global states
        using CurrentGlobalStateScalar = typename GlobalJacobian::StateScalarType;
        Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // set workset of previous global states
        using PreviousGlobalStateScalar = typename GlobalJacobian::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS);

        // set workset of current local states
        using CurrentLocalStateScalar = typename GlobalJacobian::LocalStateScalarType;
        Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // set workset of previous local states
        using PreviousLocalStateScalar = typename GlobalJacobian::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS);

        // workset control
        using ControlScalar = typename GlobalJacobian::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // workset config
        using ConfigScalar = typename GlobalJacobian::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // create result view
        using ResultScalar = typename GlobalJacobian::ResultScalarType;
        Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

        // evaluate function
        mPartialCurrentGlobalStates->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                              tCurrentLocalStateWS, tPreviousLocalStateWS,
                                              tControlWS, tConfigWS, tResultWS, aTimeStep);

        // convert AD types to POD types
        Plato::ScalarMultiVector tCriterionPartialWrtGlobalStates("criterion partial wrt global states", tNumCells, mNumGlobalDofsPerCell);
        Plato::transform_ad_type_to_pod_2Dview<mNumGlobalDofsPerCell>(tResultWS, tCriterionPartialWrtGlobalStates);
        Plato::scale_array_2D(mMultiplier, tCriterionPartialWrtGlobalStates);

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
    Plato::ScalarMultiVector gradient_up(const Plato::ScalarVector & aCurrentGlobalState,
                                         const Plato::ScalarVector & aPreviousGlobalState,
                                         const Plato::ScalarVector & aCurrentLocalState,
                                         const Plato::ScalarVector & aPreviousLocalState,
                                         const Plato::ScalarVector & aControls,
                                         Plato::Scalar aTimeStep = 0.0) const override
    {
        auto tNumCells = mScalarFuncValue->getMesh().nelems();

        // set workset of current global states
        using CurrentGlobalStateScalar = typename GlobalJacobianP::StateScalarType;
        Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // set workset of previous global states
        using PreviousGlobalStateScalar = typename GlobalJacobianP::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS);

        // set workset of current local states
        using CurrentLocalStateScalar = typename GlobalJacobianP::LocalStateScalarType;
        Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // set workset of previous local states
        using PreviousLocalStateScalar = typename GlobalJacobianP::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS);

        // workset control
        using ControlScalar = typename GlobalJacobianP::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // workset config
        using ConfigScalar = typename GlobalJacobianP::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // create result view
        using ResultScalar = typename GlobalJacobianP::ResultScalarType;
        Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

        // evaluate function
        mPartialPrevGlobalStates->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                           tCurrentLocalStateWS, tPreviousLocalStateWS,
                                           tControlWS, tConfigWS, tResultWS, aTimeStep);

        // convert AD types to POD types
        Plato::ScalarMultiVector tCriterionPartialWrtPrevGlobalState("partial wrt previous global states", tNumCells, mNumGlobalDofsPerCell);
        Plato::transform_ad_type_to_pod_2Dview<mNumGlobalDofsPerCell>(tResultWS, tCriterionPartialWrtPrevGlobalState);
        Plato::scale_array_2D(mMultiplier, tCriterionPartialWrtPrevGlobalState);

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
    Plato::ScalarMultiVector gradient_c(const Plato::ScalarVector & aCurrentGlobalState,
                                        const Plato::ScalarVector & aPreviousGlobalState,
                                        const Plato::ScalarVector & aCurrentLocalState,
                                        const Plato::ScalarVector & aPreviousLocalState,
                                        const Plato::ScalarVector & aControls,
                                        Plato::Scalar aTimeStep = 0.0) const override
    {
        auto tNumCells = mScalarFuncValue->getMesh().nelems();

        // set workset of current global states
        using CurrentGlobalStateScalar = typename LocalJacobian::StateScalarType;
        Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // set workset of previous global states
        using PreviousGlobalStateScalar = typename LocalJacobian::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS);

        // set workset of current local states
        using CurrentLocalStateScalar = typename LocalJacobian::LocalStateScalarType;
        Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // set workset of previous local states
        using PreviousLocalStateScalar = typename LocalJacobian::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS);

        // workset control
        using ControlScalar = typename LocalJacobian::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // workset config
        using ConfigScalar = typename LocalJacobian::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // create result view
        using ResultScalar = typename LocalJacobian::ResultScalarType;
        Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

        // evaluate function
        mPartialCurrentLocalStates->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                             tCurrentLocalStateWS, tPreviousLocalStateWS,
                                             tControlWS, tConfigWS, tResultWS, aTimeStep);

        // convert AD types to POD types
        Plato::ScalarMultiVector tCriterionPartialWrtLocalStates("criterion partial wrt local states", tNumCells, mNumLocalDofsPerCell);
        Plato::transform_ad_type_to_pod_2Dview<mNumLocalDofsPerCell>(tResultWS, tCriterionPartialWrtLocalStates);
        Plato::scale_array_2D(mMultiplier, tCriterionPartialWrtLocalStates);

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
    Plato::ScalarMultiVector gradient_cp(const Plato::ScalarVector & aCurrentGlobalState,
                                         const Plato::ScalarVector & aPreviousGlobalState,
                                         const Plato::ScalarVector & aCurrentLocalState,
                                         const Plato::ScalarVector & aPreviousLocalState,
                                         const Plato::ScalarVector & aControls,
                                         Plato::Scalar aTimeStep = 0.0) const override
    {
        auto tNumCells = mScalarFuncValue->getMesh().nelems();

        // set workset of current global states
        using CurrentGlobalStateScalar = typename LocalJacobianP::StateScalarType;
        Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // set workset of previous global states
        using PreviousGlobalStateScalar = typename LocalJacobianP::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS);

        // set workset of current local states
        using CurrentLocalStateScalar = typename LocalJacobianP::LocalStateScalarType;
        Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // set workset of previous local states
        using PreviousLocalStateScalar = typename LocalJacobianP::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS);

        // workset control
        using ControlScalar = typename LocalJacobianP::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // workset config
        using ConfigScalar = typename LocalJacobianP::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // create result view
        using ResultScalar = typename LocalJacobianP::ResultScalarType;
        Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

        // evaluate function
        mPartialPrevLocalStates->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                          tCurrentLocalStateWS, tPreviousLocalStateWS,
                                          tControlWS, tConfigWS, tResultWS, aTimeStep);

        // convert AD types to POD types
        Plato::ScalarMultiVector tCriterionPartialWrtPrevLocalStates("partial wrt previous local states", tNumCells, mNumLocalDofsPerCell);
        Plato::transform_ad_type_to_pod_2Dview<mNumLocalDofsPerCell>(tResultWS, tCriterionPartialWrtPrevLocalStates);
        Plato::scale_array_2D(mMultiplier, tCriterionPartialWrtPrevLocalStates);

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
    Plato::ScalarMultiVector gradient_x(const Plato::ScalarVector & aCurrentGlobalState,
                                        const Plato::ScalarVector & aPreviousGlobalState,
                                        const Plato::ScalarVector & aCurrentLocalState,
                                        const Plato::ScalarVector & aPreviousLocalState,
                                        const Plato::ScalarVector & aControls,
                                        Plato::Scalar aTimeStep = 0.0) const override
    {
        auto tNumCells = mScalarFuncValue->getMesh().nelems();

        // set workset of current global states
        using CurrentGlobalStateScalar = typename GradientX::StateScalarType;
        Plato::ScalarMultiVectorT<CurrentGlobalStateScalar> tCurrentGlobalStateWS("current global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // set workset of previous global states
        using PreviousGlobalStateScalar = typename GradientX::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PreviousGlobalStateScalar> tPreviousGlobalStateWS("previous global state workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPreviousGlobalState, tPreviousGlobalStateWS);

        // set workset of current local states
        using CurrentLocalStateScalar = typename GradientX::LocalStateScalarType;
        Plato::ScalarMultiVectorT<CurrentLocalStateScalar> tCurrentLocalStateWS("current local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // set workset of previous local states
        using PreviousLocalStateScalar = typename GradientX::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PreviousLocalStateScalar> tPreviousLocalStateWS("previous local state workset", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPreviousLocalState, tPreviousLocalStateWS);

        // workset control
        using ControlScalar = typename GradientX::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // workset config
        using ConfigScalar = typename GradientX::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // create result view
        using ResultScalar = typename GradientX::ResultScalarType;
        Plato::ScalarVectorT<ResultScalar> tResultWS("result workset", tNumCells);

        // evaluate function
        mPartialConfiguration->evaluate(tCurrentGlobalStateWS, tPreviousGlobalStateWS,
                                        tCurrentLocalStateWS, tPreviousLocalStateWS,
                                        tControlWS, tConfigWS, tResultWS, aTimeStep);

        // convert AD types to POD types
        Plato::ScalarMultiVector tCriterionPartialWrtConfiguration("criterion partial wrt configuration", tNumCells, mNumSpatialDims);
        Plato::transform_ad_type_to_pod_2Dview<mNumSpatialDims>(tResultWS, tCriterionPartialWrtConfiguration);
        Plato::scale_array_2D(mMultiplier, tCriterionPartialWrtConfiguration);

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
        mScalarFuncValue->updateProblem(aGlobalStates, aLocalStates, aControls);
        mPartialControls->updateProblem(aGlobalStates, aLocalStates, aControls);
        mPartialConfiguration->updateProblem(aGlobalStates, aLocalStates, aControls);
        mPartialPrevLocalStates->updateProblem(aGlobalStates, aLocalStates, aControls);
        mPartialPrevGlobalStates->updateProblem(aGlobalStates, aLocalStates, aControls);
        mPartialCurrentLocalStates->updateProblem(aGlobalStates, aLocalStates, aControls);
        mPartialCurrentGlobalStates->updateProblem(aGlobalStates, aLocalStates, aControls);
    }

private:
    /******************************************************************************//**
     * \brief Initialization of Physics Scalar Function
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Omega_h::Mesh& aMesh,
                    Omega_h::MeshSets& aMeshSets,
                    Teuchos::ParameterList & aInputParams)
    {
        typename PhysicsT::FunctionFactory tFactory;
        auto tInputData = aInputParams.sublist(mFunctionName);

        // FunctionType must be a hard-coded function type in Plato Analyze (e.g. Volume)
        auto tFunctionType = tInputData.get<std::string>("Scalar Function Type", "");
        mMultiplier = tInputData.get<Plato::Scalar>("Multiplier", "1.0");

        mScalarFuncValue =
            tFactory.template createLocalScalarFunctionInc<Residual>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);

        mPartialConfiguration =
            tFactory.template createLocalScalarFunctionInc<GradientX>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);

        mPartialControls =
            tFactory.template createLocalScalarFunctionInc<GradientZ>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);

        mPartialCurrentLocalStates =
            tFactory.template createLocalScalarFunctionInc<LocalJacobian>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);

        mPartialPrevLocalStates =
            tFactory.template createLocalScalarFunctionInc<LocalJacobianP>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);

        mPartialCurrentGlobalStates =
            tFactory.template createLocalScalarFunctionInc<GlobalJacobian>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);

        mPartialPrevGlobalStates =
            tFactory.template createLocalScalarFunctionInc<GlobalJacobianP>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
    }
};
// class BasicLocalScalarFunctionInc

}
// namespace Plato

#ifdef PLATOANALYZE_2D
extern template class Plato::BasicLocalScalarFunctionInc<Plato::InfinitesimalStrainPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::BasicLocalScalarFunctionInc<Plato::InfinitesimalStrainPlasticity<3>>;
#endif

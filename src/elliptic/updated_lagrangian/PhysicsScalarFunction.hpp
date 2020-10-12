#pragma once

#include "PlatoUtilities.hpp"


#include <memory>
#include <cassert>
#include <vector>

#include "WorksetBase.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/updated_lagrangian/ScalarFunctionBase.hpp"
#include "elliptic/updated_lagrangian/AbstractScalarFunction.hpp"
#include "elliptic/updated_lagrangian/EllipticUpLagSimplexFadTypes.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

/******************************************************************************//**
 * \brief Physics scalar function class
 **********************************************************************************/
template<typename PhysicsT>
class PhysicsScalarFunction : public ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumLocalDofsPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode;
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims;
    using Plato::WorksetBase<PhysicsT>::mNumNodes;
    using Plato::WorksetBase<PhysicsT>::mNumCells;

    using Plato::WorksetBase<PhysicsT>::mGlobalStateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mLocalStateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mConfigEntryOrdinal;

    using Residual  = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradientU = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<typename PhysicsT::SimplexT>::Jacobian;
    using GradientC = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<typename PhysicsT::SimplexT>::GradientC;
    using GradientX = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    using ValueFunction     = std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<Residual>>;
    using GradientUFunction = std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<GradientU>>;
    using GradientCFunction = std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<GradientC>>;
    using GradientXFunction = std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractScalarFunction<GradientZ>>;

    std::map<std::string, ValueFunction>     mValueFunctions;
    std::map<std::string, GradientUFunction> mGradientUFunctions;
    std::map<std::string, GradientCFunction> mGradientCFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap;
    std::string mFunctionName;

private:
    /******************************************************************************//**
     * \brief Initialization of Physics Scalar Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void
    initialize(
        Teuchos::ParameterList & aProblemParams
    )
    {
        typename PhysicsT::FunctionFactory tFactory;

        auto tProblemDefault = aProblemParams.sublist("Criteria").sublist(mFunctionName);
        auto tFunctionType = tProblemDefault.get<std::string>("Scalar Function Type", "");


        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mValueFunctions[tName]     = tFactory.template createScalarFunction<Residual> 
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
            mGradientUFunctions[tName] = tFactory.template createScalarFunction<GradientU> 
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
            mGradientCFunctions[tName] = tFactory.template createScalarFunction<GradientC> 
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
            mGradientXFunctions[tName] = tFactory.template createScalarFunction<GradientX>
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
            mGradientZFunctions[tName] = tFactory.template createScalarFunction<GradientZ>
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
        }
    }

public:
    /******************************************************************************//**
     * \brief Primary physics scalar function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    PhysicsScalarFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aName
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Secondary physics scalar function constructor, used for unit testing
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
    **********************************************************************************/
    PhysicsScalarFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName ("Undefined Name")
    {
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const ValueFunction & aInput,
              std::string     aName
    )
    {
        mValueFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mValueFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the Jacobian automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const GradientUFunction & aInput,
              std::string         aName
    )
    {
        mGradientUFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mGradientUFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientC automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const GradientCFunction & aInput,
              std::string         aName
    )
    {
        mGradientCFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mGradientCFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientZ automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const GradientZFunction & aInput,
              std::string         aName
    )
    {
        mGradientZFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mGradientZFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Allocate scalar function using the GradientX automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const GradientXFunction & aInput,
              std::string         aName
    )
    {
        mGradientXFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mGradientXFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void
    updateProblem(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl
    ) const override
    {
// TODO
#ifdef TODO
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVector tStateWS("state workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS, tDomain);

            Plato::ScalarMultiVector tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            Plato::ScalarArray3D tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            mValueFunctions.at(tName)->updateProblem(tStateWS, tControlWS, tConfigWS);
            mGradientUFunctions.at(tName)->updateProblem(tStateWS, tControlWS, tConfigWS);
            mGradientZFunctions.at(tName)->updateProblem(tStateWS, tControlWS, tConfigWS);
            mGradientXFunctions.at(tName)->updateProblem(tStateWS, tControlWS, tConfigWS);
        }
#endif
    }

    /******************************************************************************//**
     * \brief Evaluate physics scalar function
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aLocalState local state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar physics function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(
        const Plato::Solution          & aSolution,
        const Plato::ScalarMultiVector & aLocalStates,
        const Plato::ScalarVector      & aControl,
              Plato::Scalar              aTimeStep = 0.0
    ) const override
    {
        using ConfigScalar      = typename Residual::ConfigScalarType;
        using GlobalStateScalar = typename Residual::GlobalStateScalarType;
        using LocalStateScalar  = typename Residual::LocalStateScalarType;
        using ControlScalar     = typename Residual::ControlScalarType;
        using ResultScalar      = typename Residual::ResultScalarType;

        Plato::Scalar tReturnVal(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // create result view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);
            mDataMap.scalarVectors[mValueFunctions.at(tName)->getName()] = tResult;

            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("global state workset", tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS ("local state workset",  tNumCells, mNumLocalDofsPerCell);


            auto tGlobalStates = aSolution.State;
            auto tNumSteps = tGlobalStates.extent(0);
            for( decltype(tNumSteps) tStepIndex=0; tStepIndex < tNumSteps; ++tStepIndex )
            {
                // workset global state
                //
                auto tGlobalState = Kokkos::subview(tGlobalStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tGlobalState, tGlobalStateWS, tDomain);

                // workset local state
                //
                auto tLocalState = Kokkos::subview(aLocalStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetLocalState(tLocalState, tLocalStateWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mValueFunctions.at(tName)->evaluate(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

                // sum across elements
                //
                tReturnVal += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResult);
            }
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mValueFunctions.at(tName)->postEvaluate(tReturnVal);

        return tReturnVal;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(
        const Plato::Solution          & aSolution,
        const Plato::ScalarMultiVector & aLocalStates,
        const Plato::ScalarVector      & aControl,
              Plato::Scalar              aTimeStep = 0.0
    ) const override
    {
        using ConfigScalar      = typename GradientX::ConfigScalarType;
        using GlobalStateScalar = typename GradientX::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientX::LocalStateScalarType;
        using ControlScalar     = typename GradientX::ControlScalarType;
        using ResultScalar      = typename GradientX::ResultScalarType;

        // create return view
        //
        Plato::ScalarVector tObjGradientX("objective gradient configuration", mNumSpatialDims * mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("global state workset", tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<LocalStateScalar>  tLocalStateWS ("local state workset",  tNumCells, mNumLocalDofsPerCell);

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);

            auto tGlobalStates = aSolution.State;

            auto tNumSteps = tGlobalStates.extent(0);
            for( decltype(tNumSteps) tStepIndex=0; tStepIndex < tNumSteps; ++tStepIndex )
            {
                // workset global state
                //
                auto tGlobalState = Kokkos::subview(tGlobalStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tGlobalState, tGlobalStateWS, tDomain);

                // workset local state
                //
                auto tLocalState = Kokkos::subview(aLocalStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetLocalState(tLocalState, tLocalStateWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mGradientXFunctions.at(tName)->evaluate(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

                // create and assemble to return view
                //
                Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                    (tDomain, mConfigEntryOrdinal, tResult, tObjGradientX);

                tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
            }
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientXFunctions.at(tName)->postEvaluate(tObjGradientX, tValue);

        return tObjGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(
        const Plato::Solution          & aSolution,
        const Plato::ScalarMultiVector & aLocalStates,
        const Plato::ScalarVector      & aControl,
              Plato::OrdinalType         aStepIndex,
              Plato::Scalar              aTimeStep = 0.0
    ) const override
    {
        using ConfigScalar      = typename GradientU::ConfigScalarType;
        using GlobalStateScalar = typename GradientU::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientU::LocalStateScalarType;
        using ControlScalar     = typename GradientU::ControlScalarType;
        using ResultScalar      = typename GradientU::ResultScalarType;

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientU("objective gradient state", mNumDofsPerNode * mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            auto tGlobalStates = aSolution.State;
            auto tGlobalState = Kokkos::subview(tGlobalStates, aStepIndex, Kokkos::ALL());

            // workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("sacado-ized state", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(tGlobalState, tGlobalStateWS, tDomain);

            // workset local state
            //
            auto tLocalState = Kokkos::subview(aLocalStates, aStepIndex, Kokkos::ALL());
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("sacado-ized state", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(tLocalState, tLocalStateWS, tDomain);

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // create return view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);

            // evaluate function
            //
            mGradientUFunctions.at(tName)->evaluate(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>
                (tDomain, mGlobalStateEntryOrdinal, tResult, tObjGradientU);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientUFunctions.at(tName)->postEvaluate(tObjGradientU, tValue);

        return tObjGradientU;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_c(
        const Plato::Solution          & aSolution,
        const Plato::ScalarMultiVector & aLocalStates,
        const Plato::ScalarVector      & aControl,
              Plato::OrdinalType         aStepIndex,
              Plato::Scalar              aTimeStep = 0.0
    ) const override
    {
        using ConfigScalar      = typename GradientC::ConfigScalarType;
        using GlobalStateScalar = typename GradientC::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientC::LocalStateScalarType;
        using ControlScalar     = typename GradientC::ControlScalarType;
        using ResultScalar      = typename GradientC::ResultScalarType;

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientC("objective gradient state", mNumDofsPerNode * mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
// TODO
#ifdef TODO
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            auto tGlobalStates = aSolution.State;
            auto tGlobalState = Kokkos::subview(tGlobalStates, aStepIndex, Kokkos::ALL());

            // workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("sacado-ized state", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(tGlobalState, tGlobalStateWS, tDomain);

            // workset local state
            //
            auto tLocalState = Kokkos::subview(aLocalStates, aStepIndex, Kokkos::ALL());
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("sacado-ized state", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(tLocalState, tLocalStateWS, tDomain);

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // create return view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);

            // evaluate function
            //
            mGradientCFunctions.at(tName)->evaluate(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

            Plato::flatten_cell_workset_fad<mNumLocalDofsPerCell>
                (tDomain, tResult, tObjGradientC);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
#endif
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientCFunctions.at(tName)->postEvaluate(tObjGradientC, tValue);

        return tObjGradientC;
    }
    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(
        const Plato::Solution          & aSolution,
        const Plato::ScalarMultiVector & aLocalStates,
        const Plato::ScalarVector      & aControl,
              Plato::Scalar              aTimeStep = 0.0
    ) const override
    {        
        using ConfigScalar      = typename GradientZ::ConfigScalarType;
        using GlobalStateScalar = typename GradientZ::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientZ::LocalStateScalarType;
        using ControlScalar     = typename GradientZ::ControlScalarType;
        using ResultScalar      = typename GradientZ::ResultScalarType;

        // create return vector
        //
        Plato::ScalarVector tObjGradientZ("objective gradient control", mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("global state workset", tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<LocalStateScalar>  tLocalStateWS ("local state workset",  tNumCells, mNumLocalDofsPerCell);

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // create result
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);
 
            auto tGlobalStates = aSolution.State;

            auto tNumSteps = tGlobalStates.extent(0);
            for( decltype(tNumSteps) tStepIndex=0; tStepIndex < tNumSteps; ++tStepIndex )
            {

                // global workset state
                //
                auto tGlobalState = Kokkos::subview(tGlobalStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tGlobalState, tGlobalStateWS, tDomain);

                // local workset state
                //
                auto tLocalState = Kokkos::subview(aLocalStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetLocalState(tLocalState, tLocalStateWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mGradientZFunctions.at(tName)->evaluate(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResult, aTimeStep);

                Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>
                    (tDomain, mControlEntryOrdinal, tResult, tObjGradientZ);

                tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
            }
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientZFunctions.at(tName)->postEvaluate(tObjGradientZ, tValue);

        return tObjGradientZ;
    }

    /******************************************************************************//**
     * \brief Set user defined function name
     * \param [in] function name
    **********************************************************************************/
    void setFunctionName(const std::string aFunctionName)
    {
        mFunctionName = aFunctionName;
    }

    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    decltype(mFunctionName) name() const
    {
        return mFunctionName;
    }
};
//class PhysicsScalarFunction

} // namespace UpdatedLagrangian

} // namespace Elliptic

} // namespace Plato

#include "Mechanics.hpp"
// TODO #include "Electromechanics.hpp"
// TODO #include "Thermomechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Elliptic::UpdatedLagrangian::PhysicsScalarFunction<::Plato::Elliptic::UpdatedLagrangian::Mechanics<1>>;
// TODO extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Electromechanics<1>>;
// TODO extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Elliptic::UpdatedLagrangian::PhysicsScalarFunction<::Plato::Elliptic::UpdatedLagrangian::Mechanics<2>>;
// TODO extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Electromechanics<2>>;
// TODO extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Elliptic::UpdatedLagrangian::PhysicsScalarFunction<::Plato::Elliptic::UpdatedLagrangian::Mechanics<3>>;
// TODO extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Electromechanics<3>>;
// TODO extern template class Plato::Elliptic::PhysicsScalarFunction<::Plato::Thermomechanics<3>>;
#endif

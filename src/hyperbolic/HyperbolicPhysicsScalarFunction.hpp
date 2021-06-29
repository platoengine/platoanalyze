#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

#include "WorksetBase.hpp"
#include "PlatoStaticsTypes.hpp"
#include "hyperbolic/HyperbolicSimplexFadTypes.hpp"
#include "hyperbolic/HyperbolicScalarFunctionBase.hpp"
#include "hyperbolic/HyperbolicAbstractScalarFunction.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************//**
 * \brief Physics scalar function inc class
 **********************************************************************************/
template<typename PhysicsT>
class PhysicsScalarFunction : public Plato::Hyperbolic::ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode;
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims;
    using Plato::WorksetBase<PhysicsT>::mNumControl;
    using Plato::WorksetBase<PhysicsT>::mNumNodes;
    using Plato::WorksetBase<PhysicsT>::mNumCells;

    using Plato::WorksetBase<PhysicsT>::mGlobalStateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mConfigEntryOrdinal;

    using Residual  = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradientU = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientU;
    using GradientV = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientV;
    using GradientA = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientA;
    using GradientX = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    using ValueFunction     = std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<Residual>>;
    using GradientUFunction = std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientU>>;
    using GradientVFunction = std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientV>>;
    using GradientAFunction = std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientA>>;
    using GradientXFunction = std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientZ>>;

    std::map<std::string, ValueFunction>     mValueFunctions;
    std::map<std::string, GradientUFunction> mGradientUFunctions;
    std::map<std::string, GradientVFunction> mGradientVFunctions;
    std::map<std::string, GradientAFunction> mGradientAFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap;

    std::string mFunctionName;

    /******************************************************************************//**
     * \brief Initialization of Hyperbolic Physics Scalar Function
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void
    initialize(
        Teuchos::ParameterList & aInputParams
    )
    {
        typename PhysicsT::FunctionFactory tFactory;

        auto tProblemDefault = aInputParams.sublist("Criteria").sublist(mFunctionName);
        auto tFunctionType = tProblemDefault.get<std::string>("Scalar Function Type", "");

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mValueFunctions[tName] =
                tFactory.template createScalarFunction<Residual>(
                    tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);

            mGradientUFunctions[tName] =
                tFactory.template createScalarFunction<GradientU>(
                    tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);

            mGradientVFunctions[tName] =
                tFactory.template createScalarFunction<GradientV>(
                    tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);

            mGradientAFunctions[tName] =
                tFactory.template createScalarFunction<GradientA>(
                    tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);

            mGradientXFunctions[tName] =
                tFactory.template createScalarFunction<GradientX>(
                    tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);

            mGradientZFunctions[tName] =
                tFactory.template createScalarFunction<GradientZ>(
                    tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
        }
    }

public:
    /******************************************************************************//**
     * \brief Primary physics scalar function inc constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    PhysicsScalarFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
              std::string            & aName
    ):
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aInputParams);
    }

    /******************************************************************************//**
     * \brief Secondary physics scalar function inc constructor, used for unit testing
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
     * \brief Evaluate physics scalar function
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar physics function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        using ConfigScalar      = typename Residual::ConfigScalarType;
        using StateScalar       = typename Residual::StateScalarType;
        using StateDotScalar    = typename Residual::StateDotScalarType;
        using StateDotDotScalar = typename Residual::StateDotDotScalarType;
        using ControlScalar     = typename Residual::ControlScalarType;
        using ResultScalar      = typename Residual::ResultScalarType;

        auto tStates = aSolution.get("State");
        auto tStatesDot = aSolution.get("StateDot");
        auto tStatesDotDot = aSolution.get("StateDotDot");

        ResultScalar tReturnVal(0.0);
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
            Plato::ScalarVectorT<ResultScalar> tResult("result", tNumCells);
            mDataMap.scalarVectors[mValueFunctions.at(tName)->getName()] = tResult;

            Plato::ScalarMultiVectorT<StateScalar>       tStateWS       ("state workset",         tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotScalar>    tStateDotWS    ("state dot workset",     tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS ("state dot dot workset", tNumCells, mNumDofsPerCell);

            auto tNumSteps = tStates.extent(0);
            auto tLastStepIndex = tNumSteps - 1;
            for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

                // workset state
                //
                auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS, tDomain);

                // workset state dot
                //
                auto tStateDot = Kokkos::subview(tStatesDot, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS, tDomain);

                // workset state dot dot
                //
                auto tStateDotDot = Kokkos::subview(tStatesDotDot, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tStateDotDot, tStateDotDotWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mValueFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

                // sum across elements
                //
                tReturnVal += Plato::local_result_sum<Plato::Scalar>( tNumCells, tResult);
            }
        }
        auto tName = mSpatialModel.Domains.front().getDomainName();
        mValueFunctions.at(tName)->postEvaluate( tReturnVal );

        return tReturnVal;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        using ConfigScalar      = typename GradientX::ConfigScalarType;
        using StateScalar       = typename GradientX::StateScalarType;
        using StateDotScalar    = typename GradientX::StateDotScalarType;
        using StateDotDotScalar = typename GradientX::StateDotDotScalarType;
        using ControlScalar     = typename GradientX::ControlScalarType;
        using ResultScalar      = typename GradientX::ResultScalarType;

        auto tStates = aSolution.get("State");
        auto tStatesDot = aSolution.get("StateDot");
        auto tStatesDotDot = aSolution.get("StateDotDot");

        // create return view
        //
        Plato::ScalarVector tObjGradientX("objective gradient configuration", mNumSpatialDims*mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVectorT<StateScalar>       tStateWS       ("state workset",         tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotScalar>    tStateDotWS    ("state dot workset",     tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS ("state dot dot workset", tNumCells, mNumDofsPerCell);

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
            Plato::ScalarVectorT<ResultScalar> tResult("result", tNumCells);

            auto tNumSteps = tStates.extent(0);
            auto tLastStepIndex = tNumSteps - 1;
            for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

                // workset state
                //
                auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS, tDomain);

                // workset state dot
                //
                auto tStateDot = Kokkos::subview(tStatesDot, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS, tDomain);

                // workset state dot dot
                //
                auto tStateDotDot = Kokkos::subview(tStatesDotDot, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tStateDotDot, tStateDotDotWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mGradientXFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

                Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                    (tDomain, mConfigEntryOrdinal, tResult, tObjGradientX);

                tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
            }
        }
        auto tName = mSpatialModel.Domains.front().getDomainName();
        mGradientXFunctions.at(tName)->postEvaluate( tObjGradientX, tValue );

        return tObjGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step
     * \param [in] aStepIndex step index
     * \return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep
    ) const override
    {
        using ConfigScalar      = typename GradientU::ConfigScalarType;
        using StateScalar       = typename GradientU::StateScalarType;
        using StateDotScalar    = typename GradientU::StateDotScalarType;
        using StateDotDotScalar = typename GradientU::StateDotDotScalarType;
        using ControlScalar     = typename GradientU::ControlScalarType;
        using ResultScalar      = typename GradientU::ResultScalarType;

        auto tStates = aSolution.get("State");
        auto tStatesDot = aSolution.get("StateDot");
        auto tStatesDotDot = aSolution.get("StateDotDot");

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientU("objective gradient state", mNumDofsPerNode * mNumNodes);

        Plato::Scalar tValue(0.0);
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

            // create return view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result", tNumCells);

            // workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", tNumCells, mNumDofsPerCell);
            auto tState = Kokkos::subview(tStates, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS, tDomain);

            // workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset", tNumCells, mNumDofsPerCell);
            auto tStateDot = Kokkos::subview(tStatesDot, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS, tDomain);

            // workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("state dot dot workset", tNumCells, mNumDofsPerCell);
            auto tStateDotDot = Kokkos::subview(tStatesDotDot, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDotDot, tStateDotDotWS, tDomain);

            // evaluate function
            //
            mGradientUFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>
                (tDomain, mGlobalStateEntryOrdinal, tResult, tObjGradientU);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
        }
        auto tName = mSpatialModel.Domains.front().getDomainName();
        mGradientUFunctions.at(tName)->postEvaluate( tObjGradientU, tValue );

        return tObjGradientU;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state dot variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step
     * \param [in] aStepIndex step index
     * \return 1D view with the gradient of the physics scalar function wrt the state dot variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_v(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep
    ) const override
    {
        using ConfigScalar      = typename GradientV::ConfigScalarType;
        using StateScalar       = typename GradientV::StateScalarType;
        using StateDotScalar    = typename GradientV::StateDotScalarType;
        using StateDotDotScalar = typename GradientV::StateDotDotScalarType;
        using ControlScalar     = typename GradientV::ControlScalarType;
        using ResultScalar      = typename GradientV::ResultScalarType;

        auto tStates = aSolution.get("State");
        auto tStatesDot = aSolution.get("StateDot");
        auto tStatesDotDot = aSolution.get("StateDotDot");

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientV("objective gradient state", mNumDofsPerNode*mNumNodes);

        Plato::Scalar tValue(0.0);
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

            // create return view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result", tNumCells);

            // workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", tNumCells, mNumDofsPerCell);
            auto tState = Kokkos::subview(tStates, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS, tDomain);

            // workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset", tNumCells, mNumDofsPerCell);
            auto tStateDot = Kokkos::subview(tStatesDot, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS, tDomain);

            // workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("state dot dot workset", tNumCells, mNumDofsPerCell);
            auto tStateDotDot = Kokkos::subview(tStatesDotDot, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDotDot, tStateDotDotWS, tDomain);

            // evaluate function
            //
            mGradientVFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>
                (tDomain, mGlobalStateEntryOrdinal, tResult, tObjGradientV);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
        }
        auto tName = mSpatialModel.Domains.front().getDomainName();
        mGradientVFunctions.at(tName)->postEvaluate( tObjGradientV, tValue );

        return tObjGradientV;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the state dot dot variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step
     * \param [in] aStepIndex step index
     * \return 1D view with the gradient of the physics scalar function wrt the state dot dot variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_a(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep
    ) const override
    {
        using ConfigScalar      = typename GradientA::ConfigScalarType;
        using StateScalar       = typename GradientA::StateScalarType;
        using StateDotScalar    = typename GradientA::StateDotScalarType;
        using StateDotDotScalar = typename GradientA::StateDotDotScalarType;
        using ControlScalar     = typename GradientA::ControlScalarType;
        using ResultScalar      = typename GradientA::ResultScalarType;

        auto tStates = aSolution.get("State");
        auto tStatesDot = aSolution.get("StateDot");
        auto tStatesDotDot = aSolution.get("StateDotDot");

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientA("objective gradient state", mNumDofsPerNode * mNumNodes);

        Plato::Scalar tValue(0.0);
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

            // create return view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result", tNumCells);

            // workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", tNumCells, mNumDofsPerCell);
            auto tState = Kokkos::subview(tStates, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS, tDomain);

            // workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset", tNumCells, mNumDofsPerCell);
            auto tStateDot = Kokkos::subview(tStatesDot, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS, tDomain);

            // workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("state dot dot workset", tNumCells, mNumDofsPerCell);
            auto tStateDotDot = Kokkos::subview(tStatesDotDot, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDotDot, tStateDotDotWS, tDomain);

            // evaluate function
            //
            mGradientAFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>
                (tDomain, mGlobalStateEntryOrdinal, tResult, tObjGradientA);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
        }
        auto tName = mSpatialModel.Domains.front().getDomainName();
        mGradientAFunctions.at(tName)->postEvaluate( tObjGradientA, tValue );

        return tObjGradientA;
    }


    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        using ConfigScalar      = typename GradientZ::ConfigScalarType;
        using StateScalar       = typename GradientZ::StateScalarType;
        using StateDotScalar    = typename GradientZ::StateDotScalarType;
        using StateDotDotScalar = typename GradientZ::StateDotDotScalarType;
        using ControlScalar     = typename GradientZ::ControlScalarType;
        using ResultScalar      = typename GradientZ::ResultScalarType;

        auto tStates = aSolution.get("State");
        auto tStatesDot = aSolution.get("StateDot");
        auto tStatesDotDot = aSolution.get("StateDotDot");

        // create return vector
        //
        Plato::ScalarVector tObjGradientZ("objective gradient control", mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVectorT<StateScalar>       tStateWS       ("state workset",         tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotScalar>    tStateDotWS    ("state dot workset",     tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS ("state dot dot workset", tNumCells, mNumDofsPerCell);

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
            Plato::ScalarVectorT<ResultScalar> tResult("result", tNumCells);

            auto tNumSteps = tStates.extent(0);
            auto tLastStepIndex = tNumSteps - 1;
            for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

                // workset state
                //
                auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS, tDomain);

                // workset state dot
                //
                auto tStateDot = Kokkos::subview(tStatesDot, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS, tDomain);

                // workset state dot dot
                //
                auto tStateDotDot = Kokkos::subview(tStatesDotDot, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tStateDotDot, tStateDotDotWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mGradientZFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

                Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>
                    (tDomain, mControlEntryOrdinal, tResult, tObjGradientZ);

                tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
            }
        }
        auto tName = mSpatialModel.Domains.front().getDomainName();
        mGradientZFunctions.at(tName)->postEvaluate( tObjGradientZ, tValue );

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
    std::string name() const
    {
        return mFunctionName;
    }
}; //class PhysicsScalarFunction

} //namespace Hyperbolic

} //namespace Plato

#include "HyperbolicMechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Hyperbolic::PhysicsScalarFunction<::Plato::Hyperbolic::Mechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Hyperbolic::PhysicsScalarFunction<::Plato::Hyperbolic::Mechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Hyperbolic::PhysicsScalarFunction<::Plato::Hyperbolic::Mechanics<3>>;
#endif

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
 * @brief Physics scalar function inc class
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

    std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<Residual>>  mScalarFunctionValue;
    std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientU>> mScalarFunctionGradientU;
    std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientV>> mScalarFunctionGradientV;
    std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientA>> mScalarFunctionGradientA;
    std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientX>> mScalarFunctionGradientX;
    std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientZ>> mScalarFunctionGradientZ;

    Plato::DataMap& mDataMap;

    std::string mFunctionName;

	/******************************************************************************//**
     * @brief Initialization of Hyperbolic Physics Scalar Function
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh,
                     Omega_h::MeshSets& aMeshSets,
                     Teuchos::ParameterList & aInputParams)
    {
        typename PhysicsT::FunctionFactory tFactory;

        auto tProblemDefault = aInputParams.sublist(mFunctionName);
        auto tFunctionType = tProblemDefault.get<std::string>("Scalar Function Type", "");

        mScalarFunctionValue =
            tFactory.template createScalarFunction<Residual>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);

        mScalarFunctionGradientU =
            tFactory.template createScalarFunction<GradientU>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);

        mScalarFunctionGradientV =
            tFactory.template createScalarFunction<GradientV>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);

        mScalarFunctionGradientA =
            tFactory.template createScalarFunction<GradientA>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);

        mScalarFunctionGradientX =
            tFactory.template createScalarFunction<GradientX>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);

        mScalarFunctionGradientZ =
            tFactory.template createScalarFunction<GradientZ>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
    }

public:
    /******************************************************************************//**
     * @brief Primary physics scalar function inc constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     * @param [in] aName user defined function name
    **********************************************************************************/
    PhysicsScalarFunction(
      Omega_h::Mesh          & aMesh,
      Omega_h::MeshSets      & aMeshSets,
      Plato::DataMap         & aDataMap,
      Teuchos::ParameterList & aInputParams,
      std::string            & aName
    ):
      Plato::WorksetBase<PhysicsT>(aMesh),
      mDataMap(aDataMap),
      mFunctionName(aName)
    {
        initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * @brief Secondary physics scalar function inc constructor, used for unit testing
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
    **********************************************************************************/
    PhysicsScalarFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
      Plato::WorksetBase<PhysicsT>(aMesh),
      mScalarFunctionValue(),
      mScalarFunctionGradientU(),
      mScalarFunctionGradientV(),
      mScalarFunctionGradientA(),
      mScalarFunctionGradientX(),
      mScalarFunctionGradientZ(),
      mDataMap(aDataMap),
      mFunctionName("Undefined Name")
    {
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the residual automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateValue(const std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<Residual>>& aInput)
    {
        mScalarFunctionValue = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientU automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientU(const std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientU>>& aInput)
    {
        mScalarFunctionGradientU = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientV automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientV(const std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientV>>& aInput)
    {
        mScalarFunctionGradientV = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientA automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientA(const std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientA>>& aInput)
    {
        mScalarFunctionGradientA = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientZ automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientZ(const std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientZ>>& aInput)
    {
        mScalarFunctionGradientZ = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientX automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientX(const std::shared_ptr<Plato::Hyperbolic::AbstractScalarFunction<GradientX>>& aInput)
    {
        mScalarFunctionGradientX = aInput;
    }

    /******************************************************************************//**
     * @brief Evaluate physics scalar function
     * @param [in] aSolution Plato::Solution composed of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return scalar physics function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(const Plato::Solution     & aSolution,
          const Plato::ScalarVector & aControl,
                Plato::Scalar         aTimeStep = 0.0) const override
    {
        using ConfigScalar      = typename Residual::ConfigScalarType;
        using StateScalar       = typename Residual::StateScalarType;
        using StateDotScalar    = typename Residual::StateDotScalarType;
        using StateDotDotScalar = typename Residual::StateDotDotScalarType;
        using ControlScalar     = typename Residual::ControlScalarType;
        using ResultScalar      = typename Residual::ResultScalarType;

        auto tStates = aSolution.State;
        auto tStatesDot = aSolution.StateDot;
        auto tStatesDotDot = aSolution.StateDotDot;

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",mNumCells,mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result",mNumCells);
        mDataMap.scalarVectors[mScalarFunctionValue->getName()] = tResult;

        Plato::ScalarMultiVectorT<StateScalar>       tStateWS       ("state workset",         mNumCells, mNumDofsPerCell);
        Plato::ScalarMultiVectorT<StateDotScalar>    tStateDotWS    ("state dot workset",     mNumCells, mNumDofsPerCell);
        Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS ("state dot dot workset", mNumCells, mNumDofsPerCell);

        ResultScalar tReturnVal(0.0);

        auto tNumSteps = tStates.extent(0);
        auto tLastStepIndex = tNumSteps - 1;
        for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

          // workset state
          //
          auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
          Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

          // workset state dot
          //
          auto tStateDot = Kokkos::subview(tStatesDot, tStepIndex, Kokkos::ALL());
          Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS);

          // workset state dot dot
          //
          auto tStateDotDot = Kokkos::subview(tStatesDotDot, tStepIndex, Kokkos::ALL());
          Plato::WorksetBase<PhysicsT>::worksetState(tStateDotDot, tStateDotDotWS);

          // evaluate function
          //
          Kokkos::deep_copy(tResult, 0.0);
          mScalarFunctionValue->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

          // sum across elements
          //
          tReturnVal += Plato::local_result_sum<Plato::Scalar>(mNumCells, tResult);
        }

        mScalarFunctionValue->postEvaluate( tReturnVal );

        return tReturnVal;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * @param [in] aSolution Plato::Solution composed of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(const Plato::Solution     & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::Scalar         aTimeStep = 0.0) const override
    {
        using ConfigScalar      = typename GradientX::ConfigScalarType;
        using StateScalar       = typename GradientX::StateScalarType;
        using StateDotScalar    = typename GradientX::StateDotScalarType;
        using StateDotDotScalar = typename GradientX::StateDotDotScalarType;
        using ControlScalar     = typename GradientX::ControlScalarType;
        using ResultScalar      = typename GradientX::ResultScalarType;

        auto tStates = aSolution.State;
        auto tStatesDot = aSolution.StateDot;
        auto tStatesDotDot = aSolution.StateDotDot;

        Plato::ScalarMultiVectorT<StateScalar>       tStateWS       ("state workset",         mNumCells, mNumDofsPerCell);
        Plato::ScalarMultiVectorT<StateDotScalar>    tStateDotWS    ("state dot workset",     mNumCells, mNumDofsPerCell);
        Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS ("state dot dot workset", mNumCells, mNumDofsPerCell);

        // create return view
        //
        Plato::Scalar tObjectiveValue(0.0);
        Plato::ScalarVector tObjGradientX("objective gradient configuration", mNumSpatialDims*mNumNodes);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result",mNumCells);


        auto tNumSteps = tStates.extent(0);
        auto tLastStepIndex = tNumSteps - 1;
        for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

            // workset state
            //
            auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

            // workset state dot
            //
            auto tStateDot = Kokkos::subview(tStatesDot, tStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS);

            // workset state dot dot
            //
            auto tStateDotDot = Kokkos::subview(tStatesDotDot, tStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDotDot, tStateDotDotWS);

            // evaluate function
            //
            Kokkos::deep_copy(tResult, 0.0);
            mScalarFunctionGradientX->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>(mNumCells, mConfigEntryOrdinal, tResult, tObjGradientX);
            tObjectiveValue += Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);
        }

        mScalarFunctionGradientX->postEvaluate( tObjGradientX, tObjectiveValue );

        return tObjGradientX;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * @param [in] aSolution Plato::Solution composed of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step
     * @param [in] aStepIndex step index
     * @return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(const Plato::Solution     & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::OrdinalType    aStepIndex,
                     Plato::Scalar         aTimeStep) const override
    {
        using ConfigScalar      = typename GradientU::ConfigScalarType;
        using StateScalar       = typename GradientU::StateScalarType;
        using StateDotScalar    = typename GradientU::StateDotScalarType;
        using StateDotDotScalar = typename GradientU::StateDotDotScalarType;
        using ControlScalar     = typename GradientU::ControlScalarType;
        using ResultScalar      = typename GradientU::ResultScalarType;

        auto tStates = aSolution.State;
        auto tStatesDot = aSolution.StateDot;
        auto tStatesDotDot = aSolution.StateDotDot;

        assert(aStepIndex < tStates.extent(0));
        assert(tStates.extent(0) > 1);
        assert(aStepIndex > 0);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result",mNumCells);

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", mNumCells, mNumDofsPerCell);
        auto tState = Kokkos::subview(tStates, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset state dot
        //
        Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset", mNumCells, mNumDofsPerCell);
        auto tStateDot = Kokkos::subview(tStatesDot, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS);

        // workset state dot dot
        //
        Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("state dot dot workset", mNumCells, mNumDofsPerCell);
        auto tStateDotDot = Kokkos::subview(tStatesDotDot, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tStateDotDot, tStateDotDotWS);

        // evaluate function
        //
        mScalarFunctionGradientU->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientU("objective gradient state", mNumDofsPerNode*mNumNodes);
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>(mNumCells, mGlobalStateEntryOrdinal, tResult, tObjGradientU);
        Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);

        mScalarFunctionGradientU->postEvaluate( tObjGradientU, tObjectiveValue );

        return tObjGradientU;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the state dot variables
     * @param [in] aSolution Plato::Solution composed of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step
     * @param [in] aStepIndex step index
     * @return 1D view with the gradient of the physics scalar function wrt the state dot variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_v(const Plato::Solution     & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::OrdinalType    aStepIndex,
                     Plato::Scalar         aTimeStep) const override
    {
        using ConfigScalar      = typename GradientV::ConfigScalarType;
        using StateScalar       = typename GradientV::StateScalarType;
        using StateDotScalar    = typename GradientV::StateDotScalarType;
        using StateDotDotScalar = typename GradientV::StateDotDotScalarType;
        using ControlScalar     = typename GradientV::ControlScalarType;
        using ResultScalar      = typename GradientV::ResultScalarType;

        auto tStates = aSolution.State;
        auto tStatesDot = aSolution.StateDot;
        auto tStatesDotDot = aSolution.StateDotDot;

        assert(aStepIndex < tStates.extent(0));
        assert(tStates.extent(0) > 1);
        assert(aStepIndex > 0);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result",mNumCells);

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", mNumCells, mNumDofsPerCell);
        auto tState = Kokkos::subview(tStates, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset state dot
        //
        Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset", mNumCells, mNumDofsPerCell);
        auto tStateDot = Kokkos::subview(tStatesDot, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS);

        // workset state dot dot
        //
        Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("state dot dot workset", mNumCells, mNumDofsPerCell);
        auto tStateDotDot = Kokkos::subview(tStatesDotDot, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tStateDotDot, tStateDotDotWS);

        // evaluate function
        //
        mScalarFunctionGradientV->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientV("objective gradient state", mNumDofsPerNode*mNumNodes);
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>(mNumCells, mGlobalStateEntryOrdinal, tResult, tObjGradientV);
        Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);

        mScalarFunctionGradientV->postEvaluate( tObjGradientV, tObjectiveValue );

        return tObjGradientV;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the state dot dot variables
     * @param [in] aSolution Plato::Solution 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step
     * @param [in] aStepIndex step index
     * @return 1D view with the gradient of the physics scalar function wrt the state dot dot variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_a(const Plato::Solution     & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::OrdinalType    aStepIndex,
                     Plato::Scalar         aTimeStep) const override
    {
        using ConfigScalar      = typename GradientA::ConfigScalarType;
        using StateScalar       = typename GradientA::StateScalarType;
        using StateDotScalar    = typename GradientA::StateDotScalarType;
        using StateDotDotScalar = typename GradientA::StateDotDotScalarType;
        using ControlScalar     = typename GradientA::ControlScalarType;
        using ResultScalar      = typename GradientA::ResultScalarType;

        auto tStates = aSolution.State;
        auto tStatesDot = aSolution.StateDot;
        auto tStatesDotDot = aSolution.StateDotDot;

        assert(aStepIndex < tStates.extent(0));
        assert(tStates.extent(0) > 1);
        assert(aStepIndex > 0);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result",mNumCells);

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", mNumCells, mNumDofsPerCell);
        auto tState = Kokkos::subview(tStates, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset state dot
        //
        Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset", mNumCells, mNumDofsPerCell);
        auto tStateDot = Kokkos::subview(tStatesDot, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS);

        // workset state dot dot
        //
        Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("state dot dot workset", mNumCells, mNumDofsPerCell);
        auto tStateDotDot = Kokkos::subview(tStatesDotDot, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tStateDotDot, tStateDotDotWS);

        // evaluate function
        //
        mScalarFunctionGradientA->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientA("objective gradient state", mNumDofsPerNode*mNumNodes);
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>(mNumCells, mGlobalStateEntryOrdinal, tResult, tObjGradientA);
        Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);

        mScalarFunctionGradientA->postEvaluate( tObjGradientA, tObjectiveValue );

        return tObjGradientA;
    }


    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * @param [in] aSolution Plato::Solution composed of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(const Plato::Solution     & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::Scalar         aTimeStep = 0.0) const override
    {
        using ConfigScalar      = typename GradientZ::ConfigScalarType;
        using StateScalar       = typename GradientZ::StateScalarType;
        using StateDotScalar    = typename GradientZ::StateDotScalarType;
        using StateDotDotScalar = typename GradientZ::StateDotDotScalarType;
        using ControlScalar     = typename GradientZ::ControlScalarType;
        using ResultScalar      = typename GradientZ::ResultScalarType;

        auto tStates = aSolution.State;
        auto tStatesDot = aSolution.StateDot;
        auto tStatesDotDot = aSolution.StateDotDot;

        Plato::ScalarMultiVectorT<StateScalar>       tStateWS       ("state workset",         mNumCells, mNumDofsPerCell);
        Plato::ScalarMultiVectorT<StateDotScalar>    tStateDotWS    ("state dot workset",     mNumCells, mNumDofsPerCell);
        Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS ("state dot dot workset", mNumCells, mNumDofsPerCell);

        // initialize objective value to zero
        //
        Plato::Scalar tObjectiveValue(0.0);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result", mNumCells);

        // create return vector
        //
        Plato::ScalarVector tObjGradientZ("objective gradient control",mNumNodes);

        auto tNumSteps = tStates.extent(0);
        auto tLastStepIndex = tNumSteps - 1;
        for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

            // workset state
            //
            auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

            // workset state dot
            //
            auto tStateDot = Kokkos::subview(tStatesDot, tStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS);

            // workset state dot dot
            //
            auto tStateDotDot = Kokkos::subview(tStatesDotDot, tStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDotDot, tStateDotDotWS);

            // evaluate function
            //
            Kokkos::deep_copy(tResult, 0.0);
            mScalarFunctionGradientZ->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

            Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>(mNumCells, mControlEntryOrdinal, tResult, tObjGradientZ);

            tObjectiveValue += Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);

        }

        mScalarFunctionGradientZ->postEvaluate( tObjGradientZ, tObjectiveValue );

        return tObjGradientZ;
    }

    /******************************************************************************//**
     * @brief Set user defined function name
     * @param [in] function name
    **********************************************************************************/
    void setFunctionName(const std::string aFunctionName)
    {
        mFunctionName = aFunctionName;
    }

    /******************************************************************************//**
     * @brief Return user defined function name
     * @return User defined function name
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

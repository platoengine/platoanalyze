#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

#include "WorksetBase.hpp"
#include "PlatoStaticsTypes.hpp"

#include "parabolic/ScalarFunctionBase.hpp"
#include "parabolic/ParabolicSimplexFadTypes.hpp"
#include "parabolic/AbstractScalarFunction.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Parabolic
{

/******************************************************************************//**
 * @brief Physics scalar function inc class
 **********************************************************************************/
template<typename PhysicsT>
class PhysicsScalarFunction : public Plato::Parabolic::ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode; /*!< number of degree of freedom per node */
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims; /*!< number of spatial dimensions */
    using Plato::WorksetBase<PhysicsT>::mNumControl; /*!< number of control variables */
    using Plato::WorksetBase<PhysicsT>::mNumNodes; /*!< total number of nodes in the mesh */
    using Plato::WorksetBase<PhysicsT>::mNumCells; /*!< total number of cells/elements in the mesh */

    using Plato::WorksetBase<PhysicsT>::mGlobalStateEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mConfigEntryOrdinal; /*!< number of degree of freedom per cell/element */

    using Residual  = typename Plato::Parabolic::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradientU = typename Plato::Parabolic::Evaluation<typename PhysicsT::SimplexT>::GradientU;
    using GradientV = typename Plato::Parabolic::Evaluation<typename PhysicsT::SimplexT>::GradientV;
    using GradientX = typename Plato::Parabolic::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Parabolic::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    using ValueFunction     = std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<Residual>>;
    using GradientUFunction = std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientU>>;
    using GradientVFunction = std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientV>>;
    using GradientXFunction = std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientZ>>;

    std::map<std::string, ValueFunction>     mValueFunctions;
    std::map<std::string, GradientUFunction> mGradientUFunctions;
    std::map<std::string, GradientVFunction> mGradientVFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName;/*!< User defined function name */

    /******************************************************************************//**
     * @brief Initialization of parabolic Physics Scalar Function
     * @param [in] aInputParams input parameters database
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

            mValueFunctions[tName]     = tFactory.template createScalarFunctionParabolic<Residual>
                (tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            mGradientUFunctions[tName] = tFactory.template createScalarFunctionParabolic<GradientU>
                (tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            mGradientVFunctions[tName] = tFactory.template createScalarFunctionParabolic<GradientV>
                (tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            mGradientXFunctions[tName] = tFactory.template createScalarFunctionParabolic<GradientX>
                (tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
            mGradientZFunctions[tName] = tFactory.template createScalarFunctionParabolic<GradientZ>
                (tDomain, mDataMap, aInputParams, tFunctionType, mFunctionName);
        }
    }

public:
    /******************************************************************************//**
     * @brief Primary physics scalar function inc constructor
     * @param [in] aSpatialModel Plato Analyze spatial model
     * @param [in] aDataMap Plato Analyze data map
     * @param [in] aInputParams input parameters database
     * @param [in] aName user defined function name
    **********************************************************************************/
    PhysicsScalarFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
              std::string            & aName
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aInputParams);
    }

    /******************************************************************************//**
     * @brief Secondary physics scalar function inc constructor, used for unit testing
     * @param [in] aSpatialModel Plato Analyze spatial model
     * @param [in] aDataMap Plato Analyze data map
    **********************************************************************************/
    PhysicsScalarFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap& aDataMap
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName ("Undefined Name")
    {
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the residual automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
// TODO is this needed?
/*
    void allocateValue(const std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<Residual>>& aInput)
    {
        mScalarFunctionValue = aInput;
    }
*/

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientU automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
// TODO is this needed?
/*
    void allocateGradientU(const std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientU>>& aInput)
    {
        mScalarFunctionGradientU = aInput;
    }
*/

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientV automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
// TODO is this needed?
/*
    void allocateGradientV(const std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientV>>& aInput)
    {
        mScalarFunctionGradientV = aInput;
    }
*/

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientZ automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
// TODO is this needed?
/*
    void allocateGradientZ(const std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientZ>>& aInput)
    {
        mScalarFunctionGradientZ = aInput;
    }
*/

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientX automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
// TODO is this needed?
/*
    void allocateGradientX(const std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientX>>& aInput)
    {
        mScalarFunctionGradientX = aInput;
    }
*/

    // /******************************************************************************//**
    //  * @brief Update physics-based parameters within optimization iterations
    //  * @param [in] aState 1D view of state variables
    //  * @param [in] aControl 1D view of control variables
    //  **********************************************************************************/
    // void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const
    // {
    //     Plato::ScalarMultiVector tStateWS("state workset", mNumCells, mNumDofsPerCell);
    //     Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

    //     Plato::ScalarMultiVector tControlWS("control workset", mNumCells, mNumNodesPerCell);
    //     Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

    //     Plato::ScalarArray3D tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
    //     Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

    //     mScalarFunctionValue->updateProblem(tStateWS, tControlWS, tConfigWS);
    //     mScalarFunctionGradientU->updateProblem(tStateWS, tControlWS, tConfigWS);
    //     mScalarFunctionGradientV->updateProblem(tStateWS, tControlWS, tConfigWS);
    //     mScalarFunctionGradientZ->updateProblem(tStateWS, tControlWS, tConfigWS);
    //     mScalarFunctionGradientX->updateProblem(tStateWS, tControlWS, tConfigWS);
    // }

    /******************************************************************************//**
     * @brief Evaluate physics scalar function
     * @param [in] aSolution Plato::Solution composed of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return scalar physics function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(
        const Plato::Solution     & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        using ConfigScalar   = typename Residual::ConfigScalarType;
        using StateScalar    = typename Residual::StateScalarType;
        using StateDotScalar = typename Residual::StateDotScalarType;
        using ControlScalar  = typename Residual::ControlScalarType;
        using ResultScalar   = typename Residual::ResultScalarType;

        auto tStates = aSolution.State;
        auto tStateDots = aSolution.StateDot;

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

            Plato::ScalarMultiVectorT<StateScalar>    tStateWS("state workset", tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset", tNumCells, mNumDofsPerCell);

            auto tNumSteps = tStates.extent(0);
            auto tLastStepIndex = tNumSteps - 1;
            for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

                // workset state
                //
                auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS, tDomain);

                // workset state dot
                //
                auto tStateDot = Kokkos::subview(tStateDots, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mValueFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

                // sum across elements
                //
                tReturnVal += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResult);
            }
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mValueFunctions.at(tName)->postEvaluate( tReturnVal );

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
    gradient_x(
        const Plato::Solution     & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        using ConfigScalar   = typename GradientX::ConfigScalarType;
        using StateScalar    = typename GradientX::StateScalarType;
        using StateDotScalar = typename GradientX::StateDotScalarType;
        using ControlScalar  = typename GradientX::ControlScalarType;
        using ResultScalar   = typename GradientX::ResultScalarType;

        auto tStates = aSolution.State;
        auto tStateDots = aSolution.StateDot;

        // create return view
        //
        Plato::ScalarVector tObjGradientX("objective gradient configuration", mNumSpatialDims*mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVectorT<StateScalar>    tStateWS    ("state workset",     tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS ("state dot workset", tNumCells, mNumDofsPerCell);

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

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
                auto tStateDot = Kokkos::subview(tStateDots, tStepIndex, Kokkos::ALL());
                Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS, tDomain);

                // evaluate function
                //
                Kokkos::deep_copy(tResult, 0.0);
                mGradientXFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

                Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                    (tDomain, mConfigEntryOrdinal, tResult, tObjGradientX);

                tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
            }
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientXFunctions.at(tName)->postEvaluate( tObjGradientX, tValue );

        return tObjGradientX;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * @param [in] aSolution Plato::Solution composed of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aStepIndex step index
     * @param [in] aTimeStep time step
     * @return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(
        const Plato::Solution     & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep
    ) const override
    {
        using ConfigScalar   = typename GradientU::ConfigScalarType;
        using StateScalar    = typename GradientU::StateScalarType;
        using StateDotScalar = typename GradientU::StateDotScalarType;
        using ControlScalar  = typename GradientU::ControlScalarType;
        using ResultScalar   = typename GradientU::ResultScalarType;

        auto tStates    = aSolution.State;
        auto tStateDots = aSolution.StateDot;

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientU("objective gradient state", mNumDofsPerNode * mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            assert(aStepIndex < tStates.extent(0));
            assert(tStates.extent(0) > 0);

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
            auto tStateDot = Kokkos::subview(tStateDots, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS, tDomain);

            // evaluate function
            //
            mGradientUFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>
                (tDomain, mGlobalStateEntryOrdinal, tResult, tObjGradientU);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientUFunctions.at(tName)->postEvaluate( tObjGradientU, tValue );

        return tObjGradientU;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the state variables
     * @param [in] aSolution Plato::Solution composed of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aStepIndex step index
     * @param [in] aTimeStep time step
     * @return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_v(
        const Plato::Solution     & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep
    ) const override
    {
        using ConfigScalar   = typename GradientV::ConfigScalarType;
        using StateScalar    = typename GradientV::StateScalarType;
        using StateDotScalar = typename GradientV::StateDotScalarType;
        using ControlScalar  = typename GradientV::ControlScalarType;
        using ResultScalar   = typename GradientV::ResultScalarType;

        auto tStates    = aSolution.State;
        auto tStateDots = aSolution.StateDot;

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientV("objective gradient state", mNumDofsPerNode * mNumNodes);

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
            auto tStateDot = Kokkos::subview(tStateDots, aStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS, tDomain);

            // evaluate function
            //
            mGradientVFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>
                (tDomain, mGlobalStateEntryOrdinal, tResult, tObjGradientV);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientVFunctions.at(tName)->postEvaluate( tObjGradientV, tValue );

        return tObjGradientV;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * @param [in] aSolution Plato::Solution composed of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(
        const Plato::Solution     & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        using ConfigScalar   = typename GradientZ::ConfigScalarType;
        using StateScalar    = typename GradientZ::StateScalarType;
        using StateDotScalar = typename GradientZ::StateDotScalarType;
        using ControlScalar  = typename GradientZ::ControlScalarType;
        using ResultScalar   = typename GradientZ::ResultScalarType;

        auto tStates    = aSolution.State;
        auto tStateDots = aSolution.StateDot;

        // create return vector
        //
        Plato::ScalarVector tObjGradientZ("objective gradient control", mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVectorT<StateScalar>    tStateWS    ("state workset",     tNumCells, mNumDofsPerCell);
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS ("state dot workset", tNumCells, mNumDofsPerCell);

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
               auto tStateDot = Kokkos::subview(tStateDots, tStepIndex-1, Kokkos::ALL());
               Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS, tDomain);

               // evaluate function
               //
               Kokkos::deep_copy(tResult, 0.0);
               mGradientZFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

               Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>
                   (tDomain, mControlEntryOrdinal, tResult, tObjGradientZ);

               tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
           }
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientZFunctions.at(tName)->postEvaluate( tObjGradientZ, tValue );

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
};
//class PhysicsScalarFunction

} // namespace Parabolic

} // namespace Plato

#include "Thermal.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermal<1>>;
extern template class Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermal<2>>;
extern template class Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermal<3>>;
extern template class Plato::Parabolic::PhysicsScalarFunction<::Plato::Thermomechanics<3>>;
#endif

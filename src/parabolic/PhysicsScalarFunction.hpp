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

    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<Residual>> mScalarFunctionValue;
    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientU>> mScalarFunctionGradientU;
    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientV>> mScalarFunctionGradientV;
    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientX>> mScalarFunctionGradientX;
    std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientZ>> mScalarFunctionGradientZ;

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName;/*!< User defined function name */

	/******************************************************************************//**
     * @brief Initialization of parabolic Physics Scalar Function
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh,
                     Omega_h::MeshSets& aMeshSets,
                     Teuchos::ParameterList & aInputParams)
    {
        typename PhysicsT::FunctionFactory tFactory;

        auto tProblemDefault = aInputParams.sublist(mFunctionName);
        auto tFunctionType = tProblemDefault.get<std::string>("Scalar Function Type", ""); // Must be a hardcoded type name (e.g. Volume)

        mScalarFunctionValue =
            tFactory.template createScalarFunctionParabolic<Residual>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientU =
            tFactory.template createScalarFunctionParabolic<GradientU>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientV =
            tFactory.template createScalarFunctionParabolic<GradientV>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientX =
            tFactory.template createScalarFunctionParabolic<GradientX>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientZ =
            tFactory.template createScalarFunctionParabolic<GradientZ>(
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
    PhysicsScalarFunction(Omega_h::Mesh& aMesh,
            Omega_h::MeshSets& aMeshSets,
            Plato::DataMap & aDataMap,
            Teuchos::ParameterList& aInputParams,
            std::string& aName) :
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
    void allocateValue(const std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<Residual>>& aInput)
    {
        mScalarFunctionValue = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientU automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientU(const std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientU>>& aInput)
    {
        mScalarFunctionGradientU = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientV automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientV(const std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientV>>& aInput)
    {
        mScalarFunctionGradientV = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientZ automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientZ(const std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientZ>>& aInput)
    {
        mScalarFunctionGradientZ = aInput;
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientX automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientX(const std::shared_ptr<Plato::Parabolic::AbstractScalarFunction<GradientX>>& aInput)
    {
        mScalarFunctionGradientX = aInput;
    }

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
    value(const Plato::Solution     & aSolution,
          const Plato::ScalarVector & aControl,
                Plato::Scalar         aTimeStep = 0.0) const override
    {
        using ConfigScalar   = typename Residual::ConfigScalarType;
        using StateScalar    = typename Residual::StateScalarType;
        using StateDotScalar = typename Residual::StateDotScalarType;
        using ControlScalar  = typename Residual::ControlScalarType;
        using ResultScalar   = typename Residual::ResultScalarType;

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


        Plato::ScalarMultiVectorT<StateScalar>    tStateWS("state workset",mNumCells,mNumDofsPerCell);
        Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset",mNumCells,mNumDofsPerCell);

        ResultScalar tReturnVal(0.0);

        auto tStates = aSolution.State;
        auto tStateDots = aSolution.StateDot;

        auto tNumSteps = tStates.extent(0);
        auto tLastStepIndex = tNumSteps - 1;
        for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

        // workset state
        //
        auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset state dot
        //
        auto tStateDot = Kokkos::subview(tStateDots, tStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS);

        // evaluate function
        //
        Kokkos::deep_copy(tResult, 0.0);
        mScalarFunctionValue->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

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
        using ConfigScalar   = typename GradientX::ConfigScalarType;
        using StateScalar    = typename GradientX::StateScalarType;
        using StateDotScalar = typename GradientX::StateDotScalarType;
        using ControlScalar  = typename GradientX::ControlScalarType;
        using ResultScalar   = typename GradientX::ResultScalarType;

        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset",mNumCells,mNumDofsPerCell);
        Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset",mNumCells,mNumDofsPerCell);

        // create return view
        //
        Plato::Scalar tObjectiveValue(0.0);
        Plato::ScalarVector tObjGradientX("objective gradient configuration",mNumSpatialDims*mNumNodes);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",mNumCells,mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result",mNumCells);

        auto tStates = aSolution.State;
        auto tStateDots = aSolution.StateDot;

        auto tNumSteps = tStates.extent(0);
        auto tLastStepIndex = tNumSteps - 1;
        for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

        // workset state
        //
        auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset state dot
        //
        auto tStateDot = Kokkos::subview(tStateDots, tStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS);

        // evaluate function
        //
        Kokkos::deep_copy(tResult, 0.0);
        mScalarFunctionGradientX->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

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
     * @param [in] aStepIndex step index
     * @param [in] aTimeStep time step
     * @return 1D view with the gradient of the physics scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(const Plato::Solution     & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::OrdinalType    aStepIndex,
                     Plato::Scalar         aTimeStep) const override
    {
        using ConfigScalar   = typename GradientU::ConfigScalarType;
        using StateScalar    = typename GradientU::StateScalarType;
        using StateDotScalar = typename GradientU::StateDotScalarType;
        using ControlScalar  = typename GradientU::ControlScalarType;
        using ResultScalar   = typename GradientU::ResultScalarType;

        auto tStates = aSolution.State;
        auto tStateDots = aSolution.StateDot;

        assert(aStepIndex < tStates.extent(0));
        assert(tStates.extent(0) > 0);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",mNumCells,mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result",mNumCells);

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset",mNumCells,mNumDofsPerCell);
        auto tState = Kokkos::subview(tStates, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset state dot
        //
        Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset",mNumCells,mNumDofsPerCell);
        auto tStateDot = Kokkos::subview(tStateDots, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS);

        // evaluate function
        //
        mScalarFunctionGradientU->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientU("objective gradient state",mNumDofsPerNode*mNumNodes);
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>(mNumCells, mGlobalStateEntryOrdinal, tResult, tObjGradientU);
        Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);

        mScalarFunctionGradientU->postEvaluate( tObjGradientU, tObjectiveValue );

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
    gradient_v(const Plato::Solution     & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::OrdinalType    aStepIndex,
                     Plato::Scalar         aTimeStep) const override
    {
        using ConfigScalar   = typename GradientV::ConfigScalarType;
        using StateScalar    = typename GradientV::StateScalarType;
        using StateDotScalar = typename GradientV::StateDotScalarType;
        using ControlScalar  = typename GradientV::ControlScalarType;
        using ResultScalar   = typename GradientV::ResultScalarType;

        auto tStates = aSolution.State;
        auto tStateDots = aSolution.StateDot;

        assert(aStepIndex < tStates.extent(0));
        assert(tStates.extent(0) > 0);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset",mNumCells,mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result",mNumCells);

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset",mNumCells,mNumDofsPerCell);
        auto tState = Kokkos::subview(tStates, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

        // workset state dot
        //
        Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset",mNumCells,mNumDofsPerCell);
        auto tStateDot = Kokkos::subview(tStateDots, aStepIndex, Kokkos::ALL());
        Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS);

        // evaluate function
        //
        mScalarFunctionGradientV->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientV("objective gradient state",mNumDofsPerNode*mNumNodes);
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumDofsPerNode>(mNumCells, mGlobalStateEntryOrdinal, tResult, tObjGradientV);
        Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);

        mScalarFunctionGradientV->postEvaluate( tObjGradientV, tObjectiveValue );

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
    gradient_z(const Plato::Solution     & aSolution,
               const Plato::ScalarVector & aControl,
                     Plato::Scalar         aTimeStep = 0.0) const override
    {
        using ConfigScalar   = typename GradientZ::ConfigScalarType;
        using StateScalar    = typename GradientZ::StateScalarType;
        using StateDotScalar = typename GradientZ::StateDotScalarType;
        using ControlScalar  = typename GradientZ::ControlScalarType;
        using ResultScalar   = typename GradientZ::ResultScalarType;

        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset",mNumCells,mNumDofsPerCell);
        Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("state dot workset",mNumCells,mNumDofsPerCell);

        // initialize objective value to zero
        //
        Plato::Scalar tObjectiveValue(0.0);

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

        // create return vector
        //
        Plato::ScalarVector tObjGradientZ("objective gradient control",mNumNodes);

        auto tStates = aSolution.State;
        auto tStateDots = aSolution.StateDot;

        auto tNumSteps = tStates.extent(0);
        auto tLastStepIndex = tNumSteps - 1;
        for( decltype(tNumSteps) tStepIndex = tLastStepIndex; tStepIndex > 0; --tStepIndex ){

            // workset state
            //
            auto tState = Kokkos::subview(tStates, tStepIndex, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tState, tStateWS);

            // workset state dot
            //
            auto tStateDot = Kokkos::subview(tStateDots, tStepIndex-1, Kokkos::ALL());
            Plato::WorksetBase<PhysicsT>::worksetState(tStateDot, tStateDotWS);

            // evaluate function
            //
            Kokkos::deep_copy(tResult, 0.0);
            mScalarFunctionGradientZ->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tResult, aTimeStep );

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

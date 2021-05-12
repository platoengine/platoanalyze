/*
 * FluidsQuasiImplicit.hpp
 *
 *  Created on: Apr 10, 2021
 */

#pragma once

#include <iterator>
#include <unordered_map>

#include <Teuchos_ParameterList.hpp>

#include "BLAS1.hpp"
#include "BLAS2.hpp"
#include "UtilsIO.hpp"
#include "Solutions.hpp"
#include "SpatialModel.hpp"
#include "EssentialBCs.hpp"
#include "PlatoMathHelpers.hpp"
#include "ApplyConstraints.hpp"
#include "UtilsEssentialBCs.hpp"
#include "PlatoAbstractProblem.hpp"

#include "alg/PlatoSolverFactory.hpp"

#include "hyperbolic/FluidsUtils.hpp"
#include "hyperbolic/FluidsCriterionBase.hpp"
#include "hyperbolic/FluidsVectorFunction.hpp"
#include "hyperbolic/FluidsCriterionFactory.hpp"

namespace Plato
{

namespace Fluids
{

/******************************************************************************//**
 * \class QuasiImplicit
 *
 * \brief Main interface for the steady-state solution of incompressible fluid flow problems.
 *
 **********************************************************************************/
template<typename PhysicsT>
class QuasiImplicit : public Plato::AbstractProblem
{
private:
    static constexpr auto mNumSpatialDims      = PhysicsT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell     = PhysicsT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerNode   = PhysicsT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode  = PhysicsT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumPressDofsPerNode = PhysicsT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */

    Plato::Comm::Machine mMachine; /*!< parallel communication interface */
    Teuchos::ParameterList mInputs; /*!< input file metadata */

    Plato::DataMap mDataMap; /*!< static output fields metadata interface */
    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    bool mPrintDiagnostics = true; /*!< boolean flag use to output solver diagnostics to file */
    bool mCalculateHeatTransfer = false; /*!< boolean flag use to enable heat transfer calculations */

    std::ofstream mDiagnostics; /*!< output diagnostics */

    Plato::Scalar mTimeStepDamping = 1.0; /*!< time step damping */
    Plato::Scalar mPressureTolerance = 1e-4; /*!< pressure solver stopping tolerance */
    Plato::Scalar mPredictorTolerance = 1e-4; /*!< velocity predictor solver stopping tolerance */
    Plato::Scalar mCorrectorTolerance = 1e-4; /*!< velocity corrector solver stopping tolerance */
    Plato::Scalar mTemperatureTolerance = 1e-2; /*!< temperature solver stopping tolerance */
    Plato::Scalar mSteadyStateTolerance = 1e-5; /*!< steady-state stopping tolerance */
    Plato::Scalar mTimeStepSafetyFactor = 0.7; /*!< safety factor applied to stable time step */
    Plato::Scalar mCriticalThermalDiffusivity = 1.0; /*!< fluid thermal diffusivity - used to calculate stable time step */
    Plato::Scalar mCriticalKinematicViscocity = 1.0; /*!< fluid kinematic viscocity - used to calculate stable time step */
    Plato::Scalar mCriticalVelocityLowerBound = 0.5; /*!< dimensionless critical convective velocity upper bound */

    Plato::OrdinalType mOutputFrequency = 1e6; /*!< output frequency */
    Plato::OrdinalType mMaxPressureIterations = 5; /*!< maximum number of pressure solver iterations */
    Plato::OrdinalType mMaxPredictorIterations = 5; /*!< maximum number of predictor solver iterations */
    Plato::OrdinalType mMaxCorrectorIterations = 5; /*!< maximum number of corrector solver iterations */
    Plato::OrdinalType mMaxTemperatureIterations = 5; /*!< maximum number of temperature solver iterations */
    Plato::OrdinalType mNumForwardSolveTimeSteps = 0; /*!< number of time steps taken to reach steady state */
    Plato::OrdinalType mMaxSteadyStateIterations = 1000; /*!< maximum number of steady state iterations */

    // primal state containers
    Plato::ScalarMultiVector mPressure; /*!< pressure solution at time step n and n-1 */
    Plato::ScalarMultiVector mVelocity; /*!< velocity solution at time step n and n-1 */
    Plato::ScalarMultiVector mPredictor; /*!< velocity predictor solution at time step n and n-1 */
    Plato::ScalarMultiVector mTemperature; /*!< temperature solution at time step n and n-1 */

    // adjoint state containers
    Plato::ScalarMultiVector mAdjointPressure; /*!< adjoint pressure solution at time step n and n+1 */
    Plato::ScalarMultiVector mAdjointVelocity; /*!< adjoint velocity solution at time step n and n+1 */
    Plato::ScalarMultiVector mAdjointPredictor; /*!< adjoint velocity predictor solution at time step n and n+1 */
    Plato::ScalarMultiVector mAdjointTemperature; /*!< adjoint temperature solution at time step n and n+1 */

    // critical time step container
    std::vector<Plato::Scalar> mCriticalTimeStepHistory; /*!< critical time step history */

    // vector functions
    Plato::Fluids::VectorFunction<typename PhysicsT::MassPhysicsT>     mPressureResidual; /*!< pressure solver vector function interface */
    Plato::Fluids::VectorFunction<typename PhysicsT::MomentumPhysicsT> mPredictorResidual; /*!< velocity predictor solver vector function interface */
    Plato::Fluids::VectorFunction<typename PhysicsT::MomentumPhysicsT> mCorrectorResidual; /*!< velocity corrector solver vector function interface */
    // Using pointer since default VectorFunction constructor allocations are not permitted.
    // Temperature VectorFunction allocation is optional since heat transfer calculations are optional
    std::shared_ptr<Plato::Fluids::VectorFunction<typename PhysicsT::EnergyPhysicsT>> mTemperatureResidual; /*!< temperature solver vector function interface */

    // optimization problem criteria
    using Criterion = std::shared_ptr<Plato::Fluids::CriterionBase>; /*!< local criterion type */
    using Criteria  = std::unordered_map<std::string, Criterion>; /*!< local criterion list type */
    Criteria mCriteria;  /*!< criteria list */

    // local conservation equation, i.e. physics, types
    using MassConservationT     = typename Plato::MassConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>; /*!< local mass conservation equation type */
    using EnergyConservationT   = typename Plato::EnergyConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>; /*!< local energy conservation equation type */
    using MomentumConservationT = typename Plato::MomentumConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>; /*!< local momentum conservation equation type */

    // essential boundary conditions accessors
    Plato::EssentialBCs<MassConservationT>     mPressureEssentialBCs; /*!< pressure essential/Dirichlet boundary condition interface */
    Plato::EssentialBCs<MomentumConservationT> mVelocityEssentialBCs; /*!< velocity essential/Dirichlet boundary condition interface */
    Plato::EssentialBCs<EnergyConservationT>   mTemperatureEssentialBCs; /*!< temperature essential/Dirichlet boundary condition interface */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh     finite element mesh metadata
     * \param [in] aMeshSets mesh entity sets metadata
     * \param [in] aInputs   input file metadata
     * \param [in] aMachine  input file metadata
     **********************************************************************************/
    QuasiImplicit
    (Omega_h::Mesh          & aMesh,
     Omega_h::MeshSets      & aMeshSets,
     Teuchos::ParameterList & aInputs,
     Plato::Comm::Machine   & aMachine) :
         mMachine(aMachine),
         mInputs(aInputs),
         mSpatialModel(aMesh, aMeshSets, aInputs),
         mPressureResidual("Pressure", mSpatialModel, mDataMap, aInputs),
         mCorrectorResidual("Velocity Corrector", mSpatialModel, mDataMap, aInputs),
         mPredictorResidual("Velocity Predictor", mSpatialModel, mDataMap, aInputs),
         mPressureEssentialBCs(aInputs.sublist("Pressure Essential Boundary Conditions",false),aMeshSets),
         mVelocityEssentialBCs(aInputs.sublist("Velocity Essential Boundary Conditions",false),aMeshSets),
         mTemperatureEssentialBCs(aInputs.sublist("Temperature Essential Boundary Conditions",false),aMeshSets)
    {
        this->initialize(aInputs);
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~QuasiImplicit()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            Plato::io::close_text_file(mDiagnostics, mPrintDiagnostics);
        }
    }

    /******************************************************************************//**
     * \fn void output
     * \brief Output solution to visualization file.
     * \param [in] aFilePath output/visualization file path (default = ./output)
     **********************************************************************************/
    void output(const std::string& aFilePath) override
    {
        auto tMesh = mSpatialModel.Mesh;
        auto tWriter = Omega_h::vtk::Writer(aFilePath.c_str(), &tMesh, mNumSpatialDims);

        constexpr auto tStride = 0;
        constexpr auto tCurrentTimeStep = 1;
        const auto tNumNodes = tMesh.nverts();

        auto tPressSubView = Kokkos::subview(mPressure, 0, Kokkos::ALL());
        Omega_h::Write<Omega_h::Real> tPressure(tPressSubView.size(), "Pressure");
        Plato::copy<mNumPressDofsPerNode, mNumPressDofsPerNode>(tStride, tNumNodes, tPressSubView, tPressure);
        tMesh.add_tag(Omega_h::VERT, "Pressure", mNumPressDofsPerNode, Omega_h::Reals(tPressure));

        auto tVelSubView = Kokkos::subview(mVelocity, tCurrentTimeStep, Kokkos::ALL());
        Omega_h::Write<Omega_h::Real> tVelocity(tVelSubView.size(), "Velocity");
        Plato::copy<mNumVelDofsPerNode, mNumVelDofsPerNode>(tStride, tNumNodes, tVelSubView, tVelocity);
        tMesh.add_tag(Omega_h::VERT, "Velocity", mNumVelDofsPerNode, Omega_h::Reals(tVelocity));

        if(mCalculateHeatTransfer)
        {
            auto tTempSubView = Kokkos::subview(mTemperature, tCurrentTimeStep, Kokkos::ALL());
            Omega_h::Write<Omega_h::Real> tTemperature(tTempSubView.size(), "Temperature");
            Plato::copy<mNumTempDofsPerNode, mNumTempDofsPerNode>(tStride, tNumNodes, tTempSubView, tTemperature);
            tMesh.add_tag(Omega_h::VERT, "Temperature", mNumTempDofsPerNode, Omega_h::Reals(tTemperature));
        }

        auto tTags = Omega_h::vtk::get_all_vtk_tags(&tMesh, mNumSpatialDims);
        auto tTime = static_cast<Plato::Scalar>(tCurrentTimeStep);
        tWriter.write(tCurrentTimeStep, tTime, tTags);
    }

    /******************************************************************************//**
     * \brief Update simulation parameters within optimization iterations
     * \param [in] aControl 1D container of control variables
     * \param [in] aSolution solution database
    **********************************************************************************/
    void updateProblem
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution)
     override
    { return; }

    /******************************************************************************//**
     * \fn Plato::Solutions solution
     *
     * \brief Solve finite element simulation.
     * \param [in] aControl vector of design/optimization variables
     * \return Plato database with state solutions
     *
     **********************************************************************************/
    Plato::Solutions 
    solution(const Plato::ScalarVector& aControl)
    override
    {
        this->clear();
        this->checkProblemSetup();

        Plato::Primal tPrimal;
        auto tWriter = Omega_h::vtk::Writer("solution_history", &mSpatialModel.Mesh, mNumSpatialDims);
        this->setInitialConditions(tPrimal, tWriter);
        this->calculateCharacteristicElemSize(tPrimal);

        for(Plato::OrdinalType tIteration = 0; tIteration < mMaxSteadyStateIterations; tIteration++)
        {
            mNumForwardSolveTimeSteps = tIteration + 1;
            tPrimal.scalar("time step index", mNumForwardSolveTimeSteps);

            this->setPrimal(tPrimal);
            this->calculateCriticalTimeStep(tPrimal);
            this->checkCriticalTimeStep(tPrimal);

            this->printIteration(tPrimal);
            this->updatePredictor(aControl, tPrimal);
            this->updatePressure(aControl, tPrimal);
            this->updateCorrector(aControl, tPrimal);

            if(mCalculateHeatTransfer)
            {
                this->updateTemperature(aControl, tPrimal);
            }

            if(this->writeOutput(tIteration))
            {
                this->write(tPrimal, tWriter);
            }

            if(this->checkStoppingCriteria(tPrimal))
            {
                break;
            }
            this->savePrimal(tPrimal);
        }

        auto tSolution = this->setSolution();
        return tSolution;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionValue
     *
     * \brief Evaluate criterion.
     * \param [in] aControl  vector of design/optimization variables
     * \param [in] aSolution Plato database with state solutions
     * \param [in] aName     criterion name/identifier
     * \return criterion evaluation
     *
     **********************************************************************************/
    Plato::Scalar 
    criterionValue
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution,
     const std::string         & aName)
     override
    {
        return (this->criterionValue(aControl, aName));
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionValue
     *
     * \brief Evaluate criterion.
     * \param [in] aControl  vector of design/optimization variables
     * \param [in] aName     criterion name/identifier
     * \return criterion evaluation
     *
     **********************************************************************************/
    Plato::Scalar 
    criterionValue
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
     override
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not in the criteria list");
        }

        auto tDirectory = std::string("solution_history");
        auto tSolutionHistory = Plato::omega_h::read_pvtu_file_paths(tDirectory);
        if(tSolutionHistory.size() != static_cast<size_t>(mNumForwardSolveTimeSteps + 1))
        {
            THROWERR(std::string("Number of time steps read from the '") + tDirectory
                 + "' directory does not match the expected value: '" + std::to_string(mNumForwardSolveTimeSteps + 1) + "'.")
        }

        // evaluate steady-state criterion
        Plato::Primal tPrimal;
        auto tLastTimeStepIndex = tSolutionHistory.size() - 1u;
        tPrimal.scalar("time step index", tLastTimeStepIndex);
        this->setPrimal(tSolutionHistory, tPrimal);
        auto tOutput = tItr->second->value(aControl, tPrimal);

        return tOutput;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradient
     *
     * \brief Evaluate criterion gradient with respect to design/optimization variables.
     * \param [in] aControl  vector of design/optimization variables
     * \param [in] aSolution Plato database with state solutions
     * \param [in] aName     criterion name/identifier
     * \return criterion gradient with respect to design/optimization variables
     *
     **********************************************************************************/
    Plato::ScalarVector 
    criterionGradient
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution,
     const std::string         & aName)
    override
    {
        return (this->criterionGradient(aControl, aName));
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradient
     *
     * \brief Evaluate criterion gradient with respect to design/optimization variables.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion gradient with respect to design/optimization variables
     *
     **********************************************************************************/
    Plato::ScalarVector criterionGradient
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    override
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not defined in critria map.");
        }

        Plato::Dual tDual;
        Plato::Primal tCurrentState, tPreviousState;
        auto tDirectory = std::string("solution_history");
        auto tSolutionHistoryPaths = Plato::omega_h::read_pvtu_file_paths(tDirectory);
        if(tSolutionHistoryPaths.size() != static_cast<size_t>(mNumForwardSolveTimeSteps + 1))
        {
            THROWERR(std::string("Number of time steps read from the '") + tDirectory
                 + "' directory does not match the expected value: '" + std::to_string(mNumForwardSolveTimeSteps + 1) + "'.")
        }

        Plato::ScalarVector tTotalDerivative("total derivative", mSpatialModel.Mesh.nverts());
        for(auto tItr = tSolutionHistoryPaths.rbegin(); tItr != tSolutionHistoryPaths.rend() - 1; tItr++)
        {
            // set fields for the current primal state
            auto tCurrentStateIndex = (tSolutionHistoryPaths.size() - 1u) - std::distance(tSolutionHistoryPaths.rbegin(), tItr);
            tCurrentState.scalar("time step index", tCurrentStateIndex);
            this->setPrimal(tSolutionHistoryPaths, tCurrentState);
            this->setCriticalTimeStep(tCurrentState);

                // set fields for the previous primal state
            auto tPreviousStateIndex = tCurrentStateIndex + 1u;
            tPreviousState.scalar("time step index", tPreviousStateIndex);
            if(tPreviousStateIndex != tSolutionHistoryPaths.size())
            {
                this->setPrimal(tSolutionHistoryPaths, tPreviousState);
                this->setCriticalTimeStep(tPreviousState);
            }

                // set adjoint state
            this->setDual(tDual);

            // update adjoint states
            if(mCalculateHeatTransfer)
            {
                this->updateTemperatureAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            }
            this->updateCorrectorAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            this->updatePressureAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            this->updatePredictorAdjoint(aControl, tCurrentState, tPreviousState, tDual);

            // update total derivative with respect to control variables
            this->updateTotalDerivativeWrtControl(aName, aControl, tCurrentState, tDual, tTotalDerivative);

            this->saveDual(tDual);
        }
        return tTotalDerivative;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradientX
     *
     * \brief Evaluate criterion gradient with respect to configuration variables.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion gradient with respect to configuration variables
     *
     **********************************************************************************/
    Plato::ScalarVector 
    criterionGradientX
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
     override
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not defined in critria map.");
        }

        Plato::Dual tDual;
        Plato::Primal tCurrentState, tPreviousState;
        auto tDirectory = std::string("solution_history");
        auto tSolutionHistoryPaths = Plato::omega_h::read_pvtu_file_paths(tDirectory);
        if(tSolutionHistoryPaths.size() != static_cast<size_t>(mNumForwardSolveTimeSteps + 1))
        {
            THROWERR(std::string("Number of time steps read from the '") + tDirectory
                 + "' directory does not match the expected value: '" + std::to_string(mNumForwardSolveTimeSteps + 1) + "'.")
        }

        Plato::ScalarVector tTotalDerivative("total derivative", mSpatialModel.Mesh.nverts());
        for(auto tItr = tSolutionHistoryPaths.rbegin(); tItr != tSolutionHistoryPaths.rend() - 1; tItr++)
        {
            // set fields for the current primal state
            auto tCurrentStateIndex = (tSolutionHistoryPaths.size() - 1u) - std::distance(tSolutionHistoryPaths.rbegin(), tItr);
            tCurrentState.scalar("time step index", tCurrentStateIndex);
            this->setPrimal(tSolutionHistoryPaths, tCurrentState);
            this->setCriticalTimeStep(tCurrentState);

            // set fields for the previous primal state
            auto tPreviousStateIndex = tCurrentStateIndex + 1u;
            tPreviousState.scalar("time step index", tPreviousStateIndex);
            if(tPreviousStateIndex != tSolutionHistoryPaths.size())
            {
                this->setPrimal(tSolutionHistoryPaths, tPreviousState);
                this->setCriticalTimeStep(tPreviousState);
            }

            // set adjoint state
            this->setDual(tDual);

            // update adjoint states
            if(mCalculateHeatTransfer)
            {
                this->updateTemperatureAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            }
            this->updateCorrectorAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            this->updatePressureAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            this->updatePredictorAdjoint(aControl, tCurrentState, tPreviousState, tDual);

            // update total derivative with respect to control variables
            this->updateTotalDerivativeWrtConfig(aName, aControl, tCurrentState, tDual, tTotalDerivative);

            this->saveDual(tDual);
        }
        return tTotalDerivative;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradientX
     *
     * \brief Evaluate criterion gradient with respect to configuration variables.
     * \param [in] aControl  vector of design/optimization variables
     * \param [in] aSolution Plato database with state solutions
     * \param [in] aName     criterion name/identifier
     * \return criterion gradient with respect to configuration variables
     *
     **********************************************************************************/
    Plato::ScalarVector 
    criterionGradientX
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution,
     const std::string         & aName)
     override
    {
        return (this->criterionGradientX(aControl, aName));
    }

private:
    /******************************************************************************//**
     * \fn void write
     * \brief Write solution to visualization file. This function is mostly used for
     *   optimization purposes to avoid storing large time-dependent state history in
     *   memory. Thus, maximizing available GPU memory.
     *
     * \param [in] aPrimal primal state database
     * \param [in] aWriter interface to allow output to a VTK visualization file
     *
     **********************************************************************************/
    void write
    (const Plato::Primal& aPrimal,
     Omega_h::vtk::Writer& aWriter)
    {
        constexpr auto tStride = 0;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        const Plato::OrdinalType tTimeStepIndex = aPrimal.scalar("time step index");

        std::string tTag = tTimeStepIndex != static_cast<Plato::OrdinalType>(0) ? "current pressure" : "previous pressure";
        auto tPressureView = aPrimal.vector(tTag);
        Omega_h::Write<Omega_h::Real> tPressure(tPressureView.size(), "Pressure");
        Plato::copy<mNumPressDofsPerNode, mNumPressDofsPerNode>(tStride, tNumNodes, tPressureView, tPressure);
        mSpatialModel.Mesh.add_tag(Omega_h::VERT, "Pressure", mNumPressDofsPerNode, Omega_h::Reals(tPressure));

        tTag = tTimeStepIndex != static_cast<Plato::OrdinalType>(0) ? "current velocity" : "previous velocity";
        auto tVelocityView = aPrimal.vector(tTag);
        Omega_h::Write<Omega_h::Real> tVelocity(tVelocityView.size(), "Velocity");
        Plato::copy<mNumVelDofsPerNode, mNumVelDofsPerNode>(tStride, tNumNodes, tVelocityView, tVelocity);
        mSpatialModel.Mesh.add_tag(Omega_h::VERT, "Velocity", mNumVelDofsPerNode, Omega_h::Reals(tVelocity));

        tTag = tTimeStepIndex != static_cast<Plato::OrdinalType>(0) ? "current predictor" : "previous predictor";
        auto tPredictorView = aPrimal.vector(tTag);
        Omega_h::Write<Omega_h::Real> tPredictor(tPredictorView.size(), "Predictor");
        Plato::copy<mNumVelDofsPerNode, mNumVelDofsPerNode>(tStride, tNumNodes, tPredictorView, tPredictor);
        mSpatialModel.Mesh.add_tag(Omega_h::VERT, "Predictor", mNumVelDofsPerNode, Omega_h::Reals(tPredictor));

        if(mCalculateHeatTransfer)
        {
            tTag = tTimeStepIndex != static_cast<Plato::OrdinalType>(0) ? "current temperature" : "previous temperature";
            auto tTemperatureView = aPrimal.vector(tTag);
            Omega_h::Write<Omega_h::Real> tTemperature(tTemperatureView.size(), "Temperature");
            Plato::copy<mNumTempDofsPerNode, mNumTempDofsPerNode>(tStride, tNumNodes, tTemperatureView, tTemperature);
            mSpatialModel.Mesh.add_tag(Omega_h::VERT, "Temperature", mNumTempDofsPerNode, Omega_h::Reals(tTemperature));
        }

        auto tTags = Omega_h::vtk::get_all_vtk_tags(&mSpatialModel.Mesh, mNumSpatialDims);
        aWriter.write(tTimeStepIndex, tTimeStepIndex, tTags);
    }

    /******************************************************************************//**
     * \fn bool writeOutput
     *
     * \brief Return boolean used to determine if state solution will be written to
     *   visualization file.
     * \param [in] aIteration current solver iteration
     * \return boolean (true = output to file; false = skip output to file)
     *
     **********************************************************************************/
    bool writeOutput(const Plato::OrdinalType aIteration) const
    {
        auto tWrite = false;
        if(mOutputFrequency > static_cast<Plato::OrdinalType>(0))
        {
            auto tModulo = (aIteration + static_cast<Plato::OrdinalType>(1)) % mOutputFrequency;
            tWrite = tModulo == static_cast<Plato::OrdinalType>(0) ? true : false;
        }
        return tWrite;
    }

    /******************************************************************************//**
     * \fn void readCurrentFields
     *
     * \brief Read current states from visualization file.
     * \param [in]     aPath   visualization file path
     * \param [in/out] aStates primal state solution database
     *
     **********************************************************************************/
    void readCurrentFields
    (const Omega_h::filesystem::path & aPath,
           Plato::Primal             & aPrimal)
    {
        Plato::FieldTags tFieldTags;
        tFieldTags.set("Velocity", "current velocity");
        tFieldTags.set("Pressure", "current pressure");
        tFieldTags.set("Predictor", "current predictor");
        if(mCalculateHeatTransfer)
        {
            tFieldTags.set("Temperature", "current temperature");
        }
        Plato::omega_h::read_fields<Omega_h::VERT>(mSpatialModel.Mesh, aPath, tFieldTags, aPrimal);
    }

    /******************************************************************************//**
     * \fn void readPreviousFields
     *
     * \brief Read previous states from visualization file.
     * \param [in]     aPath   visualization file path
     * \param [in/out] aStates primal state solution database
     *
     **********************************************************************************/
    void readPreviousFields
    (const Omega_h::filesystem::path & aPath,
           Plato::Primal             & aPrimal)
    {
        Plato::FieldTags tFieldTags;
        tFieldTags.set("Velocity", "previous velocity");
        tFieldTags.set("Pressure", "previous pressure");
        if(mCalculateHeatTransfer)
        {
            tFieldTags.set("Temperature", "previous temperature");
        }
        Plato::omega_h::read_fields<Omega_h::VERT>(mSpatialModel.Mesh, aPath, tFieldTags, aPrimal);
    }

    /******************************************************************************//**
     * \fn void setPrimal
     *
     * \brief Set primal state solution database for the current optimization iteration.
     * \param [in]     aPath   list with paths to visualization files
     * \param [in/out] aPrimal primal state solution database
     *
     **********************************************************************************/
    void setPrimal
    (const std::vector<Omega_h::filesystem::path> & aPaths,
           Plato::Primal                          & aPrimal)
    {
        auto tTimeStepIndex = static_cast<size_t>(aPrimal.scalar("time step index"));
        this->readCurrentFields(aPaths[tTimeStepIndex], aPrimal);
        this->readPreviousFields(aPaths[tTimeStepIndex - 1u], aPrimal);
    }

    /******************************************************************************//**
     * \fn void setCriticalTimeStep
     *
     * \brief Set critical time step for the current optimization iteration.
     * \param [in/out] aPrimal primal state solution database
     *
     **********************************************************************************/
    void setCriticalTimeStep
    (Plato::Primal& aPrimal)
    {
        auto tTimeStepIndex = static_cast<size_t>(aPrimal.scalar("time step index"));
        Plato::ScalarVector tCriticalTimeStep("critical time step", 1);
        auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);
        tHostCriticalTimeStep(0) = mCriticalTimeStepHistory[tTimeStepIndex];
        Kokkos::deep_copy(tCriticalTimeStep, tHostCriticalTimeStep);
        aPrimal.vector("critical time step", tCriticalTimeStep);
    }

    /******************************************************************************//**
     * \fn void setSolution
     *
     * \brief Set solution database.
     * \return solution database
     *
     **********************************************************************************/
    Plato::Solutions setSolution()
    {
        Plato::Solutions tSolution("incompressible cfd");
        tSolution.set("velocity", mVelocity);
        tSolution.set("pressure", mPressure);
        if(mCalculateHeatTransfer)
        {
            tSolution.set("temperature", mTemperature);
        }
        return tSolution;
    }

    /******************************************************************************//**
     * \fn void setInitialConditions
     *
     * \brief Set initial conditions for pressure, temperature and veloctity fields.
     * \param [in] aPrimal primal state database
     * \param [in] aWriter interface to allow output to a VTK visualization file
     *
     **********************************************************************************/
    void setInitialConditions
    (Plato::Primal & aPrimal,
     Omega_h::vtk::Writer& aWriter)
    {
        const Plato::Scalar tTime = 0.0;
        const Plato::OrdinalType tTimeStep = 0;
        mCriticalTimeStepHistory.push_back(0.0);
        aPrimal.scalar("time step index", tTimeStep);
        aPrimal.scalar("critical velocity lower bound", mCriticalVelocityLowerBound);

        Plato::ScalarVector tVelBcValues;
        Plato::LocalOrdinalVector tVelBcDofs;
        mVelocityEssentialBCs.get(tVelBcDofs, tVelBcValues, tTime);
        //Plato::post_process_dirichlet_dofs(tVelBcDofs, tVelBcValues);
        auto tPreviouVel = Kokkos::subview(mVelocity, tTimeStep, Kokkos::ALL());
        Plato::enforce_boundary_condition(tVelBcDofs, tVelBcValues, tPreviouVel);
        aPrimal.vector("previous velocity", tPreviouVel);

        Plato::ScalarVector tPressBcValues;
        Plato::LocalOrdinalVector tPressBcDofs;
        mPressureEssentialBCs.get(tPressBcDofs, tPressBcValues, tTime);
        auto tPreviousPress = Kokkos::subview(mPressure, tTimeStep, Kokkos::ALL());
        Plato::enforce_boundary_condition(tPressBcDofs, tPressBcValues, tPreviousPress);
        aPrimal.vector("previous pressure", tPreviousPress);

        auto tPreviousPred = Kokkos::subview(mPredictor, tTimeStep, Kokkos::ALL());
        aPrimal.vector("previous predictor", tPreviousPred);

        if(mCalculateHeatTransfer)
        {
            Plato::ScalarVector tTempBcValues;
            Plato::LocalOrdinalVector tTempBcDofs;
            mTemperatureEssentialBCs.get(tTempBcDofs, tTempBcValues, tTime);
            auto tPreviousTemp  = Kokkos::subview(mTemperature, tTimeStep, Kokkos::ALL());
            Plato::enforce_boundary_condition(tTempBcDofs, tTempBcValues, tPreviousTemp);
            aPrimal.vector("previous temperature", tPreviousTemp);

            aPrimal.scalar("thermal diffusivity", mCriticalThermalDiffusivity);
            aPrimal.scalar("kinematic viscocity", mCriticalKinematicViscocity);
        }

        if(this->writeOutput(tTimeStep))
        {
            this->write(aPrimal, aWriter);
        }
    }

    /******************************************************************************//**
     * \fn void printIteration
     *
     * \brief Print current iteration diagnostics to diagnostic file.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    void printIteration
    (const Plato::Primal & aPrimal)
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                auto tCriticalTimeStep = aPrimal.vector("critical time step");
                auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);
                Kokkos::deep_copy(tHostCriticalTimeStep, tCriticalTimeStep);
                const Plato::OrdinalType tTimeStepIndex = aPrimal.scalar("time step index");
                tMsg << "*************************************************************************************\n";
                tMsg << "* Critical Time Step: " << tHostCriticalTimeStep(0) << "\n";
                tMsg << "* CFD Quasi-Implicit Solver Iteration: " << tTimeStepIndex << "\n";
                tMsg << "*************************************************************************************\n";
                Plato::io::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void areDianosticsEnabled
     *
     * \brief Check if diagnostics are enabled, if true, open diagnostic file.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void areDianosticsEnabled
    (Teuchos::ParameterList & aInputs)
    {
        mPrintDiagnostics = aInputs.get<bool>("Diagnostics", true);
        auto tFileName = aInputs.get<std::string>("Diagnostics File Name", "cfd_solver_diagnostics.txt");
        if(Plato::Comm::rank(mMachine) == 0)
        {
            Plato::io::open_text_file(tFileName, mDiagnostics, mPrintDiagnostics);
        }
    }

    /******************************************************************************//**
     * \fn void initialize
     *
     * \brief Initialize member data.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void initialize
    (Teuchos::ParameterList & aInputs)
    {
        this->allocatePrimalStates();
        this->areDianosticsEnabled(aInputs);
        this->parseNewtonSolverInputs(aInputs);
        this->parseConvergenceCriteria(aInputs);
        this->parseTimeIntegratorInputs(aInputs);
        this->setHeatTransferEquation(aInputs);
        this->allocateOptimizationMetadata(aInputs);
    }

    /******************************************************************************//**
     * \fn void setCriticalFluidProperties
     *
     * \brief Set fluid properties used to calculate the critical time step for heat
     *   transfer applications.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void setCriticalFluidProperties(Teuchos::ParameterList &aInputs)
    {
        mCriticalThermalDiffusivity = Plato::teuchos::parse_max_material_property<Plato::Scalar>
            (aInputs, "Thermal Properties", "Thermal Diffusivity", mSpatialModel.Domains);
        Plato::is_positive_finite_number(mCriticalThermalDiffusivity, "Thermal Diffusivity");
        mCriticalKinematicViscocity = Plato::teuchos::parse_max_material_property<Plato::Scalar>
            (aInputs, "Viscous Properties", "Kinematic Viscocity", mSpatialModel.Domains);
        Plato::is_positive_finite_number(mCriticalKinematicViscocity, "Kinematic Viscocity");
    }

    /******************************************************************************//**
     * \fn void setHeatTransferEquation
     *
     * \brief Set temperature equation vector function interface if heat transfer
     *   calculations are requested.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void setHeatTransferEquation
    (Teuchos::ParameterList & aInputs)
    {
        mCalculateHeatTransfer = Plato::Fluids::calculate_heat_transfer(aInputs);
        if(mCalculateHeatTransfer)
        {
            mTemperatureResidual = std::make_shared<Plato::Fluids::VectorFunction<typename PhysicsT::EnergyPhysicsT>>
                    ("Temperature", mSpatialModel, mDataMap, aInputs);
            this->setCriticalFluidProperties(aInputs);
        }
    }

    /******************************************************************************//**
     * \fn void parseNewtonSolverInputs
     *
     * \brief Parse Newton solver parameters from input file.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void parseNewtonSolverInputs
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Newton Iteration"))
        {
            auto tNewtonIteration = aInputs.sublist("Newton Iteration");
            mPressureTolerance = tNewtonIteration.get<Plato::Scalar>("Pressure Tolerance", 1e-4);
            mPredictorTolerance = tNewtonIteration.get<Plato::Scalar>("Predictor Tolerance", 1e-4);
            mCorrectorTolerance = tNewtonIteration.get<Plato::Scalar>("Corrector Tolerance", 1e-4);
            mTemperatureTolerance = tNewtonIteration.get<Plato::Scalar>("Temperature Tolerance", 1e-4);
            mMaxPressureIterations = tNewtonIteration.get<Plato::OrdinalType>("Pressure Iterations", 10);
            mMaxPredictorIterations = tNewtonIteration.get<Plato::OrdinalType>("Predictor Iterations", 10);
            mMaxCorrectorIterations = tNewtonIteration.get<Plato::OrdinalType>("Corrector Iterations", 10);
            mMaxTemperatureIterations = tNewtonIteration.get<Plato::OrdinalType>("Temperature Iterations", 10);
        }
    }

    /******************************************************************************//**
     * \fn void parseTimeIntegratorInputs
     *
     * \brief Parse time integration scheme parameters from input file.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void parseTimeIntegratorInputs
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mTimeStepDamping = tTimeIntegration.get<Plato::Scalar>("Damping", 1.0);
            mTimeStepSafetyFactor = tTimeIntegration.get<Plato::Scalar>("Safety Factor", 0.7);
        }
    }

    /******************************************************************************//**
     * \fn void parseConvergenceCriteria
     *
     * \brief Parse fluid solver's convergence criteria from input file.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void parseConvergenceCriteria
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Convergence"))
        {
            auto tConvergence = aInputs.sublist("Convergence");
            mSteadyStateTolerance = tConvergence.get<Plato::Scalar>("Steady State Tolerance", 1e-5);
            mMaxSteadyStateIterations = tConvergence.get<Plato::OrdinalType>("Maximum Iterations", 1000);
            mOutputFrequency = tConvergence.get<Plato::OrdinalType>("Output Frequency", mMaxSteadyStateIterations + 1);
        }
    }

    /******************************************************************************//**
     * \fn void clear
     *
     * \brief Clear forward solver state data. This function is utilized only in
     *   optimization workflows since the solver is used in re-entrant mode.
     *
     **********************************************************************************/
    void clear()
    {
        mNumForwardSolveTimeSteps = 0;
        mCriticalTimeStepHistory.clear();
        Plato::blas2::fill(0.0, mPressure);
        Plato::blas2::fill(0.0, mVelocity);
        Plato::blas2::fill(0.0, mPredictor);
        Plato::blas2::fill(0.0, mTemperature);

        auto tDirectory = std::string("solution_history");
        Plato::filesystem::remove(tDirectory);
    }

    /******************************************************************************//**
     * \fn void checkProblemSetup
     *
     * \brief Check forward problem setup.
     *
     **********************************************************************************/
    void checkProblemSetup()
    {
        if(mVelocityEssentialBCs.empty())
        {
            THROWERR("Velocity essential boundary conditions are not defined.")
        }
        if(mCalculateHeatTransfer)
        {
            if(mTemperatureEssentialBCs.empty())
            {
                THROWERR("Temperature essential boundary conditions are not defined.")
            }
            if(mTemperatureResidual.use_count() == 0)
            {
                THROWERR("Heat transfer calculation requested but temperature 'Vector Function' is not allocated.")
            }
        }
    }

    /******************************************************************************//**
     * \fn void allocateDualStates
     *
     * \brief Allocate dual state containers.
     *
     **********************************************************************************/
    void allocateDualStates()
    {
        constexpr auto tTimeSnapshotsStored = 2;
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        mAdjointPressure = Plato::ScalarMultiVector("Adjoint Pressure Snapshots", tTimeSnapshotsStored, tNumNodes);
        mAdjointVelocity = Plato::ScalarMultiVector("Adjoint Velocity Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
        mAdjointPredictor = Plato::ScalarMultiVector("Adjoint Predictor Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);

        if(mCalculateHeatTransfer)
        {
            mAdjointTemperature = Plato::ScalarMultiVector("Adjoint Temperature Snapshots", tTimeSnapshotsStored, tNumNodes);
        }
    }

    /******************************************************************************//**
     * \fn void allocatePrimalStates
     *
     * \brief Allocate primal state containers.
     *
     **********************************************************************************/
    void allocatePrimalStates()
    {
        constexpr auto tTimeSnapshotsStored = 2;
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        mPressure    = Plato::ScalarMultiVector("Pressure Snapshots", tTimeSnapshotsStored, tNumNodes);
        mVelocity    = Plato::ScalarMultiVector("Velocity Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
        mPredictor   = Plato::ScalarMultiVector("Predictor Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
        mTemperature = Plato::ScalarMultiVector("Temperature Snapshots", tTimeSnapshotsStored, tNumNodes);
    }

    /******************************************************************************//**
     * \fn void allocateCriteriaList
     *
     * \brief Allocate criteria list.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void allocateCriteriaList(Teuchos::ParameterList &aInputs)
    {
        Plato::Fluids::CriterionFactory<PhysicsT> tScalarFuncFactory;

        auto tCriteriaParams = aInputs.sublist("Criteria");
        for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); tIndex != tCriteriaParams.end(); ++tIndex)
        {
            const Teuchos::ParameterEntry& tEntry = tCriteriaParams.entry(tIndex);
            if(tEntry.isList() == false)
            {
                THROWERR("Parameter in Criteria block is not supported. Expect lists only.")
            }
            auto tName = tCriteriaParams.name(tIndex);
            auto tCriterion = tScalarFuncFactory.createCriterion(mSpatialModel, mDataMap, aInputs, tName);
            if( tCriterion != nullptr )
            {
                mCriteria[tName] = tCriterion;
            }
        }
    }

    /******************************************************************************//**
     * \fn void allocateOptimizationMetadata
     *
     * \brief Allocate optimization problem metadata.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void allocateOptimizationMetadata(Teuchos::ParameterList &aInputs)
    {
        if(aInputs.isSublist("Criteria"))
        {
            this->allocateDualStates();
            this->allocateCriteriaList(aInputs);
        }
    }

    /******************************************************************************//**
     * \fn void calculateVelocityMisfitNorm
     *
     * \brief Calculate velocity misfit euclidean norm.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    Plato::Scalar calculateVelocityMisfitNorm
    (const Plato::Primal & aPrimal)
    {
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        auto tCurrentVelocity = aPrimal.vector("current velocity");
        auto tPreviousVelocity = aPrimal.vector("previous velocity");
        auto tMisfitError = Plato::blas1::norm(tCurrentVelocity, tPreviousVelocity);
        auto tCurrentVelNorm = Plato::blas1::norm(tCurrentVelocity);
        auto tOutput = tMisfitError / tCurrentVelNorm;
        return tOutput;
    }

    /******************************************************************************//**
     * \fn void calculatePressureMisfitNorm
     *
     * \brief Calculate pressure misfit euclidean norm.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    Plato::Scalar calculatePressureMisfitNorm
    (const Plato::Primal & aPrimal)
    {
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        auto tCurrentPressure = aPrimal.vector("current pressure");
        auto tPreviousPressure = aPrimal.vector("previous pressure");
        auto tMisfitError = Plato::blas1::norm(tCurrentPressure, tPreviousPressure);
        auto tCurrentNorm = Plato::blas1::norm(tCurrentPressure);
        auto tOutput = tMisfitError / tCurrentNorm;
        return tOutput;
    }

    /******************************************************************************//**
     * \fn void printSteadyStateCriterion
     *
     * \brief Print steady state criterion to diagnostic file.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    void printSteadyStateCriterion
    (const Plato::Primal & aPrimal)
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                auto tCriterion = aPrimal.scalar("current steady state criterion");
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << std::scientific << " Steady State Convergence: " << tCriterion << "\n";
                tMsg << "-------------------------------------------------------------------------------------\n\n";
                Plato::io::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn bool isFluidSolverDiverging
     *
     * \brief Check is fluid solver is diverging.
     * \param [in] aPrimal primal state database
     * \return boolean (true = diverging; false = not diverging)
     *
     **********************************************************************************/
    bool isFluidSolverDiverging
    (Plato::Primal & aPrimal)
    {
        auto tCurrentCriterion = aPrimal.scalar("current steady state criterion");
        if(!std::isfinite(tCurrentCriterion) || std::isnan(tCurrentCriterion))
        {
            return true;
        }
        return false;
    }

    /******************************************************************************//**
     * \fn bool checkStoppingCriteria
     *
     * \brief Check fluid solver stopping criterion.
     * \param [in] aPrimal primal state database
     * \return boolean (true = converged; false = did not coverge)
     *
     **********************************************************************************/
    bool checkStoppingCriteria
    (Plato::Primal & aPrimal)
    {
        bool tStop = false;
        const Plato::OrdinalType tTimeStepIndex = aPrimal.scalar("time step index");
        const auto tCriterionValue = this->calculatePressureMisfitNorm(aPrimal);
        aPrimal.scalar("current steady state criterion", tCriterionValue);
        this->printSteadyStateCriterion(aPrimal);


        if (tCriterionValue < mSteadyStateTolerance)
        {
            tStop = true;
        }
        else if (tTimeStepIndex >= mMaxSteadyStateIterations)
        {
            tStop = true;
        }
        else if(this->isFluidSolverDiverging(aPrimal))
        {
            tStop = true;
        }

        aPrimal.scalar("previous steady state criterion", tCriterionValue);

        return tStop;
    }

    /******************************************************************************//**
     * \fn void calculateCharacteristicElemSize
     *
     * \brief Calculate characteristic element size
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    void calculateCharacteristicElemSize
    (Plato::Primal & aPrimal)
    {
        auto tElemCharSizes =
            Plato::Fluids::calculate_characteristic_element_size<mNumSpatialDims,mNumNodesPerCell>(mSpatialModel);
        aPrimal.vector("element characteristic size", tElemCharSizes);
    }

    /******************************************************************************//**
     * \fn Plato::Scalar calculateCriticalConvectiveTimeStep
     *
     * \brief Calculate critical convective time step.
     * \param [in] aPrimal   primal state database
     * \param [in] aVelocity velocity field
     * \return critical convective time step
     *
     **********************************************************************************/
    Plato::Scalar
    calculateCriticalConvectiveTimeStep
    (const Plato::Primal & aPrimal,
     const Plato::ScalarVector & aVelocity)
    {
        auto tElemCharSize = aPrimal.vector("element characteristic size");
        auto tVelMag = Plato::Fluids::calculate_magnitude_convective_velocity<mNumNodesPerCell>(mSpatialModel, aVelocity);
        auto tCriticalTimeStep = Plato::Fluids::calculate_critical_convective_time_step
            (mSpatialModel, tElemCharSize, tVelMag, mTimeStepSafetyFactor);
        return tCriticalTimeStep;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar calculateCriticalDiffusionTimeStep
     *
     * \brief Calculate critical diffusive time step.
     * \param [in] aPrimal primal state database
     * \return critical diffusive time step
     *
     **********************************************************************************/
    Plato::Scalar
    calculateCriticalDiffusionTimeStep
    (const Plato::Primal & aPrimal)
    {
        auto tElemCharSize = aPrimal.vector("element characteristic size");
        auto tKinematicViscocity = aPrimal.scalar("kinematic viscocity");
        auto tThermalDiffusivity = aPrimal.scalar("thermal diffusivity");
        auto tCriticalTimeStep = Plato::Fluids::calculate_critical_diffusion_time_step
            (tKinematicViscocity, tThermalDiffusivity, tElemCharSize, mTimeStepSafetyFactor);
        return tCriticalTimeStep;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar calculateCriticalTimeStepUpperBound
     *
     * \brief Calculate critical time step upper bound.
     * \param [in] aPrimal primal state database
     * \return critical time step upper bound
     *
     **********************************************************************************/
    inline Plato::Scalar
    calculateCriticalTimeStepUpperBound
    (const Plato::Primal &aPrimal)
    {
        auto tElemCharSize = aPrimal.vector("element characteristic size");
        auto tVelLowerBound = aPrimal.scalar("critical velocity lower bound");
        auto tOutput = Plato::Fluids::calculate_critical_time_step_upper_bound(tVelLowerBound, tElemCharSize);
        return tOutput;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector criticalTimeStep
     *
     * \brief Calculate critical time step.
     * \param [in] aPrimal primal state database
     * \param [in] aVelocity velocity field
     * \return critical time step
     *
     **********************************************************************************/
    Plato::ScalarVector
    criticalTimeStep
    (const Plato::Primal & aPrimal,
     const Plato::ScalarVector & aVelocity)
    {
        Plato::ScalarVector tCriticalTimeStep("critical time step", 1);
        auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);

        tHostCriticalTimeStep(0) = this->calculateCriticalConvectiveTimeStep(aPrimal, aVelocity);
        if(mCalculateHeatTransfer)
        {
            auto tCriticalDiffusionTimeStep = this->calculateCriticalDiffusionTimeStep(aPrimal);
            auto tMinCriticalTimeStep = std::min(tCriticalDiffusionTimeStep, tHostCriticalTimeStep(0));
            tHostCriticalTimeStep(0) = tMinCriticalTimeStep;
        }

        auto tCriticalTimeStepUpperBound = this->calculateCriticalTimeStepUpperBound(aPrimal);
        auto tMinCriticalTimeStep = std::min(tCriticalTimeStepUpperBound, tHostCriticalTimeStep(0));
        tHostCriticalTimeStep(0) = mTimeStepDamping * tMinCriticalTimeStep;
        mCriticalTimeStepHistory.push_back(tHostCriticalTimeStep(0));
        Kokkos::deep_copy(tCriticalTimeStep, tHostCriticalTimeStep);

        return tCriticalTimeStep;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector initialCriticalTimeStep
     *
     * \brief Calculate initial critical time step.
     * \param [in] aPrimal primal state database
     * \return critical time step
     *
     **********************************************************************************/
    Plato::ScalarVector
    initialCriticalTimeStep
    (const Plato::Primal & aPrimal)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);
        //Plato::post_process_dirichlet_dofs(tBcDofs, tBcValues);
        auto tPreviousVelocity = aPrimal.vector("previous velocity");
        Plato::ScalarVector tInitialVelocity("initial velocity", tPreviousVelocity.size());
        Plato::blas1::update(1.0, tPreviousVelocity, 0.0, tInitialVelocity);
        Plato::enforce_boundary_condition(tBcDofs, tBcValues, tInitialVelocity);
        auto tCriticalTimeStep = this->criticalTimeStep(aPrimal, tInitialVelocity);
        return tCriticalTimeStep;
    }

    /******************************************************************************//**
     * \fn void checkCriticalTimeStep
     *
     * \brief Check critical time step, an runtime error is thrown if an unstable time step is detected.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    void checkCriticalTimeStep
    (const Plato::Primal &aPrimal)
    {
        auto tCriticalTimeStep = aPrimal.vector("critical time step");
        auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);
        Kokkos::deep_copy(tHostCriticalTimeStep, tCriticalTimeStep);
        if(tHostCriticalTimeStep(0) < std::numeric_limits<Plato::Scalar>::epsilon())
        {
            std::ostringstream tOutSStream;
            tOutSStream << tHostCriticalTimeStep(0);
            THROWERR(std::string("Unstable critical time step (dt = '") + tOutSStream.str()
                 + "') detected. Refine the finite element mesh or coarsen the steady state stopping tolerance.")
        }
    }

    /******************************************************************************//**
     * \fn void calculateCriticalTimeStep
     *
     * \brief Calculate critical time step.
     * \param [in\out] aPrimal primal state database
     *
     **********************************************************************************/
    void calculateCriticalTimeStep
    (Plato::Primal & aPrimal)
    {
        auto tIteration = aPrimal.scalar("time step index");
        if(tIteration > 1)
        {
            auto tPreviousVelocity = aPrimal.vector("previous velocity");
            auto tCriticalTimeStep = this->criticalTimeStep(aPrimal, tPreviousVelocity);
            aPrimal.vector("critical time step", tCriticalTimeStep);
        }
        else
        {
            auto tCriticalTimeStep = this->initialCriticalTimeStep(aPrimal);
            aPrimal.vector("critical time step", tCriticalTimeStep);
        }
    }

    /******************************************************************************//**
     * \fn void setDual
     *
     * \brief Set dual state database
     * \param [in\out] aDual dual state database
     *
     **********************************************************************************/
    void setDual
    (Plato::Dual& aDual)
    {
        constexpr auto tCurrentSnapshot = 1u;
        auto tCurrentAdjointVel = Kokkos::subview(mAdjointVelocity, tCurrentSnapshot, Kokkos::ALL());
        auto tCurrentAdjointPred = Kokkos::subview(mAdjointPredictor, tCurrentSnapshot, Kokkos::ALL());
        auto tCurrentAdjointPress = Kokkos::subview(mAdjointPressure, tCurrentSnapshot, Kokkos::ALL());
        aDual.vector("current velocity adjoint", tCurrentAdjointVel);
        aDual.vector("current pressure adjoint", tCurrentAdjointPress);
        aDual.vector("current predictor adjoint", tCurrentAdjointPred);

        constexpr auto tPreviousSnapshot = tCurrentSnapshot - 1u;
        auto tPreviouAdjointVel = Kokkos::subview(mAdjointVelocity, tPreviousSnapshot, Kokkos::ALL());
        auto tPreviousAdjointPred = Kokkos::subview(mAdjointPredictor, tPreviousSnapshot, Kokkos::ALL());
        auto tPreviousAdjointPress = Kokkos::subview(mAdjointPressure, tPreviousSnapshot, Kokkos::ALL());
        aDual.vector("previous velocity adjoint", tPreviouAdjointVel);
        aDual.vector("previous predictor adjoint", tPreviousAdjointPred);
        aDual.vector("previous pressure adjoint", tPreviousAdjointPress);

            if(mCalculateHeatTransfer)
            {
                auto tCurrentAdjointTemp = Kokkos::subview(mAdjointTemperature, tCurrentSnapshot, Kokkos::ALL());
                auto tPreviousAdjointTemp = Kokkos::subview(mAdjointTemperature, tPreviousSnapshot, Kokkos::ALL());
                aDual.vector("current temperature adjoint", tCurrentAdjointTemp);
                aDual.vector("previous temperature adjoint", tPreviousAdjointTemp);
            }
    }

    /******************************************************************************//**
     * \fn void saveDual
     *
     * \brief Set previous dual state for the next iteration.
     * \param [in\out] aDual dual state database
     *
     **********************************************************************************/
    void saveDual
    (Plato::Dual & aDual)
    {
        constexpr auto tPreviousSnapshot = 0u;

        auto tCurrentAdjointVelocity = aDual.vector("current velocity adjoint");
        auto tPreviousAdjointVelocity = Kokkos::subview(mAdjointVelocity, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentAdjointVelocity, tPreviousAdjointVelocity);

        auto tCurrentAdjointPressure = aDual.vector("current pressure adjoint");
        auto tPreviousAdjointPressure = Kokkos::subview(mAdjointPressure, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentAdjointPressure, tPreviousAdjointPressure);

        auto tCurrentAdjointPredictor = aDual.vector("current predictor adjoint");
        auto tPreviousAdjointPredictor = Kokkos::subview(mAdjointPredictor, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentAdjointPredictor, tPreviousAdjointPredictor);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentAdjointTemperature = aDual.vector("current temperature adjoint");
            auto tPreviousAdjointTemperature = Kokkos::subview(mAdjointTemperature, tPreviousSnapshot, Kokkos::ALL());
            Plato::blas1::copy(tCurrentAdjointTemperature, tPreviousAdjointTemperature);
        }
    }

    /******************************************************************************//**
     * \fn void savePrimal
     *
     * \brief Set previous primal state for the next iteration.
     * \param [in\out] aPrimal primal state database
     *
     **********************************************************************************/
    void savePrimal
    (Plato::Primal & aPrimal)
    {
        constexpr auto tPreviousSnapshot = 0u;

        auto tCurrentVelocity = aPrimal.vector("current velocity");
        auto tPreviousVelocity = Kokkos::subview(mVelocity, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentVelocity, tPreviousVelocity);

        auto tCurrentPressure = aPrimal.vector("current pressure");
        auto tPreviousPressure = Kokkos::subview(mPressure, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentPressure, tPreviousPressure);

        auto tCurrentPredictor = aPrimal.vector("current predictor");
        auto tPreviousPredictor = Kokkos::subview(mPredictor, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentPredictor, tPreviousPredictor);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentTemperature = aPrimal.vector("current temperature");
            auto tPreviousTemperature = Kokkos::subview(mTemperature, tPreviousSnapshot, Kokkos::ALL());
            Plato::blas1::copy(tCurrentTemperature, tPreviousTemperature);
        }
    }

    /******************************************************************************//**
     * \fn void setPrimal
     *
     * \brief Set previous and current primal states.
     * \param [in\out] aPrimal primal state database
     *
     **********************************************************************************/
    void setPrimal
    (Plato::Primal & aPrimal)
    {
        constexpr Plato::OrdinalType tCurrentState = 1;
        auto tCurrentVel   = Kokkos::subview(mVelocity, tCurrentState, Kokkos::ALL());
        auto tCurrentPred  = Kokkos::subview(mPredictor, tCurrentState, Kokkos::ALL());
        auto tCurrentPress = Kokkos::subview(mPressure, tCurrentState, Kokkos::ALL());
        aPrimal.vector("current velocity", tCurrentVel);
        aPrimal.vector("current pressure", tCurrentPress);
        aPrimal.vector("current predictor", tCurrentPred);

        constexpr auto tPrevState = tCurrentState - 1;
        auto tPreviouVel = Kokkos::subview(mVelocity, tPrevState, Kokkos::ALL());
        auto tPreviousPred = Kokkos::subview(mPredictor, tPrevState, Kokkos::ALL());
        auto tPreviousPress = Kokkos::subview(mPressure, tPrevState, Kokkos::ALL());
        aPrimal.vector("previous velocity", tPreviouVel);
        aPrimal.vector("previous predictor", tPreviousPred);
        aPrimal.vector("previous pressure", tPreviousPress);

        auto tCurrentTemp = Kokkos::subview(mTemperature, tCurrentState, Kokkos::ALL());
        aPrimal.vector("current temperature", tCurrentTemp);
        auto tPreviousTemp = Kokkos::subview(mTemperature, tPrevState, Kokkos::ALL());
        aPrimal.vector("previous temperature", tPreviousTemp);
    }

    /******************************************************************************//**
     * \fn void printCorrectorSolverHeader
     *
     * \brief Print diagnostic header for velocity corrector solver.
     *
     **********************************************************************************/
    void printCorrectorSolverHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << "*                           Momentum Corrector Solver                               *\n";
                tMsg << "-------------------------------------------------------------------------------------\n";
                Plato::io::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void updateCorrector
     *
     * \brief Solve for current velocity field using the Newton method.
     * \param [in]     aControl control/optimization variables
     * \param [in\out] aPrimal  primal state database
     *
     **********************************************************************************/
    void updateCorrector
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aPrimal)
    {
        this->printCorrectorSolverHeader();
        this->printNewtonHeader();

        auto tCurrentVelocity = aPrimal.vector("current velocity");
        Plato::blas1::fill(0.0, tCurrentVelocity);

        // calculate current residual and jacobian matrix
        auto tJacobian = mCorrectorResidual.gradientCurrentVel(aControl, aPrimal);

        // apply constraints
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);
        //Plato::post_process_dirichlet_dofs(tBcDofs, tBcValues);

        // create linear solver
        if( mInputs.isSublist("Linear Solver") == false )
        { THROWERR("Parameter list 'Linear Solver' is not defined.") }
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);

        // set initial guess for current velocity
        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0, tInitialNormResidual = 0.0;
        Plato::ScalarVector tDeltaCorrector("delta corrector", tCurrentVelocity.size());
        while(true)
        {
            aPrimal.scalar("newton iteration", tIteration);

            auto tResidual = mCorrectorResidual.value(aControl, aPrimal);
            Plato::blas1::scale(-1.0, tResidual);
            Plato::blas1::fill(0.0, tDeltaCorrector);
            tSolver->solve(*tJacobian, tDeltaCorrector, tResidual);
            Plato::blas1::update(1.0, tDeltaCorrector, 1.0, tCurrentVelocity);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            if( !std::isfinite(tNormResidual) )
            { THROWERR("The norm of the residual is not a finite number") }
            auto tNormStep = Plato::blas1::norm(tDeltaCorrector);
            if( !std::isfinite(tNormStep) )
            { THROWERR("The norm of the step is not a finite number") }
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
                tInitialNormResidual = tNormResidual;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aPrimal.scalar("norm step", tNormStep);
            tNormResidual = tNormResidual / tInitialNormResidual;
            aPrimal.scalar("norm residual", tNormResidual);

            this->printNewtonDiagnostics(aPrimal);
            if(tNormStep <= mCorrectorTolerance || tIteration >= mMaxCorrectorIterations)
            {
                break;
            }

            tIteration++;
        }
        Plato::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentVelocity);
    }

    /******************************************************************************//**
     * \fn void printNewtonHeader
     *
     * \brief Print Newton solver header to diagnostics text file.
     *
     **********************************************************************************/
    void printNewtonHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "Iteration" << std::setw(16) << "Delta(u*)" << std::setw(18) << "Residual\n";
                Plato::io::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void printPredictorSolverHeader
     *
     * \brief Print velocity predictor solver header to diagnostics text file.
     *
     **********************************************************************************/
    void printPredictorSolverHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << "*                           Momentum Predictor Solver                               *\n";
                tMsg << "-------------------------------------------------------------------------------------\n";
                Plato::io::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void printNewtonDiagnostics
     *
     * \brief Print Newton's solver diagnostics to text file.
     * \param [in] aPrimal  primal state database
     *
     **********************************************************************************/
    void printNewtonDiagnostics
    (Plato::Primal & aPrimal)
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                auto tNormStep = aPrimal.scalar("norm step");
                auto tNormResidual = aPrimal.scalar("norm residual");
                Plato::OrdinalType tIteration = aPrimal.scalar("newton iteration");
                tMsg << tIteration << std::setw(24) << std::scientific << tNormStep << std::setw(18) << tNormResidual << "\n";
                Plato::io::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void updatePredictor
     *
     * \brief Solve for current velocity predictor field using the Newton method.
     * \param [in]     aControl control/optimization variables
     * \param [in\out] aPrimal  primal state database
     *
     **********************************************************************************/
    void updatePredictor
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aStates)
    {
        this->printPredictorSolverHeader();
        this->printNewtonHeader();

        auto tCurrentPredictor = aStates.vector("current predictor");
        Plato::blas1::fill(0.0, tCurrentPredictor);

        // calculate current residual and jacobian matrix
        auto tResidual = mPredictorResidual.value(aControl, aStates);
        auto tJacobian = mPredictorResidual.gradientPredictor(aControl, aStates);

        // create linear solver
        if( mInputs.isSublist("Linear Solver") == false )
        { THROWERR("Parameter list 'Linear Solver' is not defined.") }
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);

        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0, tInitialNormResidual = 0.0;
        Plato::ScalarVector tDeltaPredictor("delta predictor", tCurrentPredictor.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            Plato::blas1::fill(0.0, tDeltaPredictor);
            Plato::blas1::scale(-1.0, tResidual);
            tSolver->solve(*tJacobian, tDeltaPredictor, tResidual);
            Plato::blas1::update(1.0, tDeltaPredictor, 1.0, tCurrentPredictor);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            if( !std::isfinite(tNormResidual) )
            { THROWERR("The norm of the residual is not a finite number") }
            auto tNormStep = Plato::blas1::norm(tDeltaPredictor);
            if( !std::isfinite(tNormStep) )
            { THROWERR("The norm of the step is not a finite number") }
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
                tInitialNormResidual = tNormResidual;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);
            tNormResidual = tNormResidual / tInitialNormResidual;
            aStates.scalar("norm residual", tNormResidual);

            this->printNewtonDiagnostics(aStates);
            if(tNormStep <= mPredictorTolerance || tIteration >= mMaxPredictorIterations)
            {
                break;
            }

            tResidual = mPredictorResidual.value(aControl, aStates);

            tIteration++;
        }
    }

    /******************************************************************************//**
     * \fn void printPressureSolverHeader
     *
     * \brief Print pressure solver header to diagnostics text file.
     *
     **********************************************************************************/
    void printPressureSolverHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << "*                                Pressure Solver                                    *\n";
                tMsg << "-------------------------------------------------------------------------------------\n";
                Plato::io::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void updatePressure
     *
     * \brief Solve for current pressure field using the Newton method.
     * \param [in]     aControl control/optimization variables
     * \param [in\out] aPrimal  primal state database
     *
     **********************************************************************************/
    void updatePressure
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aStates)
    {
        this->printPressureSolverHeader();
        this->printNewtonHeader();

        auto tCurrentPressure = aStates.vector("current pressure");
        Plato::blas1::fill(0.0, tCurrentPressure);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mPressureEssentialBCs.get(tBcDofs, tBcValues);

        // create linear solver
        if( mInputs.isSublist("Linear Solver") == false )
        { THROWERR("Parameter list 'Linear Solver' is not defined.") }
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumPressDofsPerNode);

        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0, tInitialNormResidual = 0.0;
        Plato::ScalarVector tDeltaPressure("delta pressure", tCurrentPressure.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            auto tResidual = mPressureResidual.value(aControl, aStates);
            Plato::blas1::scale(-1.0, tResidual);
            auto tJacobian = mPressureResidual.gradientCurrentPress(aControl, aStates);

            Plato::Scalar tScale = (tIteration == 1) ? 1.0 : 0.0;
            Plato::apply_constraints<mNumPressDofsPerNode>(tBcDofs, tBcValues, tJacobian, tResidual, tScale);
            Plato::blas1::fill(0.0, tDeltaPressure);
            tSolver->solve(*tJacobian, tDeltaPressure, tResidual);
            Plato::blas1::update(1.0, tDeltaPressure, 1.0, tCurrentPressure);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            if( !std::isfinite(tNormResidual) )
            { THROWERR("The norm of the residual is not a finite number") }
            auto tNormStep = Plato::blas1::norm(tDeltaPressure);
            if( !std::isfinite(tNormStep) )
            { THROWERR("The norm of the step is not a finite number") }
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
                tInitialNormResidual = tNormResidual;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);
            tNormResidual = tNormResidual / tInitialNormResidual;
            aStates.scalar("norm residual", tNormResidual);

            this->printNewtonDiagnostics(aStates);
            if(tNormStep <= mPressureTolerance || tIteration >= mMaxPressureIterations)
            //if(tNormResidual <= mPressureTolerance || tIteration >= mMaxPressureIterations)
            {
                break;
            }

            tIteration++;
        }
    }

    /******************************************************************************//**
     * \fn void printTemperatureSolverHeader
     *
     * \brief Print temperature solver header to diagnostics text file.
     *
     **********************************************************************************/
    void printTemperatureSolverHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << "*                             Temperature Solver                                    *\n";
                tMsg << "-------------------------------------------------------------------------------------\n";
                Plato::io::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void updateTemperature
     *
     * \brief Solve for current temperature field using the Newton method.
     * \param [in]     aControl control/optimization variables
     * \param [in\out] aPrimal  primal state database
     *
     **********************************************************************************/
    void updateTemperature
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aStates)
    {
        this->printTemperatureSolverHeader();
        this->printNewtonHeader();

        auto tCurrentTemperature = aStates.vector("current temperature");
        Plato::blas1::fill(0.0, tCurrentTemperature);

        // apply constraints
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mTemperatureEssentialBCs.get(tBcDofs, tBcValues);

        // solve energy equation (consistent or mass lumped)
        if( mInputs.isSublist("Linear Solver") == false )
        { THROWERR("Parameter list 'Linear Solver' is not defined.") }
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumTempDofsPerNode);

        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0, tInitialNormResidual = 0.0;
        Plato::ScalarVector tDeltaTemperature("delta temperature", tCurrentTemperature.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            // update residual and jacobian
            auto tResidual = mTemperatureResidual->value(aControl, aStates);
            Plato::blas1::scale(-1.0, tResidual);
            auto tJacobian = mTemperatureResidual->gradientCurrentTemp(aControl, aStates);

            // solve system of equations
            Plato::Scalar tScale = (tIteration == 1) ? 1.0 : 0.0;
            Plato::apply_constraints<mNumTempDofsPerNode>(tBcDofs, tBcValues, tJacobian, tResidual, tScale);
            Plato::blas1::fill(0.0, tDeltaTemperature);
            tSolver->solve(*tJacobian, tDeltaTemperature, tResidual);
            Plato::blas1::update(1.0, tDeltaTemperature, 1.0, tCurrentTemperature);

            // calculate stopping criteria
            auto tNormResidual = Plato::blas1::norm(tResidual);
            if( !std::isfinite(tNormResidual) )
            { THROWERR("The norm of the residual is not a finite number") }
            auto tNormStep = Plato::blas1::norm(tDeltaTemperature);
            if( !std::isfinite(tNormStep) )
            { THROWERR("The norm of the step is not a finite number") }
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
                tInitialNormResidual = tNormResidual;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);
            tNormResidual = tNormResidual / tInitialNormResidual;
            aStates.scalar("norm residual", tNormResidual);

            // check stopping criteria
            this->printNewtonDiagnostics(aStates);
            if(tNormResidual < mTemperatureTolerance || tIteration >= mMaxTemperatureIterations)
            {
                break;
            }

            tIteration++;
        }
    }

    /******************************************************************************//**
     * \fn void updatePredictorAdjoint
     *
     * \brief Solve for the current velocity predictor adjoint field using the Newton method.
     * \param [in]     aControl        control/optimization variables
     * \param [in]     aCurrentPrimal  current primal state database
     * \param [in]     aPreviousPrimal previous primal state database
     * \param [in/out] aDual           current dual state database
     *
     **********************************************************************************/
    void updatePredictorAdjoint
    (const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Primal       & aPreviousPrimal,
           Plato::Dual         & aDual)
    {
        auto tCurrentPredictorAdjoint = aDual.vector("current predictor adjoint");
        Plato::blas1::fill(0.0, tCurrentPredictorAdjoint);

        // add PDE contribution from current state to right hand side adjoint vector
        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tJacCorrectorResWrtPredictor = mCorrectorResidual.gradientPredictor(aControl, aCurrentPrimal);
        Plato::ScalarVector tRHS("right hand side", tCurrentVelocityAdjoint.size());
        Plato::MatrixTimesVectorPlusVector(tJacCorrectorResWrtPredictor, tCurrentVelocityAdjoint, tRHS);

        auto tCurrentPressureAdjoint = aDual.vector("current pressure adjoint");
        auto tGradResPressWrtPredictor = mPressureResidual.gradientPredictor(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPredictor, tCurrentPressureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        // solve adjoint system of equations
        if( mInputs.isSublist("Linear Solver") == false )
        { THROWERR("Parameter list 'Linear Solver' is not defined.") }
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aControl, aCurrentPrimal);
        tSolver->solve(*tJacobianPredictor, tCurrentPredictorAdjoint, tRHS);
    }

    /******************************************************************************//**
     * \fn void updatePressureAdjoint
     *
     * \brief Solve for the current pressure adjoint field using the Newton method.
     * \param [in]     aControl        control/optimization variables
     * \param [in]     aCurrentPrimal  current primal state database
     * \param [in]     aPreviousPrimal previous primal state database
     * \param [in/out] aDual           current dual state database
     *
     **********************************************************************************/
    void updatePressureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Primal       & aPreviousPrimal,
           Plato::Dual         & aDual)
    {
        // initialize data
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimal.scalar("time step index"));
        auto tCurrentPressAdjoint = aDual.vector("current pressure adjoint");
        Plato::blas1::fill(0.0, tCurrentPressAdjoint);

        // add objective function contribution to right hand side adjoint vector
        auto tNumDofs = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tRightHandSide("right hand side vector", tNumDofs);
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tPartialObjWrtCurrentPressure = mCriteria[aName]->gradientCurrentPress(aControl, aCurrentPrimal);
            Plato::blas1::update(1.0, tPartialObjWrtCurrentPressure, 0.0, tRightHandSide);
        }

        // add PDE contribution from current state to right hand side adjoint vector
        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tJacCorrectorResWrtCurPress = mCorrectorResidual.gradientCurrentPress(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tJacCorrectorResWrtCurPress, tCurrentVelocityAdjoint, tRightHandSide);

        // add PDE contribution from previous state to right hand side adjoint vector
        if(tCurrentTimeStepIndex != mNumForwardSolveTimeSteps)
        {
            auto tPreviousPressureAdjoint = aDual.vector("previous pressure adjoint");
            auto tJacPressResWrtPrevPress = mPressureResidual.gradientPreviousPress(aControl, aPreviousPrimal);
            Plato::MatrixTimesVectorPlusVector(tJacPressResWrtPrevPress, tPreviousPressureAdjoint, tRightHandSide);

            auto tPreviousVelocityAdjoint = aDual.vector("previous velocity adjoint");
            auto tJacCorrectorResWrtPrevVel = mCorrectorResidual.gradientPreviousPress(aControl, aPreviousPrimal);
            Plato::MatrixTimesVectorPlusVector(tJacCorrectorResWrtPrevVel, tPreviousVelocityAdjoint, tRightHandSide);
        }
        Plato::blas1::scale(-1.0, tRightHandSide);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mPressureEssentialBCs.get(tBcDofs, tBcValues);
        Plato::blas1::fill(0.0, tBcValues);

        // solve adjoint system of equations
        if( mInputs.isSublist("Linear Solver") == false )
        { THROWERR("Parameter list 'Linear Solver' is not defined.") }
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumPressDofsPerNode);
        auto tJacPressResWrtCurPress = mPressureResidual.gradientCurrentPress(aControl, aCurrentPrimal);
        Plato::apply_constraints<mNumPressDofsPerNode>(tBcDofs, tBcValues, tJacPressResWrtCurPress, tRightHandSide);
        tSolver->solve(*tJacPressResWrtCurPress, tCurrentPressAdjoint, tRightHandSide);
    }

    /******************************************************************************//**
     * \fn void updateTemperatureAdjoint
     *
     * \brief Solve for the current temperature adjoint field using the Newton method.
     * \param [in]     aControl        control/optimization variables
     * \param [in]     aCurrentPrimal  current primal state database
     * \param [in]     aPreviousPrimal previous primal state database
     * \param [in/out] aDual           current dual state database
     *
     **********************************************************************************/
    void updateTemperatureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Primal       & aPreviousPrimal,
           Plato::Dual         & aDual)
    {
        // initialize data
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimal.scalar("time step index"));
        auto tCurrentTempAdjoint = aDual.vector("current temperature adjoint");
        Plato::blas1::fill(0.0, tCurrentTempAdjoint);

        // add objective function contribution to right hand side adjoint vector
        auto tNumDofs = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tRightHandSide("right hand side vector", tNumDofs);
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tPartialObjWrtCurrentTemperature = mCriteria[aName]->gradientCurrentTemp(aControl, aCurrentPrimal);
            Plato::blas1::update(1.0, tPartialObjWrtCurrentTemperature, 0.0, tRightHandSide);
        }

        // add PDE contribution from previous state to right hand side adjoint vector
        if(tCurrentTimeStepIndex != mNumForwardSolveTimeSteps)
        {
            auto tPreviousPredAdjoint = aDual.vector("previous predictor adjoint");
            auto tGradResPredWrtPreviousTemp = mPredictorResidual.gradientPreviousTemp(aControl, aPreviousPrimal);
            Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPreviousTemp, tPreviousPredAdjoint, tRightHandSide);

            if(mCalculateHeatTransfer)
            {
                auto tPreviousTempAdjoint = aDual.vector("previous temperature adjoint");
                auto tJacTempResWrtPreviousTemp = mTemperatureResidual->gradientPreviousTemp(aControl, aPreviousPrimal);
                Plato::MatrixTimesVectorPlusVector(tJacTempResWrtPreviousTemp, tPreviousTempAdjoint, tRightHandSide);
            }
        }
        Plato::blas1::scale(-1.0, tRightHandSide);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mTemperatureEssentialBCs.get(tBcDofs, tBcValues);
        Plato::blas1::fill(0.0, tBcValues);

        // solve adjoint system of equations
        if( mInputs.isSublist("Linear Solver") == false )
        { THROWERR("Parameter list 'Linear Solver' is not defined.") }
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumTempDofsPerNode);
        auto tJacobianCurrentTemp = mTemperatureResidual->gradientCurrentTemp(aControl, aCurrentPrimal);
        Plato::apply_constraints<mNumTempDofsPerNode>(tBcDofs, tBcValues, tJacobianCurrentTemp, tRightHandSide);
        tSolver->solve(*tJacobianCurrentTemp, tCurrentTempAdjoint, tRightHandSide);
    }

    /******************************************************************************//**
     * \fn void updateCorrectorAdjoint
     *
     * \brief Solve for the current velocity adjoint field using the Newton method.
     * \param [in]     aControl        control/optimization variables
     * \param [in]     aCurrentPrimal  current primal state database
     * \param [in]     aPreviousPrimal previous primal state database
     * \param [in/out] aDual           current dual state database
     *
     **********************************************************************************/
    void updateCorrectorAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimalState,
     const Plato::Primal       & aPreviousPrimalState,
           Plato::Dual         & aDual)
    {
        // initialize data
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimalState.scalar("time step index"));
        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        Plato::blas1::fill(0.0, tCurrentVelocityAdjoint);

        // add objective function contribution to right hand side adjoint vector
        auto tNumDofs = mSpatialModel.Mesh.nverts() * mNumVelDofsPerNode;
        Plato::ScalarVector tRightHandSide("right hand side vector", tNumDofs);
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tPartialObjFuncWrtCurrentVel = mCriteria[aName]->gradientCurrentVel(aControl, aCurrentPrimalState);
            Plato::blas1::update(1.0, tPartialObjFuncWrtCurrentVel, 0.0, tRightHandSide);
        }

        // add PDE contribution from current state to right hand side adjoint vector
        if(mCalculateHeatTransfer)
        {
            auto tCurrentTempAdjoint = aDual.vector("current temperature adjoint");
            auto tJacTempResWrtCurVel = mTemperatureResidual->gradientCurrentVel(aControl, aCurrentPrimalState);
            Plato::MatrixTimesVectorPlusVector(tJacTempResWrtCurVel, tCurrentTempAdjoint, tRightHandSide);
        }


        // add PDE contribution from previous state to right hand side adjoint vector
        if(tCurrentTimeStepIndex != mNumForwardSolveTimeSteps)
        {
            auto tPreviousPredictorAdjoint = aDual.vector("previous predictor adjoint");
            auto tJacPredResWrtPrevVel = mPredictorResidual.gradientPreviousVel(aControl, aPreviousPrimalState);
            Plato::MatrixTimesVectorPlusVector(tJacPredResWrtPrevVel, tPreviousPredictorAdjoint, tRightHandSide);

            auto tPreviousPressureAdjoint = aDual.vector("previous pressure adjoint");
            auto tJacPressResWrtPrevVel = mPressureResidual.gradientPreviousVel(aControl, aPreviousPrimalState);
            Plato::MatrixTimesVectorPlusVector(tJacPressResWrtPrevVel, tPreviousPressureAdjoint, tRightHandSide);

            auto tPreviousVelocityAdjoint = aDual.vector("previous velocity adjoint");
            auto tJacCorrectorResWrtPrevVel = mCorrectorResidual.gradientPreviousVel(aControl, aPreviousPrimalState);
            Plato::MatrixTimesVectorPlusVector(tJacCorrectorResWrtPrevVel, tPreviousVelocityAdjoint, tRightHandSide);
        }
        Plato::blas1::scale(-1.0, tRightHandSide);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);
        Plato::blas1::fill(0.0, tBcValues);

        // solve adjoint system of equations
        if( mInputs.isSublist("Linear Solver") == false )
        { THROWERR("Parameter list 'Linear Solver' is not defined.") }
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);
        auto tJacCorrectorResWrtCurVel = mCorrectorResidual.gradientCurrentVel(aControl, aCurrentPrimalState);
        Plato::set_dofs_values(tBcDofs, tRightHandSide, 0.0);
        tSolver->solve(*tJacCorrectorResWrtCurVel, tCurrentVelocityAdjoint, tRightHandSide);
    }

    /******************************************************************************//**
     * \fn void updateTotalDerivativeWrtControl
     *
     * \brief Update total derivative of the criterion with respect to control variables.
     * \param [in]     aName            criterion name
     * \param [in]     aControl         control/optimization variables
     * \param [in]     aCurrentPrimal   current primal state database
     * \param [in]     aDual            current dual state database
     * \param [in/out] aTotalDerivative total derivative
     *
     **********************************************************************************/
    void  updateTotalDerivativeWrtControl
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Dual         & aDual,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimal.scalar("time step index"));
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tGradCriterionWrtControl = mCriteria[aName]->gradientControl(aControl, aCurrentPrimal);
            Plato::blas1::update(1.0, tGradCriterionWrtControl, 1.0, aTotalDerivative);
        }

        auto tCurrentPredictorAdjoint = aDual.vector("current predictor adjoint");
        auto tGradResPredWrtControl = mPredictorResidual.gradientControl(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtControl, tCurrentPredictorAdjoint, aTotalDerivative);

        auto tCurrentPressureAdjoint = aDual.vector("current pressure adjoint");
        auto tGradResPressWrtControl = mPressureResidual.gradientControl(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtControl, tCurrentPressureAdjoint, aTotalDerivative);

        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tGradResVelWrtControl = mCorrectorResidual.gradientControl(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtControl, tCurrentVelocityAdjoint, aTotalDerivative);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentTemperatureAdjoint = aDual.vector("current temperature adjoint");
            auto tGradResTempWrtControl = mTemperatureResidual->gradientControl(aControl, aCurrentPrimal);
            Plato::MatrixTimesVectorPlusVector(tGradResTempWrtControl, tCurrentTemperatureAdjoint, aTotalDerivative);
        }
    }

    /******************************************************************************//**
     * \fn void updateTotalDerivativeWrtConfig
     *
     * \brief Update total derivative of the criterion with respect to the configuration variables.
     * \param [in]     aName            criterion name
     * \param [in]     aControl         control/optimization variables
     * \param [in]     aCurrentPrimal   current primal state database
     * \param [in]     aDual            current dual state database
     * \param [in/out] aTotalDerivative total derivative
     *
     **********************************************************************************/
    void updateTotalDerivativeWrtConfig
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Dual         & aDual,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimal.scalar("time step index"));
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tGradCriterionWrtConfig = mCriteria[aName]->gradientConfig(aControl, aCurrentPrimal);
            Plato::blas1::update(1.0, tGradCriterionWrtConfig, 1.0, aTotalDerivative);
        }

        auto tCurrentPredictorAdjoint = aDual.vector("current predictor adjoint");
        auto tGradResPredWrtConfig = mPredictorResidual.gradientConfig(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtConfig, tCurrentPredictorAdjoint, aTotalDerivative);

        auto tCurrentPressureAdjoint = aDual.vector("current pressure adjoint");
        auto tGradResPressWrtConfig = mPressureResidual.gradientConfig(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtConfig, tCurrentPressureAdjoint, aTotalDerivative);

        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tGradResVelWrtConfig = mCorrectorResidual.gradientConfig(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtConfig, tCurrentVelocityAdjoint, aTotalDerivative);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentTemperatureAdjoint = aDual.vector("current temperature adjoint");
            auto tGradResTempWrtConfig = mTemperatureResidual->gradientConfig(aControl, aCurrentPrimal);
            Plato::MatrixTimesVectorPlusVector(tGradResTempWrtConfig, tCurrentTemperatureAdjoint, aTotalDerivative);
        }
    }
};
// class QuasiImplicit

}
// namespace Fluids

}
// namespace Plato

#include "hyperbolic/IncompressibleFluids.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<3>>;
#endif

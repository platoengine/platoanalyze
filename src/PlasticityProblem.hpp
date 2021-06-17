/*
 * PlasticityProblem.hpp
 *
 *  Created on: Mar 2, 2020
 */

/*
  NOTES:
  1.  If mGlobalStates is a 3D view [NumSequence, NumSteps, NumDofs] this will waste memory,
      since NumSteps varies from sequence step to sequence step.
*/

#pragma once

#include <memory>

#include "BLAS1.hpp"
#include "BLAS2.hpp"
#include "Solutions.hpp"
#include "UtilsOmegaH.hpp"
#include "SpatialModel.hpp"
#include "EssentialBCs.hpp"
#include "PlatoSequence.hpp"
#include "NewtonRaphsonSolver.hpp"
#include "PlatoAbstractProblem.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "alg/PlatoAbstractSolver.hpp"
#include "PathDependentAdjointSolver.hpp"
#include "InfinitesimalStrainPlasticity.hpp"
#include "InfinitesimalStrainThermoPlasticity.hpp"
#include "PathDependentScalarFunctionFactory.hpp"
#include "elliptic/updated_lagrangian/LagrangianUpdate.hpp"
#include "TimeData.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Plasticity problem manager.  This interface is responsible for the
 * evaluation of the criteria value, criteria sensitivities, and residual.
 *
 * \tparam PhysicsT physics type, e.g. Plato::InfinitesimalStrainPlasticity
*******************************************************************************/
template<typename PhysicsT>
class PlasticityProblem : public Plato::AbstractProblem
{

// private member data
private:

    using Criterion = std::shared_ptr<Plato::LocalScalarFunctionInc>;
    using Criteria  = std::map<std::string, Criterion>;

    static constexpr auto mSpaceDim                = PhysicsT::mSpaceDim;             /*!< spatial dimensions*/
    static constexpr auto mNumNodesPerCell         = PhysicsT::mNumNodesPerCell;      /*!< number of nodes per cell*/
    static constexpr auto mPressureDofOffset       = PhysicsT::mPressureDofOffset;    /*!< number of pressure dofs offset*/
    static constexpr auto mTemperatureDofOffset    = PhysicsT::mTemperatureDofOffset; /*!< number of temperature dofs offset*/
    static constexpr auto mNumGlobalDofsPerNode    = PhysicsT::mNumDofsPerNode;       /*!< number of global degrees of freedom per node*/
    static constexpr auto mNumGlobalDofsPerCell    = PhysicsT::mNumDofsPerCell;       /*!< number of global degrees of freedom per cell (i.e. element)*/
    static constexpr auto mNumLocalDofsPerCell     = PhysicsT::mNumLocalDofsPerCell;  /*!< number of local degrees of freedom per cell (i.e. element)*/
    static constexpr auto mNumPressGradDofsPerCell = PhysicsT::mNumNodeStatePerCell;  /*!< number of projected pressure gradient degrees of freedom per cell (i.e. element)*/
    static constexpr auto mNumPressGradDofsPerNode = PhysicsT::mNumNodeStatePerNode;  /*!< number of projected pressure gradient degrees of freedom per node*/
    static constexpr auto mNumConfigDofsPerCell    = mSpaceDim * mNumNodesPerCell;    /*!< number of configuration (i.e. coordinates) degrees of freedom per cell (i.e. element) */

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    Plato::Sequence<mSpaceDim> mSequence;

    // Required
    using PlasticityT = typename PhysicsT::LocalPhysicsT;
    using ProjectorT  = typename Plato::Projection<mSpaceDim, PhysicsT::mNumDofsPerNode, PhysicsT::mPressureDofOffset>;
    std::shared_ptr<Plato::VectorFunctionVMS<ProjectorT>> mProjectionEquation;  /*!< global pressure gradient projection interface */
    std::shared_ptr<Plato::GlobalVectorFunctionInc<PhysicsT>> mGlobalEquation;  /*!< global equality constraint interface */
    std::shared_ptr<Plato::LocalVectorFunctionInc<PlasticityT>> mLocalEquation; /*!< local equality constraint interface */
    std::shared_ptr<Plato::TimeData> mTimeData;                                 /*!< active time data object */
    std::vector<std::shared_ptr<Plato::TimeData>> mSequenceTimeData;            /*!< sequence time data objects */
    std::string mPhysics; /*!< simulated physics */
    std::string mPDE; /*!< simulated pde */

    // Optional
    Criteria mCriteria;

    Plato::Scalar mPressureScaling;               /*!< pressure term scaling */
    Plato::Scalar mTemperatureScaling;            /*!< temperature term scaling */
    Plato::Scalar mReferenceTemperature;          /*!< reference temperature */
    Plato::Scalar mInitialNormResidual;           /*!< initial norm of global residual */
    Plato::Scalar mDispControlConstant;           /*!< displacement control constant */

    Plato::ScalarVector mControl;                 /*!< control variables for output */
    Plato::ScalarVector mPressure;                /*!< projected pressure field */
//    Plato::ScalarMultiVector mLocalStates;        /*!< local state variables */
//    Plato::ScalarMultiVector mGlobalStates;       /*!< global state variables */
//    Plato::ScalarMultiVector mTotalStates;        /*!< total state variables */
//    Plato::ScalarMultiVector mReactionForce;      /*!< reaction */
//    Plato::ScalarMultiVector mProjectedPressGrad; /*!< projected pressure gradient (# Time Steps, # Projected Pressure Gradient dofs) */

    Plato::Solutions mSolutions;

    std::shared_ptr<Plato::EssentialBCs<PhysicsT>> mEssentialBCs; /*!< essential boundary conditions shared pointer */
    Plato::ScalarVector mPreviousStepDirichletValues;            /*!< previous time step values associated with the Dirichlet boundary conditions */
    Plato::ScalarVector mDirichletValues;         /*!< values associated with the Dirichlet boundary conditions */
    Plato::LocalOrdinalVector mDirichletDofs;     /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */

    Plato::WorksetBase<PhysicsT> mWorksetBase;    /*!< assembly routine interface */

    Plato::SolverFactory mLinearSolverFactory;            /*!< linear solver factory */
    std::shared_ptr<Plato::AbstractSolver> mLinearSolver; /*!< linear solver object */

    std::shared_ptr<Plato::NewtonRaphsonSolver<PhysicsT>> mNewtonSolver;         /*!< Newton-Raphson solve interface */
    std::shared_ptr<Plato::PathDependentAdjointSolver<PhysicsT>> mAdjointSolver; /*!< Path-dependent adjoint solver interface */

    bool mStopOptimization; /*!< stops optimization problem if Newton-Raphson solver fails to converge during an optimization run */

    Plato::LagrangianUpdate<PhysicsT> mLagrangianUpdate;

// public functions
public:
    /***************************************************************************//**
     * \brief Plasticity problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputs input parameters database
     * \param [in] aMachine MPI communicator wrapper
    *******************************************************************************/
    PlasticityProblem(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Teuchos::ParameterList& aInputs,
      Comm::Machine& aMachine
    ) :
      mSpatialModel(aMesh, aMeshSets, aInputs),
      mSequence(mSpatialModel, aInputs),
      mLocalEquation(std::make_shared<Plato::LocalVectorFunctionInc<PlasticityT>>(mSpatialModel, mDataMap, aInputs)),
      mPDE(aInputs.get<std::string>("PDE Constraint")),
      mGlobalEquation(std::make_shared<Plato::GlobalVectorFunctionInc<PhysicsT>>(mSpatialModel, mDataMap, aInputs, aInputs.get<std::string>("PDE Constraint"))),
      mProjectionEquation(std::make_shared<Plato::VectorFunctionVMS<ProjectorT>>(mSpatialModel, mDataMap, aInputs, std::string("State Gradient Projection"))),
      mTimeData(std::make_shared<Plato::TimeData>(aInputs)),
      mSequenceTimeData(mSequence.getNumSteps(), mTimeData),
      mPhysics(aInputs.get<std::string>("Physics")),
      mPressureScaling(1.0),
      mTemperatureScaling(1.0),
      mReferenceTemperature(1.0),
      mInitialNormResidual(std::numeric_limits<Plato::Scalar>::max()),
      mDispControlConstant(std::numeric_limits<Plato::Scalar>::min()),
      mPressure("Previous Pressure Field", aMesh.nverts()),
      mSolutions(mPhysics, mPDE, mSequence.getNumSteps()),
      mWorksetBase(aMesh),
      mLinearSolverFactory(aInputs.sublist("Linear Solver")),
      mLinearSolver(mLinearSolverFactory.create(aMesh, aMachine, PhysicsT::mNumDofsPerNode)),
      mNewtonSolver(std::make_shared<Plato::NewtonRaphsonSolver<PhysicsT>>(aMesh, aInputs, mLinearSolver)),
      mAdjointSolver(std::make_shared<Plato::PathDependentAdjointSolver<PhysicsT>>(aMesh, aInputs, mLinearSolver)),
      mStopOptimization(false),
      mEssentialBCs(nullptr),
      mLagrangianUpdate(mSpatialModel)
    {
        this->initialize(aInputs);
    }


    /***************************************************************************//**
     * \brief PLATO Plasticity Problem destructor
    *******************************************************************************/
    virtual ~PlasticityProblem()
    {
    }

    /***************************************************************************//**
     * \brief Append global residual evaluation interface
     * \param [in] aInput global residual evaluation interface
    *******************************************************************************/
    void appendGlobalResidual(const std::shared_ptr<Plato::GlobalVectorFunctionInc<PhysicsT>>& aInput)
    {
        mGlobalEquation = aInput;
    }

    /***************************************************************************//**
     * \brief Read essential (Dirichlet) boundary conditions from the Exodus file.
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aInputs input parameters database
    *******************************************************************************/
    void readEssentialBoundaryConditions(Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Essential Boundary Conditions") == false)
        {
            THROWERR("Plasticity Problem: Essential Boundary Conditions are not defined for this problem.")
        }
        Plato::OrdinalType tPressureDofOffset    = mPressureDofOffset;
        Plato::OrdinalType tTemperatureDofOffset = mTemperatureDofOffset;

        std::map<Plato::OrdinalType, Plato::Scalar> tDofOffsetToScaleFactor;
        tDofOffsetToScaleFactor[tPressureDofOffset]    = mPressureScaling;
        tDofOffsetToScaleFactor[tTemperatureDofOffset] = mTemperatureScaling;

        mEssentialBCs =
        std::make_shared<Plato::EssentialBCs<PhysicsT>>
             (aInputs.sublist("Essential Boundary Conditions", false), mSpatialModel.MeshSets, tDofOffsetToScaleFactor);
        mEssentialBCs->get(mDirichletDofs, mDirichletValues); // BCs at time = 0
        Kokkos::resize(mPreviousStepDirichletValues, mDirichletValues.size());
        Plato::blas1::fill(0.0, mPreviousStepDirichletValues);
    }

    /***************************************************************************//**
     * \brief Set Dirichlet boundary conditions
     * \param [in] aDirichletDofs   degrees of freedom associated with Dirichlet boundary conditions
     * \param [in] aDirichletValues values associated with Dirichlet degrees of freedom
    *******************************************************************************/
    void setEssentialBoundaryConditions(const Plato::LocalOrdinalVector & aDirichletDofs, const Plato::ScalarVector & aDirichletValues)
    {
        if(aDirichletDofs.size() != aDirichletValues.size())
        {
            std::ostringstream tError;
            tError << "PLASTICITY PROBLEM: DIMENSION MISMATCH: THE NUMBER OF ELEMENTS IN INPUT DOFS AND VALUES ARRAY DO NOT MATCH."
                << "DOFS SIZE = " << aDirichletDofs.size() << " AND VALUES SIZE = " << aDirichletValues.size();
            THROWERR(tError.str())
        }
        mDirichletDofs   = aDirichletDofs;
        mDirichletValues = aDirichletValues;
        Kokkos::resize(mPreviousStepDirichletValues, aDirichletValues.size());
        Plato::blas1::fill(0.0, mPreviousStepDirichletValues);
    }

    /***************************************************************************//**
     * \brief Save states to visualization file
     * \param [in] aFilepath output/viz directory path
    *******************************************************************************/
    void output(const std::string& aFilepath)
    {
        auto tMesh = mSpatialModel.Mesh;
        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aFilepath.c_str(), &tMesh, mSpaceDim);

        auto tNumNodes = mGlobalEquation->numNodes();

        Plato::Scalar tTime = 0.0;
        Plato::OrdinalType tStep = 0;
        for (Plato::OrdinalType tSequenceStepIndex=0; tSequenceStepIndex<mSequence.getSteps().size(); tSequenceStepIndex++)
        {
            auto tGlobalStates  = mSolutions.get("Global States", tSequenceStepIndex);
            auto tTotalStates   = mSolutions.get("Total States", tSequenceStepIndex);
            auto tReactionForce = mSolutions.get("Reaction Forces", tSequenceStepIndex);

            Plato::ScalarMultiVector tPressure("Pressure", tGlobalStates.extent(0), tNumNodes);
            Plato::ScalarMultiVector tTemperature("Temperature", tGlobalStates.extent(0), tNumNodes);
            Plato::ScalarMultiVector tDisplacements("Displacements", tGlobalStates.extent(0), tNumNodes*mSpaceDim);
            Plato::ScalarMultiVector tTotalDisplacements("Total Displacements", tTotalStates.extent(0), tNumNodes*mSpaceDim);

            Plato::blas2::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(tGlobalStates, tPressure);
            Plato::blas2::extract<mNumGlobalDofsPerNode, mSpaceDim>(tNumNodes, tGlobalStates, tDisplacements);
            Plato::blas2::extract<mNumGlobalDofsPerNode, mSpaceDim>(tNumNodes, tTotalStates, tTotalDisplacements);
            Plato::blas2::scale(mPressureScaling, tPressure);

            if (mTemperatureDofOffset > 0)
            {
                Plato::blas2::extract<mNumGlobalDofsPerNode, mTemperatureDofOffset>(tGlobalStates, tTemperature);
                Plato::blas2::scale(mTemperatureScaling, tTemperature);
            }

            for(Plato::OrdinalType tSnapshot = 0; tSnapshot < tDisplacements.extent(0); tSnapshot++)
            {
                auto tPressSubView = Kokkos::subview(tPressure, tSnapshot, Kokkos::ALL());
                auto tPressSubViewDefaultMirror = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), tPressSubView);
                tMesh.add_tag(Omega_h::VERT, "Pressure", 1, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tPressSubViewDefaultMirror)));

                auto tForceSubView = Kokkos::subview(tReactionForce, tSnapshot, Kokkos::ALL());
                auto tForceSubViewDefaultMirror = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), tForceSubView);
                tMesh.add_tag(Omega_h::VERT, "Reaction Force", 1, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tForceSubViewDefaultMirror)));

                auto tDispSubView = Kokkos::subview(tDisplacements, tSnapshot, Kokkos::ALL());
                auto tDispSubViewDefaultMirror = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), tDispSubView);
                tMesh.add_tag(Omega_h::VERT, "Displacements", mSpaceDim, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tDispSubViewDefaultMirror)));

                auto tTotDispSubView = Kokkos::subview(tTotalDisplacements, tSnapshot, Kokkos::ALL());
                auto tTotDispSubViewDefaultMirror = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), tTotDispSubView);
                tMesh.add_tag(Omega_h::VERT, "Total Displacements", mSpaceDim, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tTotDispSubViewDefaultMirror)));

                if (mTemperatureDofOffset > 0)
                {
                    auto tTemperatureSubView = Kokkos::subview(tTemperature, tSnapshot, Kokkos::ALL());
                    auto tTemperatureSubViewDefaultMirror = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(), tTemperatureSubView);
                    tMesh.add_tag(Omega_h::VERT, "Temperature", 1, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tTemperatureSubViewDefaultMirror)));
                }

                Plato::add_state_tags(tMesh, mDataMap, tStep);
                auto tTags = Omega_h::vtk::get_all_vtk_tags(&tMesh, mSpaceDim);
//                auto tTime = mTimeData->mCurrentTimeStepSize * static_cast<Plato::Scalar>(tSnapshot + 1);
                tWriter.write(tStep, tTime, tTags);
                tStep++;
                tTime += 1.0;
            }
        }
    }

    /***************************************************************************//**
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    *******************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector) {return;}

    /***************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControls 1D container of control variables
     * \param [in] aSolution solution database
    *******************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControls,
                       const Plato::Solutions    & aSolution) override
    {
        auto tLocalStates        = mSolutions.get("Local States");
        auto tGlobalStates       = mSolutions.get("Global States");

        mLocalEquation->updateProblem(tGlobalStates, tLocalStates, aControls, *mTimeData);
        mGlobalEquation->updateProblem(tGlobalStates, tLocalStates, aControls, *mTimeData);
        mProjectionEquation->updateProblem(tGlobalStates, aControls, mTimeData->mCurrentTime);

        for( auto tCriterion : mCriteria )
        {
            tCriterion.second->updateProblem(tGlobalStates, tLocalStates, aControls, *mTimeData);
        }
    }

    /***************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControls 1D view of control variables
     * \return solution database
    *******************************************************************************/
    Plato::Solutions solution(const Plato::ScalarVector &aControls) override
    {
        // TODO: NOTES
        // 1. WRITE LOCAL STATES, PRESSURE, AND GLOBAL STATES HISTORY TO FILE - MEMORY CONCERNS
        //   1.1. NO NEED TO STORE MEMBER DATA FOR THESE QUANTITIES
        //   1.2. READ DATA FROM FILES DURING ADJOINT SOLVE
        // 4. HOW WILL OUTPUT DATA BE PRESENTED TO THE USERS, WE CANNOT SEND TIME-DEPENDENT DATA THROUGH THE ENGINE.

        mDataMap.clearStates();

        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: INPUT CONTROL VECTOR IS EMPTY.")
        }

        auto& tSequenceSteps = mSequence.getSteps();
        auto tNumSequenceSteps = tSequenceSteps.size();

        mDataMap.scalarVectors["LoadControlVector"] = Plato::ScalarVector("Load control vector", mSpatialModel.Mesh.nelems());

        for (Plato::OrdinalType tSequenceStepIndex=0; tSequenceStepIndex<tNumSequenceSteps; tSequenceStepIndex++)
        {
            const auto& tSequenceStep = tSequenceSteps[tSequenceStepIndex];
            mSpatialModel.applyMask(tSequenceStep.getMask());

            if(tSequenceStepIndex > 0)
            {
                auto tPrevTotalStates = mSolutions.get("Total States", tSequenceStepIndex-1);
                Plato::OrdinalType tLastStepIndex = tPrevTotalStates.extent(0) - 1;

                auto tTotalStates = mSolutions.get("Total States", tSequenceStepIndex);

                Plato::ScalarVector tTotalState = Kokkos::subview(tTotalStates, 0, Kokkos::ALL());
                Plato::ScalarVector tPrevTotalState = Kokkos::subview(tPrevTotalStates, tLastStepIndex, Kokkos::ALL());
                Kokkos::deep_copy(tTotalState, tPrevTotalState);

                mDataMap.scalarMultiVectors["Previous Strain"] = Plato::blas1::copy(mDataMap.stateDataMaps.back().scalarMultiVectors["Total Strain"]);
            }
            else
            {
                mDataMap.scalarMultiVectors["Previous Strain"] = Plato::ScalarMultiVector("previous strain", mSpatialModel.Mesh.nelems(), PhysicsT::mNumVoigtTerms);
            }

            bool tStop = false;
            while (tStop == false)
            {
                bool tDidSolverConverge = this->solveForwardProblem(aControls, tSequenceStepIndex);
                if (tDidSolverConverge == true)
                {
                    tStop = true;
                    std::stringstream tMsg;
                    tMsg << "\n**** Forward Solve Was Successful ****\n";
                    mNewtonSolver->appendOutputMessage(tMsg);
                    break;
                }
                else
                {
                    if(mTimeData->mMaxNumTimeStepsReached == true)
                    {
                        tStop = true;
                        std::stringstream tMsg;
                        tMsg << "\n**** Maximum Number of Pseudo Time Steps Was Reached. "
                                << "Plasticity Problem failed to converge to a solution. ****\n";
                        REPORT(tMsg.str().c_str());
                        mNewtonSolver->appendOutputMessage(tMsg);
                        mStopOptimization = true;
                        break;
                    }
                    this->resizeTimeDependentStates();
                    std::stringstream tMsg;
                    tMsg << "\n**** Forward Solve Was Not Successful ****\n"
                            << "Number of pseudo time steps will be increased to '" << mTimeData->mNumTimeSteps << "'\n.";
                    REPORT(tMsg.str().c_str());
                    mNewtonSolver->appendOutputMessage(tMsg);
                }
            }
        }

        return mSolutions;
    }

    /***************************************************************************//**
     * \fn Plato::Scalar criterionValue
     * \brief Evaluate criterion function and return its value
     * \param [in] aControls 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return criterion function value
    *******************************************************************************/
    Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControls,
        const Plato::Solutions    & aSolutions,
        const std::string         & aName
    ) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aSolutions.empty())
        {
            THROWERR("PLASTICITY PROBLEM: SOLUTION DATABASE IS EMPTY.");
        }

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];

            this->shouldOptimizationProblemStop();
            auto tOutput = this->evaluateCriterion(*tCriterion, aSolutions, aControls);

            return (tOutput);
        }
        else
        {
            const std::string tErrorMessage = std::string("REQUESTED CRITERION '") + aName + "' NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage);
        }
    }

    /***************************************************************************//**
     * \brief Evaluate criterion function and return its value
     * \param [in] aControls 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return criterion function value
    *******************************************************************************/
    Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControls,
        const std::string         & aName
    ) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];

            this->shouldOptimizationProblemStop();
            auto tOutput = this->evaluateCriterion(*tCriterion, mSolutions, aControls);

            return (tOutput);
        }
        else
        {
            const std::string tErrorMessage = std::string("REQUESTED CRITERION '") + aName + "' NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage);
        }

    }

    /***************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt control variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt control variables
    *******************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControls,
        const std::string         & aName
    ) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];

            this->shouldOptimizationProblemStop();
            auto tTotalDerivative = this->criterionGradient(aControls, mSolutions, tCriterion);

            return tTotalDerivative;
        }
        else
        {
            const std::string tErrorMessage = std::string("REQUESTED CRITERION '") + aName + "' NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage);
        }
    }

    /***************************************************************************//**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view of the criterion gradient wrt control variables
    *******************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControls,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aSolution.empty())
        {
            THROWERR("PLASTICITY PROBLEM: SOLUTION DATABASE IS EMPTY.");
        }

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];

            this->shouldOptimizationProblemStop();
            auto tTotalDerivative = this->criterionGradient(aControls, aSolution, tCriterion);

            return (tTotalDerivative);
        }
        else
        {
            const std::string tErrorMessage = std::string("REQUESTED CRITERION '") + aName + "' NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage);
        }
    }

    /***************************************************************************//**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view of the criterion gradient wrt control variables
    *******************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControls,
        const Plato::Solutions    & aSolution,
              Criterion             aCriterion
    )
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aSolution.empty())
        {
            THROWERR("PLASTICITY PROBLEM: SOLUTION DATABASE IS EMPTY.");
        }

        this->shouldOptimizationProblemStop();
        mAdjointSolver->appendScalarFunction(aCriterion);

        auto tNumNodes = mGlobalEquation->numNodes();
        Plato::ScalarVector tTotalDerivative("Total Derivative", tNumNodes);
        this->backwardTimeIntegration(Plato::PartialDerivative::CONTROL, aControls, tTotalDerivative);
        this->addCriterionPartialDerivativeZ(*aCriterion, aControls, tTotalDerivative);

        return (tTotalDerivative);
    }

    /***************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt configuration variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt configuration variables
    *******************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControls,
        const std::string         & aName
    ) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];
            
            this->shouldOptimizationProblemStop();
            auto tTotalDerivative = this->criterionGradientX(aControls, mSolutions, tCriterion);

            return tTotalDerivative;
        }
        else
        {
            const std::string tErrorMessage = std::string("REQUESTED CRITERION '") + aName + "' NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage);
        }

    }

    /***************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view of the criterion gradient wrt control variables
    *******************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControls,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aSolution.empty())
        {
            THROWERR("PLASTICITY PROBLEM: SOLUTIONS DATABASE IS EMPTY.");
        }

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];
            auto tTotalDerivative = this->criterionGradientX(aControls, aSolution, tCriterion);

            return (tTotalDerivative);
        }
        else
        {
            const std::string tErrorMessage = std::string("REQUESTED CRITERION '") + aName + "' NOT DEFINED BY THE USER.";
            THROWERR(tErrorMessage);
        }
    }

    /***************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aSolution solution database
     * \param [in] aName Name of criterion.
     * \return 1D view of the criterion gradient wrt control variables
    *******************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControls,
        const Plato::Solutions    & aSolution,
              Criterion             aCriterion
    )
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aSolution.empty())
        {
            THROWERR("PLASTICITY PROBLEM: SOLUTION DATABASE IS EMPTY.");
        }


        this->shouldOptimizationProblemStop();
        mAdjointSolver->appendScalarFunction(aCriterion);
        Plato::ScalarVector tTotalDerivative("Total Derivative", mNumConfigDofsPerCell);
        this->backwardTimeIntegration(Plato::PartialDerivative::CONFIGURATION, aControls, tTotalDerivative);
        this->addCriterionPartialDerivativeX(*aCriterion, aControls, tTotalDerivative);

        return (tTotalDerivative);
    }

    /***************************************************************************//**
     * \brief Compute reaction force.
     * \param [in] aControl 1D view of controls
     * \param [in] aStates  C++ structure with current state information
    *******************************************************************************/
    void computeReactionForce(const Plato::ScalarVector &aControl, Plato::CurrentStates &aStates)
    {
        mDataMap.mScalarValues["LoadControlConstant"] = 0.0;
        auto tInternalForce = mGlobalEquation->value(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                     aStates.mCurrentLocalState,  aStates.mPreviousLocalState,
                                                     aStates.mProjectedPressGrad, aControl, *(aStates.mTimeData));

        auto tNumNodes = mGlobalEquation->numNodes();
        auto tReactionForces = mSolutions.get("Reaction Forces");
        auto tReactionForce = Kokkos::subview(tReactionForces, aStates.mCurrentStepIndex, Kokkos::ALL());
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType &aOrdinal)
        {
            for(Plato::OrdinalType tDim = 0; tDim < mSpaceDim; tDim++)
            {
                tReactionForce(aOrdinal) += tInternalForce(aOrdinal*mNumGlobalDofsPerNode+tDim);
            }
        }, "reaction force");
    }

    /***************************************************************************//**
     * \brief Set initial temperature field equal to the reference temperature
     * \param [in]     aCurrentStepIndex    current time step index
     * \param [in/out] aPreviousGlobalState previous global state
    *******************************************************************************/
    void setInitialTemperature(const Plato::OrdinalType & aCurrentStepIndex,
                               const Plato::ScalarVector & aPreviousGlobalState) const
    {
        if (aCurrentStepIndex != static_cast<Plato::OrdinalType>(0)) return;

        auto tReferenceTemperature = mReferenceTemperature;
        auto tTemperatureScaling   = mTemperatureScaling;
        auto tNumGlobalDofsPerNode = mNumGlobalDofsPerNode;
        auto tTemperatureDofOffset = mTemperatureDofOffset;
        auto tNumVerts             = mSpatialModel.Mesh.nverts();

        if (tTemperatureDofOffset > static_cast<Plato::OrdinalType>(0))
        {
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumVerts),
                                    LAMBDA_EXPRESSION(const Plato::OrdinalType &aVertexOrdinal)
            {
                Plato::OrdinalType tIndex = aVertexOrdinal * tNumGlobalDofsPerNode + tTemperatureDofOffset;
                aPreviousGlobalState(tIndex) = tReferenceTemperature / tTemperatureScaling;
            }, "set temperature to reference");
        }
    }

// private functions
private:
    /***************************************************************************//**
     * \brief Initialize member data
     * \param [in] aInputParams  input parameter list
    *******************************************************************************/
    void initialize(Teuchos::ParameterList& aInputParams)
    {
        this->allocateCriteria(aInputParams);

        this->allocateSolutions();

        if(aInputParams.isSublist("Material Models") == false)
        {
            THROWERR("Plasticity Problem: 'Material Models' Parameter Sublist is not defined.")
        }
        Teuchos::ParameterList tMaterialsInputs = aInputParams.get<Teuchos::ParameterList>("Material Models");

        mPressureScaling    = tMaterialsInputs.get<Plato::Scalar>("Pressure Scaling", 1.0);
        mTemperatureScaling = tMaterialsInputs.get<Plato::Scalar>("Temperature Scaling", 1.0);

        // Get the reference temperature and make sure all materials have an identical reference temperature
        if (mTemperatureDofOffset > 0)
        {
            std::vector<Plato::Scalar> tReferenceTemperatures;
            for(Teuchos::ParameterList::ConstIterator tIndex =  tMaterialsInputs.begin(); 
                                                      tIndex != tMaterialsInputs.end(); ++tIndex)
            {
                const Teuchos::ParameterEntry & tEntry = tMaterialsInputs.entry(tIndex);
                if (!tEntry.isList()) continue;

                std::string tName = tMaterialsInputs.name(tIndex);
                Teuchos::ParameterList tMaterialList = tMaterialsInputs.sublist(tName);

                if (!tMaterialList.isSublist("Isotropic Linear Thermoelastic")) continue;
                auto tThermoelasticSubList = tMaterialList.sublist("Isotropic Linear Thermoelastic");
                auto tReferenceTemperature = tThermoelasticSubList.get<Plato::Scalar>("Reference Temperature");
                tReferenceTemperatures.push_back(tReferenceTemperature);
            }
            if (tReferenceTemperatures.empty())
            {
                THROWERR("Reference temperature should be set for at least one material.")
            }
            else if (tReferenceTemperatures.size() == 1)
            {
                mReferenceTemperature = tReferenceTemperatures[0];
            }
            else
            {
                mReferenceTemperature = tReferenceTemperatures[0];
                for (Plato::OrdinalType tIndex = 0; tIndex < tReferenceTemperatures.size(); ++tIndex)
                    if (std::abs(mReferenceTemperature - tReferenceTemperatures[tIndex]) > 5.0e-7)
                        THROWERR("All materials must have the same reference temperature.")
            }
        }

    }

    /***************************************************************************//**
     * \brief Checks if optimization problem should be stop due to Newton-Raphson
     * solver's failure to converge during optimization.
    *******************************************************************************/
    void shouldOptimizationProblemStop()
    {
        if(mStopOptimization == true)
        {
            std::stringstream tMsg;
            tMsg << "\n**** Plasticity Problem: Newton-Raphson solver failed to converge during optimization. "
                    << "Optimization results are going to be impacted by the solver's failure to converge. ****\n";
            THROWERR(tMsg.str().c_str())
        }
    }

    /***************************************************************************//**
     * \brief Resize time-dependent state containers and decrease pseudo-time step size.
    *******************************************************************************/
    void resizeTimeDependentStates()
    {
        mTimeData->increaseNumTimeSteps();

        Plato::blas1::fill(0.0, mPreviousStepDirichletValues);

        auto tLocalStates        = mSolutions.get("Local States");
        auto tGlobalStates       = mSolutions.get("Global States");
        auto tReactionForces     = mSolutions.get("Reaction Forces");
        auto tProjectedPressGrad = mSolutions.get("Projected Pressure Gradient");

        Kokkos::resize(tLocalStates,        mTimeData->mNumTimeSteps, mLocalEquation->size());
        Kokkos::resize(tGlobalStates,       mTimeData->mNumTimeSteps, mGlobalEquation->size());
        Kokkos::resize(tReactionForces,     mTimeData->mNumTimeSteps, mGlobalEquation->numNodes());
        Kokkos::resize(tProjectedPressGrad, mTimeData->mNumTimeSteps, mProjectionEquation->size());
    }

    /***************************************************************************//**
     * \brief Initialize Newton-Raphson solver
    *******************************************************************************/
    void initializeNewtonSolver()
    {
        mNewtonSolver->appendDirichletDofs(mDirichletDofs);
        mNewtonSolver->appendDirichletValues(mDirichletValues);

        mNewtonSolver->appendLocalEquation(mLocalEquation);
        mNewtonSolver->appendGlobalEquation(mGlobalEquation);

        std::stringstream tMsg("\n\n**** NEW NEWTON-RAPHSON SOLVE ****\n\n");
        mNewtonSolver->appendOutputMessage(tMsg);
    }

    /***************************************************************************//**
     * \brief Solve forward problem
     * \param [in] aControls 1-D view of controls, e.g. design variables
     * \return flag used to indicate forward problem was solved to completion
    *******************************************************************************/
    bool solveForwardProblem(const Plato::ScalarVector & aControls, Plato::OrdinalType aSequenceStepIndex)
    {
        auto tTotalStates = mSolutions.get("Total States", aSequenceStepIndex);
        auto tGlobalStates = mSolutions.get("Global States", aSequenceStepIndex);

        Plato::CurrentStates tCurrentState(mTimeData);
        auto tNumCells = mLocalEquation->numCells();
        tCurrentState.mDeltaGlobalState = Plato::ScalarVector("Global State Increment", mGlobalEquation->size());

        this->initializeNewtonSolver();

        auto& tSequenceStep = mSequence.getSteps()[aSequenceStepIndex];

        bool tForwardProblemSolved = false;
        mTimeData = mSequenceTimeData[aSequenceStepIndex];
        for(Plato::OrdinalType tCurrentStepIndex = 0; tCurrentStepIndex < mTimeData->mNumTimeSteps; tCurrentStepIndex++)
        {
            mTimeData->updateTimeData(tCurrentStepIndex);

            std::stringstream tMsg;
            tMsg << "TIME STEP #" << mTimeData->getTimeStepIndexPlusOne() << " OUT OF " << mTimeData->mNumTimeSteps
                 << " TIME STEPS, TOTAL TIME = " << mTimeData->mCurrentTime << "\n";
            mNewtonSolver->appendOutputMessage(tMsg);

            tCurrentState.mCurrentStepIndex = tCurrentStepIndex;
            this->cacheStateData(tCurrentState, aSequenceStepIndex);

            // update displacement and load control multiplier
            this->updateDispAndLoadControlMultipliers(tCurrentStepIndex);

            // update local and global states
            bool tNewtonRaphsonConverged = mNewtonSolver->solve(aControls, tCurrentState, tSequenceStep);

            mDataMap.scalarNodeFields["topology"] = aControls;
            Plato::toMap(mDataMap.scalarNodeFields, tSequenceStep.getMask()->nodeMask(), "node_mask");

            Plato::ScalarVector tTotalState = Kokkos::subview(tTotalStates, tCurrentStepIndex, Kokkos::ALL());
            Plato::ScalarVector tGlobalState = Kokkos::subview(tGlobalStates, tCurrentStepIndex, Kokkos::ALL());
            if(tCurrentStepIndex > 0)
            {
                Plato::ScalarVector tPrevTotalState = Kokkos::subview(tTotalStates, tCurrentStepIndex-1, Kokkos::ALL());
                Kokkos::deep_copy(tTotalState, tPrevTotalState);
            }
            Plato::blas1::axpy(1.0, tGlobalState, tTotalState);

            Plato::ScalarMultiVector tStrainIncrement = mDataMap.scalarMultiVectors["Strain Increment"];
            Plato::ScalarMultiVector tPreviousStrain  = mDataMap.scalarMultiVectors["Previous Strain"];

            Plato::ScalarMultiVector tTotalStrain = Plato::ScalarMultiVector("total strain", mSpatialModel.Mesh.nelems(), PhysicsT::mNumVoigtTerms);
            mDataMap.scalarMultiVectors["Total Strain"] = tTotalStrain;
            Kokkos::deep_copy(tTotalStrain, tPreviousStrain);

            Plato::blas1::axpy2d(1.0, tStrainIncrement, tTotalStrain);

            // compute reaction force
            this->computeReactionForce(aControls, tCurrentState);

            mDataMap.saveState();
            mDataMap.scalarVectors["LoadControlVector"] = Plato::blas1::copy(mDataMap.stateDataMaps.back().scalarVectors["LoadControlVector"]);
            mDataMap.scalarMultiVectors["Previous Strain"] = Plato::blas1::copy(mDataMap.stateDataMaps.back().scalarMultiVectors["Total Strain"]);

            if(tNewtonRaphsonConverged == false)
            {
                std::stringstream tMsg;
                tMsg << "**** Newton-Raphson Solver did not converge at time step #"
                     << mTimeData->getTimeStepIndexPlusOne()
                     << ".  Number of pseudo time steps will be increased to '"
                     << static_cast<Plato::OrdinalType>(mTimeData->mNumTimeSteps * mTimeData->mTimeStepExpansionMultiplier) << "'. ****\n\n";
                mNewtonSolver->appendOutputMessage(tMsg);
                return tForwardProblemSolved;
            }

            // update projected pressure gradient state
            this->updateProjectedPressureGradient(aControls, tCurrentState);
        }
        
        tForwardProblemSolved = true;
        // mControl = aControls;
        // this->saveStates("PlasticityOutput");
        return tForwardProblemSolved;
    }

public:
    /***************************************************************************//**
     * \brief Update displacement and load control multiplier.
     * \param [in] aInput  current time step index
    *******************************************************************************/
    void updateDispAndLoadControlMultipliers(const Plato::OrdinalType& aInput)
    {
        auto tLoadControlConstant = mTimeData->mCurrentTimeStepSize * static_cast<Plato::Scalar>(aInput + 1);
        mDataMap.mScalarValues["LoadControlConstant"] = tLoadControlConstant;

        auto tLoadControlVector = mDataMap.scalarVectors["LoadControlVector"];

        // loop over blocks
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tCellOrds = tDomain.cellOrdinals();
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aOrdinal)
            {
                auto tGlobalCellOrd = tCellOrds(aOrdinal);
                tLoadControlVector(tGlobalCellOrd) = (tLoadControlVector(tGlobalCellOrd) == 1.0) ? 1.0 : tLoadControlConstant;
            }, "update load control vector");
        }

        if (aInput != static_cast<Plato::OrdinalType>(0))
          Plato::blas1::copy(mDirichletValues, mPreviousStepDirichletValues);
        else
          Plato::blas1::fill(0.0, mPreviousStepDirichletValues);

        auto tCurrentTime = tLoadControlConstant;
        if (mEssentialBCs != nullptr)
          mEssentialBCs->get(mDirichletDofs, mDirichletValues, tCurrentTime);
        else
          THROWERR("EssentialBCs pointer is null!")

        Plato::ScalarVector tNewtonUpdateDirichletValues("Dirichlet Increment Values", mDirichletValues.size());
        Plato::blas1::copy(mDirichletValues, tNewtonUpdateDirichletValues);
        Plato::blas1::axpy(static_cast<Plato::Scalar>(-1.0), mPreviousStepDirichletValues, tNewtonUpdateDirichletValues);

        mNewtonSolver->appendDirichletValues(tNewtonUpdateDirichletValues);
    }
private:

    /***************************************************************************//**
     * \brief Update projected pressure gradient.
     * \param [in]     aControls  1-D view of controls, e.g. design variables
     * \param [in/out] aStateData data manager with current and previous global and local state data
    *******************************************************************************/
    void updateProjectedPressureGradient(const Plato::ScalarVector &aControls,
                                         Plato::CurrentStates &aStateData)
    {
        Plato::OrdinalType tNextStepIndex = aStateData.mCurrentStepIndex + static_cast<Plato::OrdinalType>(1);
        if(tNextStepIndex >= mTimeData->mNumTimeSteps)
        {
            return;
        }

        // copy projection state, i.e. pressure
        Plato::blas1::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(aStateData.mCurrentGlobalState, mPressure);

        // compute projected pressure gradient
        auto tProjectedPressGrad = mSolutions.get("Projected Pressure Gradient");
        auto tNextProjectedPressureGradient = Kokkos::subview(tProjectedPressGrad, tNextStepIndex, Kokkos::ALL());
        Plato::blas1::fill(0.0, tNextProjectedPressureGradient);
        auto tProjResidual = mProjectionEquation->value(tNextProjectedPressureGradient, mPressure, aControls, tNextStepIndex);
        auto tProjJacobian = mProjectionEquation->gradient_u(tNextProjectedPressureGradient, mPressure, aControls, tNextStepIndex);
        Plato::blas1::scale(-1.0, tProjResidual);
        Plato::Solve::RowSummed<PhysicsT::mNumSpatialDims>(tProjJacobian, tNextProjectedPressureGradient, tProjResidual);
    }

    /***************************************************************************//**
     * \brief Get previous state
     * \param [in]     aCurrentStepIndex current time step index
     * \param [in]     aStates           states at each time step
     * \param [in/out] aOutput           previous state
    *******************************************************************************/
    void
    getPreviousState(
        const Plato::OrdinalType       & aCurrentStepIndex,
        const Plato::ScalarMultiVector & aStates,
              Plato::ScalarVector      & aOutput
    ) const
    {
        auto tPreviousStepIndex = aCurrentStepIndex - static_cast<Plato::OrdinalType>(1);
        if(tPreviousStepIndex >= static_cast<Plato::OrdinalType>(0))
        {
            aOutput = Kokkos::subview(aStates, tPreviousStepIndex, Kokkos::ALL());
        }
        else
        {
            auto tLength = aStates.extent(1);
            aOutput = Plato::ScalarVector("State t=i-1", tLength);
            Plato::blas1::fill(0.0, aOutput);
        }
    }
    

    /***************************************************************************//**
     * \brief Evaluate criterion
     * \param [in] aCriterion   criterion scalar function interface
     * \param [in] aSolution solution database
     * \param [in] aControls    current controls, e.g. design variables
     * \return new criterion value
    *******************************************************************************/
    Plato::Scalar
    evaluateCriterion(
              Plato::LocalScalarFunctionInc & aCriterion,
        const Plato::Solutions              & aSolutions,
        const Plato::ScalarVector           & aControls
    )
    {

        auto tGlobalStates = mSolutions.get("Global States");
        auto tLocalStates  = mSolutions.get("Local States");

        const auto& tSequenceSteps = mSequence.getSteps();

        Plato::Scalar tOutput = 0;
        for (Plato::OrdinalType tSequenceStepIndex=0; tSequenceStepIndex<tSequenceSteps.size(); tSequenceStepIndex++)
        {
            const auto& tSequenceStep = tSequenceSteps[tSequenceStepIndex];

            mSpatialModel.applyMask(tSequenceStep.getMask());

            Plato::ScalarVector tPreviousLocalState;
            Plato::ScalarVector tPreviousGlobalState;

            for(Plato::OrdinalType tCurrentStepIndex = 0; tCurrentStepIndex < mTimeData->mNumTimeSteps; tCurrentStepIndex++)
            {
                // SET CURRENT STATES
                auto tCurrentLocalState = Kokkos::subview(tLocalStates, tCurrentStepIndex, Kokkos::ALL());
                auto tCurrentGlobalState = Kokkos::subview(tGlobalStates, tCurrentStepIndex, Kokkos::ALL());

                // SET PREVIOUS AND FUTURE STATES
                this->getPreviousState(tCurrentStepIndex, tLocalStates, tPreviousLocalState);
                this->getPreviousState(tCurrentStepIndex, tGlobalStates, tPreviousGlobalState);
                this->setInitialTemperature(tCurrentStepIndex, tPreviousGlobalState);

                mTimeData->updateTimeData(tCurrentStepIndex);
                tOutput += aCriterion.value(tCurrentGlobalState, tPreviousGlobalState,
                                            tCurrentLocalState, tPreviousLocalState,
                                            aControls, *mTimeData);
            }
        }
        return tOutput;
    }

    /***************************************************************************//**
     * \brief Add contribution from partial derivative of criterion with respect to
     * controls to total derivative of criterion with respect to controls.
     * \param [in]     aCriterion     design criterion interface
     * \param [in]     aControls      current controls, e.g. design variables
     * \param [in/out] aTotalGradient total derivative of criterion with respect to controls
    *******************************************************************************/
    void
    addCriterionPartialDerivativeZ(
              Plato::LocalScalarFunctionInc & aCriterion,
        const Plato::ScalarVector           & aControls,
              Plato::ScalarVector           & aTotalGradient
    )
    {
        auto tGlobalStates = mSolutions.get("Global States");
        auto tLocalStates  = mSolutions.get("Local States");

        Plato::ScalarVector tPreviousLocalState;
        Plato::ScalarVector tPreviousGlobalState;
        for(Plato::OrdinalType tCurrentStepIndex = 0; tCurrentStepIndex < mTimeData->mNumTimeSteps; tCurrentStepIndex++)
        {
            auto tCurrentLocalState = Kokkos::subview(tLocalStates, tCurrentStepIndex, Kokkos::ALL());
            auto tCurrentGlobalState = Kokkos::subview(tGlobalStates, tCurrentStepIndex, Kokkos::ALL());

            // SET PREVIOUS STATES
            this->getPreviousState(tCurrentStepIndex, tLocalStates, tPreviousLocalState);
            this->getPreviousState(tCurrentStepIndex, tGlobalStates, tPreviousGlobalState);
            this->setInitialTemperature(tCurrentStepIndex, tPreviousGlobalState);

            mTimeData->updateTimeData(tCurrentStepIndex);
            auto tDfDz = aCriterion.gradient_z(tCurrentGlobalState, tPreviousGlobalState,
                                               tCurrentLocalState, tPreviousLocalState,
                                               aControls, *mTimeData);
            mWorksetBase.assembleScalarGradientZ(tDfDz, aTotalGradient);
        }
    }

    /***************************************************************************//**
     * \brief Add contribution from partial derivative of criterion with respect to
     * configuration to total derivative of criterion with respect to configuration.
     * \param [in]     aCriterion     design criterion interface
     * \param [in]     aControls      current controls, e.g. design variables
     * \param [in/out] aTotalGradient total derivative of criterion with respect to configuration
    *******************************************************************************/
    void
    addCriterionPartialDerivativeX(
              Plato::LocalScalarFunctionInc & aCriterion,
        const Plato::ScalarVector           & aControls,
              Plato::ScalarVector           & aTotalGradient
    )
    {
        auto tGlobalStates = mSolutions.get("Global States");
        auto tLocalStates  = mSolutions.get("Local States");

        Plato::ScalarVector tPreviousLocalState;
        Plato::ScalarVector tPreviousGlobalState;
        for(Plato::OrdinalType tCurrentStepIndex = 0; tCurrentStepIndex < mTimeData->mNumTimeSteps; tCurrentStepIndex++)
        {
            auto tCurrentLocalState = Kokkos::subview(tLocalStates, tCurrentStepIndex, Kokkos::ALL());
            auto tCurrentGlobalState = Kokkos::subview(tGlobalStates, tCurrentStepIndex, Kokkos::ALL());

            // SET PREVIOUS AND FUTURE LOCAL STATES
            this->getPreviousState(tCurrentStepIndex, tLocalStates, tPreviousLocalState);
            this->getPreviousState(tCurrentStepIndex, tGlobalStates, tPreviousGlobalState);
            this->setInitialTemperature(tCurrentStepIndex, tPreviousGlobalState);

            mTimeData->updateTimeData(tCurrentStepIndex);
            auto tDfDX = aCriterion.gradient_x(tCurrentGlobalState, tPreviousGlobalState,
                                               tCurrentLocalState, tPreviousLocalState,
                                               aControls, *mTimeData);
            mWorksetBase.assembleVectorGradientX(tDfDX, aTotalGradient);
        }
    }

    /***************************************************************************//**
     * \brief Initialize adjoint solver, i.e. append necessary system of equation interfaces.
    *******************************************************************************/
    void initializeAdjointSolver()
    {
        mAdjointSolver->appendDirichletDofs(mDirichletDofs);
        mAdjointSolver->appendLocalEquation(mLocalEquation);
        mAdjointSolver->appendGlobalEquation(mGlobalEquation);
        mAdjointSolver->appendProjectionEquation(mProjectionEquation);
        mAdjointSolver->setNumPseudoTimeSteps(mTimeData->mNumTimeSteps);
    }

    /***************************************************************************//**
     * \brief Perform backward time integration and add Partial Differential Equation
     * (PDE) contribution to total gradient.
     * \param [in]     aType      partial derivative type
     * \param [in]     aControls current controls, e.g. design variables
     * \param [in/out] aOutput   total derivative of criterion with respect to controls
    *******************************************************************************/
    void backwardTimeIntegration(const Plato::PartialDerivative::derivative_t & aType,
                                 const Plato::ScalarVector & aControls,
                                 Plato::ScalarVector aTotalDerivative)
    {
        // Create state data manager
        auto tNumCells = mLocalEquation->numCells();
        Plato::ForwardStates tCurrentStates(aType, *mTimeData);
        Plato::ForwardStates tPreviousStates(aType, *mTimeData);
        Plato::AdjointStates tAdjointStates(mGlobalEquation->size(), mLocalEquation->size(), mProjectionEquation->size());
        tAdjointStates.mInvLocalJacT = ScalarArray3D("Inv(DhDc)^T", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);

        this->initializeAdjointSolver();

        // outer loop for pseudo time steps
        auto tLastStepIndex = mTimeData->mNumTimeSteps - static_cast<Plato::OrdinalType>(1);
        for(tCurrentStates.mCurrentStepIndex = tLastStepIndex; tCurrentStates.mCurrentStepIndex >= 0; tCurrentStates.mCurrentStepIndex--)
        {
            tCurrentStates.mTimeData.updateTimeData(tCurrentStates.mCurrentStepIndex);

            tPreviousStates.mCurrentStepIndex = tCurrentStates.mCurrentStepIndex + 1;
            tPreviousStates.mTimeData.updateTimeData(tPreviousStates.mCurrentStepIndex);
            if(tPreviousStates.mCurrentStepIndex < mTimeData->mNumTimeSteps)
            {
                this->updateForwardState(tPreviousStates);
            }
            this->updateForwardState(tCurrentStates);
            this->updateAdjointState(tAdjointStates);

            mAdjointSolver->updateAdjointVariables(aControls, tCurrentStates, tPreviousStates, tAdjointStates);
            mAdjointSolver->addContributionFromPDE(aControls, tCurrentStates, tAdjointStates, aTotalDerivative);
        }
    }

    /***************************************************************************//**
     * \brief Update state data for time step n, i.e. current time step:
     * \param [in] aStateData state data manager
     * \param [in] aZeroEntries flag - zero all entries in current states (default = false)
    *******************************************************************************/
    void cacheStateData(Plato::CurrentStates & aStateData, Plato::OrdinalType aSequenceStepIndex=0)
    {
        auto tLocalStates        = mSolutions.get("Local States", aSequenceStepIndex);
        auto tGlobalStates       = mSolutions.get("Global States", aSequenceStepIndex);
        auto tProjectedPressGrad = mSolutions.get("Projected Pressure Gradient", aSequenceStepIndex);

        // GET CURRENT STATE
        aStateData.mCurrentLocalState = Kokkos::subview(tLocalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mCurrentGlobalState = Kokkos::subview(tGlobalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mProjectedPressGrad = Kokkos::subview(tProjectedPressGrad, aStateData.mCurrentStepIndex, Kokkos::ALL());

        // GET PREVIOUS STATE
        if(aSequenceStepIndex > 0 && aStateData.mCurrentStepIndex == 0) 
        {
            auto tPrevLocalStates  = mSolutions.get("Local States", aSequenceStepIndex-1);
            auto tPrevGlobalStates = mSolutions.get("Global States", aSequenceStepIndex-1);
            Plato::OrdinalType tLastStepIndex = tPrevLocalStates.extent(0)-1;
            aStateData.mPreviousLocalState = Kokkos::subview(tPrevLocalStates, tLastStepIndex, Kokkos::ALL());
            aStateData.mPreviousGlobalState = Kokkos::subview(tPrevGlobalStates, tLastStepIndex, Kokkos::ALL());
        }
        else
        {
            this->getPreviousState(aStateData.mCurrentStepIndex, tLocalStates, aStateData.mPreviousLocalState);
            this->getPreviousState(aStateData.mCurrentStepIndex, tGlobalStates, aStateData.mPreviousGlobalState);
        }

        // SET ENTRIES IN CURRENT STATES TO ZERO
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), aStateData.mCurrentLocalState);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), aStateData.mCurrentGlobalState);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mPressure);
    }

    /***************************************************************************//**
     * \brief Update state data for time step n, i.e. current time step:
     * \param [in] aStateData state data manager
     * \param [in] aZeroEntries flag - zero all entries in current states (default = false)
    *******************************************************************************/
    void updateForwardState(Plato::ForwardStates & aStateData)
    {
        auto tLocalStates        = mSolutions.get("Local States");
        auto tGlobalStates       = mSolutions.get("Global States");
        auto tReactionForces     = mSolutions.get("Reaction Forces");
        auto tProjectedPressGrad = mSolutions.get("Projected Pressure Gradient");

        // GET CURRENT STATE
        aStateData.mCurrentLocalState = Kokkos::subview(tLocalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mCurrentGlobalState = Kokkos::subview(tGlobalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mProjectedPressGrad = Kokkos::subview(tProjectedPressGrad, aStateData.mCurrentStepIndex, Kokkos::ALL());
        if(aStateData.mPressure.size() <= static_cast<Plato::OrdinalType>(0))
        {
            auto tNumVerts = mSpatialModel.Mesh.nverts();
            aStateData.mPressure = Plato::ScalarVector("Current Pressure Field", tNumVerts);
        }
        Plato::blas1::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(aStateData.mCurrentGlobalState, aStateData.mPressure);

        // GET PREVIOUS STATE.
        this->getPreviousState(aStateData.mCurrentStepIndex, tLocalStates, aStateData.mPreviousLocalState);
        this->getPreviousState(aStateData.mCurrentStepIndex, tGlobalStates, aStateData.mPreviousGlobalState);
        this->setInitialTemperature(aStateData.mCurrentStepIndex, aStateData.mPreviousGlobalState);
    }

    /***************************************************************************//**
     * \brief Update adjoint data for time step n, i.e. current time step:
     * \param [in] aAdjointData adjoint data manager
    *******************************************************************************/
    void updateAdjointState(Plato::AdjointStates& aAdjointStates)
    {
        // NOTE: CURRENT ADJOINT VARIABLES ARE UPDATED AT SOLVE TIME. THERE IS NO NEED TO SET THEM TO ZERO HERE.
        const Plato::Scalar tAlpha = 1.0; const Plato::Scalar tBeta = 0.0;
        Plato::blas1::update(tAlpha, aAdjointStates.mCurrentLocalAdjoint, tBeta, aAdjointStates.mPreviousLocalAdjoint);
        Plato::blas1::update(tAlpha, aAdjointStates.mCurrentGlobalAdjoint, tBeta, aAdjointStates.mPreviousGlobalAdjoint);
        Plato::blas1::update(tAlpha, aAdjointStates.mProjPressGradAdjoint, tBeta, aAdjointStates.mPreviousProjPressGradAdjoint);
    }

    /***************************************************************************//**
     * \brief Allocate objective function interface and adjoint containers
     * \param [in] aInputParams input parameters database
    *******************************************************************************/
    void allocateCriteria(Teuchos::ParameterList& aInputParams)
    {
        if(aInputParams.isSublist("Criteria"))
        {
            Plato::PathDependentScalarFunctionFactory<PhysicsT> tObjectiveFunctionFactory;

            auto tCriteriaParams = aInputParams.sublist("Criteria");
            for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); tIndex != tCriteriaParams.end(); ++tIndex)
            {
                const Teuchos::ParameterEntry & tEntry = tCriteriaParams.entry(tIndex);
                std::string tName = tCriteriaParams.name(tIndex);

                TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error,
                  " Parameter in Criteria block not valid.  Expect lists only.");

                if( tCriteriaParams.sublist(tName).get<bool>("Linear", false) == false )
                {
                    auto tCriterion = tObjectiveFunctionFactory.create(mSpatialModel, mDataMap, aInputParams, tName);
                    if( tCriterion != nullptr )
                    {
                        mCriteria[tName] = tCriterion;
                    }
                }
            }
        }
        if(mCriteria.size() == 0)
        {
            REPORT("Plasticity Problem: No criteria defined for this problem.")
        }
    }
    /***************************************************************************//**
     * \brief Allocate storage for solutions
    *******************************************************************************/
    void allocateSolutions()
    {
        long unsigned int tNumTimeSteps  = mTimeData->mNumTimeSteps;
        long unsigned int tNumLocalDofs  = mLocalEquation->size();
        long unsigned int tNumGlobalDofs = mGlobalEquation->size();
        long unsigned int tNumProjDofs   = mProjectionEquation->size();
        long unsigned int tNumVerts      = mSpatialModel.Mesh.nverts();

        for (Plato::OrdinalType tSequenceStepIndex=0; tSequenceStepIndex<mSequence.getSteps().size(); tSequenceStepIndex++)
        {
            mSolutions.set("Local States",   Plato::ScalarMultiVector("Local States",   tNumTimeSteps, tNumLocalDofs),  tSequenceStepIndex);
            mSolutions.set("Global States",  Plato::ScalarMultiVector("Global States",  tNumTimeSteps, tNumGlobalDofs), tSequenceStepIndex);
            mSolutions.set("Total States",   Plato::ScalarMultiVector("Total States",   tNumTimeSteps, tNumGlobalDofs), tSequenceStepIndex);

            mSolutions.set("Reaction Forces", Plato::ScalarMultiVector("Reaction Forces", tNumTimeSteps, tNumVerts), tSequenceStepIndex),

            mSolutions.set("Projected Pressure Gradient", Plato::ScalarMultiVector("Projected Pressure Gradient", tNumTimeSteps, tNumProjDofs), tSequenceStepIndex);
        }
    }
};
// class PlasticityProblem

}
// namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<1>>;
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainThermoPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<2>>;
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainThermoPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<3>>;
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainThermoPlasticity<3>>;
#endif


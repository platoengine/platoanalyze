/*
 * PlasticityProblem.hpp
 *
 *  Created on: Mar 2, 2020
 */

#pragma once

#include <memory>

#include "EssentialBCs.hpp"
#include "OmegaHUtilities.hpp"
#include "NewtonRaphsonSolver.hpp"
#include "PlatoAbstractProblem.hpp"
#include "PathDependentAdjointSolver.hpp"
#include "InfinitesimalStrainPlasticity.hpp"
#include "PathDependentScalarFunctionFactory.hpp"

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
    static constexpr auto mSpaceDim = PhysicsT::mSpaceDim;                           /*!< spatial dimensions*/
    static constexpr auto mNumNodesPerCell = PhysicsT::mNumNodesPerCell;              /*!< number of nodes per cell*/
    static constexpr auto mPressureDofOffset = PhysicsT::mPressureDofOffset;          /*!< number of pressure dofs offset*/
    static constexpr auto mNumGlobalDofsPerNode = PhysicsT::mNumDofsPerNode;          /*!< number of global degrees of freedom per node*/
    static constexpr auto mNumGlobalDofsPerCell = PhysicsT::mNumDofsPerCell;          /*!< number of global degrees of freedom per cell (i.e. element)*/
    static constexpr auto mNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;      /*!< number of local degrees of freedom per cell (i.e. element)*/
    static constexpr auto mNumPressGradDofsPerCell = PhysicsT::mNumNodeStatePerCell;  /*!< number of projected pressure gradient degrees of freedom per cell (i.e. element)*/
    static constexpr auto mNumPressGradDofsPerNode = PhysicsT::mNumNodeStatePerNode;  /*!< number of projected pressure gradient degrees of freedom per node*/
    static constexpr auto mNumConfigDofsPerCell = mSpaceDim * mNumNodesPerCell; /*!< number of configuration (i.e. coordinates) degrees of freedom per cell (i.e. element) */

    // Required
    using PlasticityT = typename Plato::Plasticity<mSpaceDim>;
    using ProjectorT  = typename Plato::Projection<mSpaceDim, PhysicsT::mNumDofsPerNode, PhysicsT::mPressureDofOffset>;
    std::shared_ptr<Plato::VectorFunctionVMS<ProjectorT>> mProjectionEquation;  /*!< global pressure gradient projection interface */
    std::shared_ptr<Plato::GlobalVectorFunctionInc<PhysicsT>> mGlobalEquation;  /*!< global equality constraint interface */
    std::shared_ptr<Plato::LocalVectorFunctionInc<PlasticityT>> mLocalEquation; /*!< local equality constraint interface */

    // Optional
    std::shared_ptr<Plato::LocalScalarFunctionInc> mObjective;  /*!< objective constraint interface */
    std::shared_ptr<Plato::LocalScalarFunctionInc> mConstraint; /*!< constraint constraint interface */

    Plato::OrdinalType mNumPseudoTimeSteps;       /*!< current number of pseudo time steps */
    Plato::OrdinalType mMaxNumPseudoTimeSteps;    /*!< maximum number of pseudo time steps */

    Plato::Scalar mPseudoTimeStep;                /*!< pseudo time step */
    Plato::Scalar mInitialNormResidual;           /*!< initial norm of global residual */
    Plato::Scalar mDispControlConstant;           /*!< displacement control constant */
    Plato::Scalar mCurrentPseudoTimeStep;         /*!< current pseudo time step */
    Plato::Scalar mPseudoTimeStepMultiplier;      /*!< dynamically increases number of pseudo time steps if Newton solver did not converge */

    Plato::ScalarVector mPressure;                /*!< projected pressure field */
    Plato::ScalarMultiVector mLocalStates;        /*!< local state variables */
    Plato::ScalarMultiVector mGlobalStates;       /*!< global state variables */
    Plato::ScalarMultiVector mReactionForce;      /*!< reaction */
    Plato::ScalarMultiVector mProjectedPressGrad; /*!< projected pressure gradient (# Time Steps, # Projected Pressure Gradient dofs) */

    Plato::ScalarVector mDirichletValues;         /*!< values associated with the Dirichlet boundary conditions */
    Plato::LocalOrdinalVector mDirichletDofs;     /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */

    Plato::WorksetBase<Plato::SimplexPlasticity<mSpaceDim>> mWorksetBase; /*!< assembly routine interface */

    std::shared_ptr<Plato::NewtonRaphsonSolver<PhysicsT>> mNewtonSolver;         /*!< Newton-Raphson solve interface */
    std::shared_ptr<Plato::PathDependentAdjointSolver<PhysicsT>> mAdjointSolver; /*!< Path-dependent adjoint solver interface */

    bool mStopOptimization; /*!< stops optimization problem if Newton-Raphson solver fails to converge during an optimization run */
    bool mMaxNumPseudoTimeStepsReached; /*!< use to check if maximum number of allowable pseudo time steps has been reached */

// public functions
public:
    /***************************************************************************//**
     * \brief Plasticity problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputs input parameters database
    *******************************************************************************/
    PlasticityProblem(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputs) :
            mLocalEquation(std::make_shared<Plato::LocalVectorFunctionInc<PlasticityT>>(aMesh, aMeshSets, mDataMap, aInputs)),
            mGlobalEquation(std::make_shared<Plato::GlobalVectorFunctionInc<PhysicsT>>(aMesh, aMeshSets, mDataMap, aInputs, aInputs.get<std::string>("PDE Constraint"))),
            mProjectionEquation(std::make_shared<Plato::VectorFunctionVMS<ProjectorT>>(aMesh, aMeshSets, mDataMap, aInputs, std::string("State Gradient Projection"))),
            mObjective(nullptr),
            mConstraint(nullptr),
            mNumPseudoTimeSteps(Plato::ParseTools::getSubParam<Plato::OrdinalType>(aInputs, "Time Stepping", "Initial Num. Pseudo Time Steps", 20)),
            mMaxNumPseudoTimeSteps(Plato::ParseTools::getSubParam<Plato::OrdinalType>(aInputs, "Time Stepping", "Maximum Num. Pseudo Time Steps", 80)),
            mPseudoTimeStep(1.0/(static_cast<Plato::Scalar>(mNumPseudoTimeSteps))),
            mInitialNormResidual(std::numeric_limits<Plato::Scalar>::max()),
            mDispControlConstant(std::numeric_limits<Plato::Scalar>::min()),
            mCurrentPseudoTimeStep(0.0),
            mPseudoTimeStepMultiplier(Plato::ParseTools::getSubParam<Plato::Scalar>(aInputs, "Time Stepping", "Expansion Multiplier", 2)),
            mPressure("Previous Pressure Field", aMesh.nverts()),
            mLocalStates("Local States", mNumPseudoTimeSteps, mLocalEquation->size()),
            mGlobalStates("Global States", mNumPseudoTimeSteps, mGlobalEquation->size()),
            mReactionForce("Reaction Force", mNumPseudoTimeSteps, aMesh.nverts()),
            mProjectedPressGrad("Projected Pressure Gradient", mNumPseudoTimeSteps, mProjectionEquation->size()),
            mWorksetBase(aMesh),
            mNewtonSolver(std::make_shared<Plato::NewtonRaphsonSolver<PhysicsT>>(aMesh, aInputs)),
            mAdjointSolver(std::make_shared<Plato::PathDependentAdjointSolver<PhysicsT>>(aMesh, aInputs)),
            mStopOptimization(false),
            mMaxNumPseudoTimeStepsReached(false)
    {
        this->initialize(aMesh, aMeshSets, aInputs);
    }

    /***************************************************************************//**
     * \brief Plasticity problem constructor
     * \param [in] aMesh mesh database
    *******************************************************************************/
    explicit PlasticityProblem(Omega_h::Mesh& aMesh) :
            mLocalEquation(nullptr),
            mGlobalEquation(nullptr),
            mProjectionEquation(nullptr),
            mObjective(nullptr),
            mConstraint(nullptr),
            mNumPseudoTimeSteps(20),
            mMaxNumPseudoTimeSteps(80),
            mPseudoTimeStep(1.0/(static_cast<Plato::Scalar>(mNumPseudoTimeSteps))),
            mInitialNormResidual(std::numeric_limits<Plato::Scalar>::max()),
            mDispControlConstant(std::numeric_limits<Plato::Scalar>::min()),
            mCurrentPseudoTimeStep(0.0),
            mPseudoTimeStepMultiplier(2),
            mPressure("Pressure Field", aMesh.nverts()),
            mLocalStates("Local States", mNumPseudoTimeSteps, aMesh.nelems() * mNumLocalDofsPerCell),
            mGlobalStates("Global States", mNumPseudoTimeSteps, aMesh.nverts() * mNumGlobalDofsPerNode),
            mReactionForce("Reaction Force", mNumPseudoTimeSteps, aMesh.nverts()),
            mProjectedPressGrad("Projected Pressure Gradient", mNumPseudoTimeSteps, aMesh.nverts() * mNumPressGradDofsPerNode),
            mWorksetBase(aMesh),
            mNewtonSolver(std::make_shared<Plato::NewtonRaphsonSolver<PhysicsT>>(aMesh)),
            mAdjointSolver(std::make_shared<Plato::PathDependentAdjointSolver<PhysicsT>>(aMesh)),
            mStopOptimization(false),
            mMaxNumPseudoTimeStepsReached(false)
    {
    }

    /***************************************************************************//**
     * \brief PLATO Plasticity Problem destructor
    *******************************************************************************/
    virtual ~PlasticityProblem()
    {
    }

    /***************************************************************************//**
     * \brief Append objective function evaluation interface
     * \param [in] aInput objective function evaluation interface
    *******************************************************************************/
    void appendObjective(const std::shared_ptr<Plato::LocalScalarFunctionInc>& aInput)
    {
        mObjective = aInput;
    }

    /***************************************************************************//**
     * \brief Append constraint function evaluation interface
     * \param [in] aInput constraint function evaluation interface
    *******************************************************************************/
    void appendConstraint(const std::shared_ptr<Plato::LocalScalarFunctionInc>& aInput)
    {
        mConstraint = aInput;
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
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputs input parameters database
    *******************************************************************************/
    void readEssentialBoundaryConditions(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputs)
    {
        if(aInputs.isSublist("Essential Boundary Conditions") == false)
        {
            THROWERR("Plasticity Problem: Essential Boundary Conditions are not defined for this problem.")
        }
        Plato::EssentialBCs<PhysicsT> tDirichletBCs(aInputs.sublist("Essential Boundary Conditions", false));
        tDirichletBCs.get(aMeshSets, mDirichletDofs, mDirichletValues);
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
        mDirichletDofs = aDirichletDofs;
        mDirichletValues = aDirichletValues;
    }

    /***************************************************************************//**
     * \brief Save states to visualization file
     * \param [in] aFilepath output/viz directory path
     * \param [in] aMesh     Omega_h mesh database
    *******************************************************************************/
    void saveStates(const std::string& aFilepath, Omega_h::Mesh& aMesh)
    {
        auto tNumNodes = mGlobalEquation->numNodes();
        Plato::ScalarMultiVector tPressure("Pressure", mGlobalStates.extent(0), tNumNodes);
        Plato::ScalarMultiVector tDisplacements("Displacements", mGlobalStates.extent(0), tNumNodes*mSpaceDim);
        Plato::blas2::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(mGlobalStates, tPressure);
        Plato::blas2::extract<mNumGlobalDofsPerNode, mSpaceDim>(tNumNodes, mGlobalStates, tDisplacements);

        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aFilepath.c_str(), &aMesh, mSpaceDim);
        for(Plato::OrdinalType tSnapshot = 0; tSnapshot < tDisplacements.extent(0); tSnapshot++)
        {
            auto tPressSubView = Kokkos::subview(tPressure, tSnapshot, Kokkos::ALL());
            aMesh.add_tag(Omega_h::VERT, "Pressure", 1, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tPressSubView)));
            auto tForceSubView = Kokkos::subview(mReactionForce, tSnapshot, Kokkos::ALL());
            aMesh.add_tag(Omega_h::VERT, "Reaction Force", 1, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tForceSubView)));
            auto tDispSubView = Kokkos::subview(tDisplacements, tSnapshot, Kokkos::ALL());
            aMesh.add_tag(Omega_h::VERT, "Displacements", mSpaceDim, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tDispSubView)));
            Plato::add_element_state_tags(aMesh, mDataMap, tSnapshot);
            auto tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, mSpaceDim);
            auto tTime = mPseudoTimeStep * static_cast<Plato::Scalar>(tSnapshot + 1);
            tWriter.write(tSnapshot, tTime, tTags);
        }
    }

    /***************************************************************************//**
     * \brief Return number of global degrees of freedom in solution.
     * \return Number of global degrees of freedom
    *******************************************************************************/
    Plato::OrdinalType getNumSolutionDofs() override
    {
        return (mGlobalEquation->size());
    }

    /***************************************************************************//**
     * \brief Set global state variables
     * \param [in] aGlobalState 2D view of global state variables - (NumTimeSteps, TotalDofs)
    *******************************************************************************/
    void setGlobalState(const Plato::ScalarMultiVector & aGlobalState) override
    {
        assert(aGlobalState.extent(0) == mGlobalStates.extent(0));
        assert(aGlobalState.extent(1) == mGlobalStates.extent(1));
        Kokkos::deep_copy(mGlobalStates, aGlobalState);
    }

    /***************************************************************************//**
     * \brief Return 2D view of global state variables - (NumTimeSteps, TotalDofs)
     * \return 2D view of global state variables
    *******************************************************************************/
    Plato::ScalarMultiVector getGlobalState() override
    {
        return mGlobalStates;
    }

    /***************************************************************************//**
     * \brief Set local state variables
     * \param [in] aLocalState 2D view of local state variables, e.g. LS(NumTimeSteps, TotalDofs)
    *******************************************************************************/
    void setLocalState(const Plato::ScalarMultiVector & aLocalState) override
    {
        assert(aLocalState.extent(0) == mLocalStates.extent(0));
        assert(aLocalState.extent(1) == mLocalStates.extent(1));
        Kokkos::deep_copy(mLocalStates, aLocalState);
    }

    /***************************************************************************//**
     * \brief Return 2D view of local state variables, e.g. LS(NumTimeSteps, TotalDofs)
     * \return 2D view of global state variables
    *******************************************************************************/
    Plato::ScalarMultiVector getLocalState() override
    {
        return mLocalStates;
    }

    /***************************************************************************//**
     * \brief Return 2D view of global adjoint variables - (2, TotalDofs)
     * \return 2D view of global adjoint variables
    *******************************************************************************/
    Plato::ScalarMultiVector getAdjoint() override
    {
        THROWERR("PLASTICITY PROBLEM: ADJOINT MEMBER DATA IS NOT DEFINED");
    }

    /***************************************************************************//**
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    *******************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector) {return;}

    /***************************************************************************//**
     * \brief Fill right-hand-side vector values
    *******************************************************************************/
    void applyBoundaryLoads(const Plato::ScalarVector & aForce) override { return; }

    /***************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControls 1D container of control variables
     * \param [in] aGlobalState 2D container of global state variables
    *******************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControls,
                       const Plato::ScalarMultiVector & aGlobalState) override
    {
        mLocalEquation->updateProblem(aGlobalState, mLocalStates, aControls, mCurrentPseudoTimeStep);
        mGlobalEquation->updateProblem(aGlobalState, mLocalStates, aControls, mCurrentPseudoTimeStep);
        mProjectionEquation->updateProblem(aGlobalState, aControls, mCurrentPseudoTimeStep);

        if(mObjective != nullptr)
        {
            mObjective->updateProblem(aGlobalState, mLocalStates, aControls, mCurrentPseudoTimeStep);
        }

        if(mConstraint != nullptr)
        {
            mConstraint->updateProblem(aGlobalState, mLocalStates, aControls, mCurrentPseudoTimeStep);
        }
    }

    /***************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControls 1D view of control variables
     * \return 2D view of state variables
    *******************************************************************************/
    Plato::ScalarMultiVector solution(const Plato::ScalarVector &aControls) override
    {
        // TODO: NOTES
        // 1. WRITE LOCAL STATES, PRESSURE, AND GLOBAL STATES HISTORY TO FILE - MEMORY CONCERNS
        //   1.1. NO NEED TO STORE MEMBER DATA FOR THESE QUANTITIES
        //   1.2. READ DATA FROM FILES DURING ADJOINT SOLVE
        // 4. HOW WILL OUTPUT DATA BE PRESENTED TO THE USERS, WE CANNOT SEND TIME-DEPENDENT DATA THROUGH THE ENGINE.
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: INPUT CONTROL VECTOR IS EMPTY.")
        }

        bool tStop = false;
        while (tStop == false)
        {
            bool tDidSolverConverge = this->solveForwardProblem(aControls);
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
                if(mMaxNumPseudoTimeStepsReached == true)
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
                        << "Number of pseudo time steps will be increased to '" << mNumPseudoTimeSteps << "'\n.";
                REPORT(tMsg.str().c_str());
                mNewtonSolver->appendOutputMessage(tMsg);
            }
        }

        return mGlobalStates;
    }

    /***************************************************************************//**
     * \fn Plato::Scalar objectiveValue(const Plato::ScalarVector & aControls,
     *                                  const Plato::ScalarMultiVector & aGlobalState)
     * \brief Evaluate objective function and return its value
     * \param [in] aControls 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return objective function value
    *******************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControls,
                                 const Plato::ScalarMultiVector & aGlobalState) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aGlobalState.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: GLOBAL STATE 2D VIEW IS EMPTY.");
        }
        if(mObjective == nullptr)
        {
            THROWERR("PLASTICITY PROBLEM: OBJECTIVE PTR IS NULL.");
        }

        this->shouldOptimizationProblemStop();
        auto tOutput = this->evaluateCriterion(*mObjective, aGlobalState, mLocalStates, aControls);

        return (tOutput);
    }

    /***************************************************************************//**
     * \brief Evaluate objective function and return its value
     * \param [in] aControls 1D view of control variables
     * \return objective function value
    *******************************************************************************/
    Plato::Scalar objectiveValue(const Plato::ScalarVector & aControls) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(mObjective == nullptr)
        {
            THROWERR("PLASTICITY PROBLEM: OBJECTIVE PTR IS NULL.");
        }

        this->shouldOptimizationProblemStop();
        auto tOutput = this->evaluateCriterion(*mObjective, mGlobalStates, mLocalStates, aControls);

        return (tOutput);
    }

    /***************************************************************************//**
     * \brief Evaluate constraint function and return its value
     * \param [in] aControls 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return constraint function value
    *******************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControls,
                                  const Plato::ScalarMultiVector & aGlobalState) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aGlobalState.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: GLOBAL STATE 2D VIEW IS EMPTY.");
        }
        if(mConstraint == nullptr)
        {
            THROWERR("PLASTICITY PROBLEM: CONSTRAINT PTR IS NULL.");
        }

        this->shouldOptimizationProblemStop();
        auto tOutput = this->evaluateCriterion(*mConstraint, mGlobalStates, mLocalStates, aControls);

        return (tOutput);
    }

    /***************************************************************************//**
     * \brief Evaluate constraint function and return its value
     * \param [in] aControls 1D view of control variables
     * \return constraint function value
    *******************************************************************************/
    Plato::Scalar constraintValue(const Plato::ScalarVector & aControls) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(mConstraint == nullptr)
        {
            THROWERR("PLASTICITY PROBLEM: CONSTRAINT PTR IS NULL.");
        }

        this->shouldOptimizationProblemStop();
        auto tOutput = this->evaluateCriterion(*mConstraint, mGlobalStates, mLocalStates, aControls);

        return tOutput;
    }

    /***************************************************************************//**
     * \brief Evaluate objective partial derivative wrt control variables
     * \param [in] aControls 1D view of control variables
     * \return 1D view - objective partial derivative wrt control variables
    *******************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControls) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(mObjective == nullptr)
        {
            THROWERR("PLASTICITY PROBLEM: OBJECTIVE PTR IS NULL.");
        }

        this->shouldOptimizationProblemStop();
        auto tTotalDerivative = this->objectiveGradient(aControls, mGlobalStates);

        return tTotalDerivative;
    }

    /***************************************************************************//**
     * \brief Evaluate objective gradient wrt control variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aGlobalState 2D view of global state variables
     * \return 1D view of the objective gradient wrt control variables
    *******************************************************************************/
    Plato::ScalarVector objectiveGradient(const Plato::ScalarVector & aControls,
                                          const Plato::ScalarMultiVector & aGlobalState) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aGlobalState.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: GLOBAL STATE 2D VIEW IS EMPTY.");
        }
        if(mObjective == nullptr)
        {
            THROWERR("PLASTICITY PROBLEM: OBJECTIVE PTR IS NULL.");
        }

        this->shouldOptimizationProblemStop();
        mAdjointSolver->appendScalarFunction(mObjective);

        auto tNumNodes = mGlobalEquation->numNodes();
        Plato::ScalarVector tTotalDerivative("Total Derivative", tNumNodes);
        this->backwardTimeIntegration(Plato::PartialDerivative::CONTROL, aControls, tTotalDerivative);
        this->addCriterionPartialDerivativeZ(*mObjective, aControls, tTotalDerivative);

        return (tTotalDerivative);
    }

    /***************************************************************************//**
     * \brief Evaluate objective partial derivative wrt configuration variables
     * \param [in] aControls 1D view of control variables
     * \return 1D view - objective partial derivative wrt configuration variables
    *******************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControls) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(mObjective == nullptr)
        {
            THROWERR("PLASTICITY PROBLEM: OBJECTIVE PTR IS NULL.");
        }

        this->shouldOptimizationProblemStop();
        auto tTotalDerivative = this->objectiveGradientX(aControls, mGlobalStates);

        return tTotalDerivative;
    }

    /***************************************************************************//**
     * \brief Evaluate objective gradient wrt configuration variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aGlobalState 2D view of global state variables
     * \return 1D view of the objective gradient wrt control variables
    *******************************************************************************/
    Plato::ScalarVector objectiveGradientX(const Plato::ScalarVector & aControls,
                                           const Plato::ScalarMultiVector & aGlobalState) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aGlobalState.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: GLOBAL STATE 2D VIEW IS EMPTY.");
        }
        if(mObjective == nullptr)
        {
            THROWERR("PLASTICITY PROBLEM: OBJECTIVE PTR IS NULL.");
        }

        this->shouldOptimizationProblemStop();
        mAdjointSolver->appendScalarFunction(mObjective);
        Plato::ScalarVector tTotalDerivative("Total Derivative", mNumConfigDofsPerCell);
        this->backwardTimeIntegration(Plato::PartialDerivative::CONFIGURATION, aControls, tTotalDerivative);
        this->addCriterionPartialDerivativeX(*mObjective, aControls, tTotalDerivative);

        return (tTotalDerivative);
    }

    /***************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt control variables
     * \param [in] aControls 1D view of control variables
     * \return 1D view - constraint partial derivative wrt control variables
    *******************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControls) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(mConstraint == nullptr)
        {
            THROWERR("PLASTICITY PROBLEM: CONSTRAINT PTR IS NULL.");
        }

        auto tTotalDerivative = this->constraintGradient(aControls, mGlobalStates);

        return tTotalDerivative;
    }

    /***************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt control variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return 1D view - constraint partial derivative wrt control variables
    *******************************************************************************/
    Plato::ScalarVector constraintGradient(const Plato::ScalarVector & aControls,
                                           const Plato::ScalarMultiVector & aGlobalState) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aGlobalState.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: GLOBAL STATE 2D VIEW IS EMPTY.");
        }
        if(mConstraint == nullptr)
        {
            THROWERR("PLASTICITY PROBLEM: CONSTRAINT PTR IS NULL.");
        }

        this->shouldOptimizationProblemStop();
        mAdjointSolver->appendScalarFunction(mConstraint);
        auto tNumNodes = mGlobalEquation->numNodes();
        Plato::ScalarVector tTotalDerivative("Total Derivative", tNumNodes);
        this->backwardTimeIntegration(Plato::PartialDerivative::CONTROL, aControls, tTotalDerivative);
        this->addCriterionPartialDerivativeZ(*mConstraint, aControls, tTotalDerivative);

        return (tTotalDerivative);
    }

    /***************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt configuration variables
     * \param [in] aControls 1D view of control variables
     * \return 1D view - constraint partial derivative wrt configuration variables
    *******************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControls) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(mConstraint == nullptr)
        {
            THROWERR("PLASTICITY PROBLEM: CONSTRAINT PTR IS NULL.");
        }

        this->shouldOptimizationProblemStop();
        auto tTotalDerivative = this->constraintGradientX(aControls, mGlobalStates);

        return tTotalDerivative;
    }

    /***************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt configuration variables
     * \param [in] aControls 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return 1D view - constraint partial derivative wrt configuration variables
    *******************************************************************************/
    Plato::ScalarVector constraintGradientX(const Plato::ScalarVector & aControls,
                                            const Plato::ScalarMultiVector & aGlobalState) override
    {
        if(aControls.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: CONTROL 1D VIEW IS EMPTY.");
        }
        if(aGlobalState.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("PLASTICITY PROBLEM: GLOBAL STATE 2D VIEW IS EMPTY.");
        }
        if(mConstraint == nullptr)
        {
           THROWERR("PLASTICITY PROBLEM: CONSTRAINT PTR IS NULL.");
        }

        this->shouldOptimizationProblemStop();
        mAdjointSolver->appendScalarFunction(mConstraint);
        Plato::ScalarVector tTotalDerivative("Total Derivative", mNumConfigDofsPerCell);
        this->backwardTimeIntegration(Plato::PartialDerivative::CONFIGURATION, aControls, tTotalDerivative);
        this->addCriterionPartialDerivativeX(*mConstraint, aControls, tTotalDerivative);

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
                                                     aStates.mProjectedPressGrad, aControl, aStates.mCurrentStepIndex);

        auto tNumNodes = mGlobalEquation->numNodes();
        auto tReactionForce = Kokkos::subview(mReactionForce, aStates.mCurrentStepIndex, Kokkos::ALL());
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType &aOrdinal)
        {
            for(Plato::OrdinalType tDim = 0; tDim < mSpaceDim; tDim++)
            {
                tReactionForce(aOrdinal) += tInternalForce(aOrdinal*mNumGlobalDofsPerNode+tDim);
            }
        }, "reaction force");
    }

// private functions
private:
    /***************************************************************************//**
     * \brief Initialize member data
     * \param [in] aControls current set of controls, i.e. design variables
    *******************************************************************************/
    void initialize(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams)
    {
        this->allocateObjectiveFunction(aMesh, aMeshSets, aInputParams);
        this->allocateConstraintFunction(aMesh, aMeshSets, aInputParams);
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
     * \brief Resize time-dependent state containers and decrease pseudo-time step.
    *******************************************************************************/
    void resizeTimeDependentStates()
    {
        mNumPseudoTimeSteps = static_cast<Plato::OrdinalType>(mNumPseudoTimeSteps * mPseudoTimeStepMultiplier);
        if(mNumPseudoTimeSteps >= mMaxNumPseudoTimeSteps)
        {
            mNumPseudoTimeSteps = mMaxNumPseudoTimeSteps;
            mMaxNumPseudoTimeStepsReached = true;
        }
        mPseudoTimeStep = static_cast<Plato::Scalar>(1.0) / static_cast<Plato::Scalar>(mNumPseudoTimeSteps);

        Kokkos::resize(mLocalStates, mNumPseudoTimeSteps, mLocalEquation->size());
        Kokkos::resize(mGlobalStates, mNumPseudoTimeSteps, mGlobalEquation->size());
        Kokkos::resize(mReactionForce, mNumPseudoTimeSteps, mGlobalEquation->numNodes());
        Kokkos::resize(mProjectedPressGrad, mNumPseudoTimeSteps, mProjectionEquation->size());
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
    bool solveForwardProblem(const Plato::ScalarVector & aControls)
    {
        mDataMap.clearStates();

        Plato::CurrentStates tCurrentState;
        auto tNumCells = mLocalEquation->numCells();
        tCurrentState.mDeltaGlobalState = Plato::ScalarVector("Global State Increment", mGlobalEquation->size());

        this->initializeNewtonSolver();

        bool tForwardProblemSolved = false;
        for(Plato::OrdinalType tCurrentStepIndex = 0; tCurrentStepIndex < mNumPseudoTimeSteps; tCurrentStepIndex++)
        {
            std::stringstream tMsg;
            mCurrentPseudoTimeStep = mPseudoTimeStep * static_cast<Plato::Scalar>(tCurrentStepIndex + 1);
            tMsg << "TIME STEP #" << tCurrentStepIndex + static_cast<Plato::OrdinalType>(1) << " OUT OF " << mNumPseudoTimeSteps
                 << " TIME STEPS, TOTAL TIME = " << mCurrentPseudoTimeStep << "\n";
            mNewtonSolver->appendOutputMessage(tMsg);

            tCurrentState.mCurrentStepIndex = tCurrentStepIndex;
            this->cacheStateData(tCurrentState);

            // update displacement and load control multiplier
            this->updateDispAndLoadControlMultipliers(tCurrentStepIndex);

            // update local and global states
            bool tNewtonRaphsonConverged = mNewtonSolver->solve(aControls, tCurrentState);
            mDataMap.saveState();

            // compute reaction force
            this->computeReactionForce(aControls, tCurrentState);

            if(tNewtonRaphsonConverged == false)
            {
                std::stringstream tMsg;
                tMsg << "**** Newton-Raphson Solver did not converge at time step #"
                     << tCurrentStepIndex + static_cast<Plato::OrdinalType>(1)
                     << ".  Number of pseudo time steps will be increased to '"
                     << static_cast<Plato::OrdinalType>(mNumPseudoTimeSteps * mPseudoTimeStepMultiplier) << "'. ****\n\n";
                mNewtonSolver->appendOutputMessage(tMsg);
                return tForwardProblemSolved;
            }

            // update projected pressure gradient state
            this->updateProjectedPressureGradient(aControls, tCurrentState);
        }
        tForwardProblemSolved = true;
        return tForwardProblemSolved;
    }

    /***************************************************************************//**
     * \brief Update displacement and load control multiplier.
     * \param [in] aInput  current time step index
    *******************************************************************************/
    void updateDispAndLoadControlMultipliers(const Plato::OrdinalType& aInput)
    {
        mNewtonSolver->setDirichletValuesMultiplier(mPseudoTimeStep);
        auto tLoadControlConstant = mPseudoTimeStep * static_cast<Plato::Scalar>(aInput + 1);
        mDataMap.mScalarValues["LoadControlConstant"] = tLoadControlConstant;
    }

    /***************************************************************************//**
     * \brief Update projected pressure gradient.
     * \param [in]     aControls  1-D view of controls, e.g. design variables
     * \param [in/out] aStateData data manager with current and previous global and local state data
    *******************************************************************************/
    void updateProjectedPressureGradient(const Plato::ScalarVector &aControls,
                                         Plato::CurrentStates &aStateData)
    {
        Plato::OrdinalType tNextStepIndex = aStateData.mCurrentStepIndex + static_cast<Plato::OrdinalType>(1);
        if(tNextStepIndex >= mNumPseudoTimeSteps)
        {
            return;
        }

        // copy projection state, i.e. pressure
        Plato::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(aStateData.mCurrentGlobalState, mPressure);

        // compute projected pressure gradient
        auto tNextProjectedPressureGradient = Kokkos::subview(mProjectedPressGrad, tNextStepIndex, Kokkos::ALL());
        Plato::fill(0.0, tNextProjectedPressureGradient);
        auto tProjResidual = mProjectionEquation->value(tNextProjectedPressureGradient, mPressure, aControls, tNextStepIndex);
        auto tProjJacobian = mProjectionEquation->gradient_u(tNextProjectedPressureGradient, mPressure, aControls, tNextStepIndex);
        Plato::Solve::RowSummed<PhysicsT::mNumSpatialDims>(tProjJacobian, aStateData.mProjectedPressGrad, tProjResidual);
    }

    /***************************************************************************//**
     * \brief Get previous state
     * \param [in]     aCurrentStepIndex current time step index
     * \param [in]     aStates           states at each time step
     * \param [in/out] aOutput           previous state
    *******************************************************************************/
    void getPreviousState(const Plato::OrdinalType & aCurrentStepIndex,
                          const Plato::ScalarMultiVector & aStates,
                          Plato::ScalarVector & aOutput) const
    {
        auto tPreviousStepIndex = aCurrentStepIndex - static_cast<Plato::OrdinalType>(1);
        if(tPreviousStepIndex >= static_cast<Plato::OrdinalType>(0))
        {
            aOutput = Kokkos::subview(aStates, tPreviousStepIndex, Kokkos::ALL());
        }
        else
        {
            auto tLength = aStates.extent(1);
            aOutput = Plato::ScalarVector("Local State t=i-1", tLength);
            Plato::fill(0.0, aOutput);
        }
    }

    /***************************************************************************//**
     * \brief Evaluate criterion
     * \param [in] aCriterion   criterion scalar function interface
     * \param [in] aGlobalState global states for all time steps
     * \param [in] aLocalState  local states for all time steps
     * \param [in] aControls    current controls, e.g. design variables
     * \return new criterion value
    *******************************************************************************/
    Plato::Scalar evaluateCriterion(Plato::LocalScalarFunctionInc & aCriterion,
                                    const Plato::ScalarMultiVector & aGlobalState,
                                    const Plato::ScalarMultiVector & aLocalState,
                                    const Plato::ScalarVector & aControls)
    {
        Plato::ScalarVector tPreviousLocalState;
        Plato::ScalarVector tPreviousGlobalState;

        Plato::Scalar tOutput = 0;
        for(Plato::OrdinalType tCurrentStepIndex = 0; tCurrentStepIndex < mNumPseudoTimeSteps; tCurrentStepIndex++)
        {
            // SET CURRENT STATES
            auto tCurrentLocalState = Kokkos::subview(aLocalState, tCurrentStepIndex, Kokkos::ALL());
            auto tCurrentGlobalState = Kokkos::subview(aGlobalState, tCurrentStepIndex, Kokkos::ALL());

            // SET PREVIOUS AND FUTURE STATES
            this->getPreviousState(tCurrentStepIndex, aLocalState, tPreviousLocalState);
            this->getPreviousState(tCurrentStepIndex, aGlobalState, tPreviousGlobalState);

            tOutput += aCriterion.value(tCurrentGlobalState, tPreviousGlobalState,
                                        tCurrentLocalState, tPreviousLocalState,
                                        aControls, tCurrentStepIndex);
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
    void addCriterionPartialDerivativeZ(Plato::LocalScalarFunctionInc & aCriterion,
                                        const Plato::ScalarVector & aControls,
                                        Plato::ScalarVector & aTotalGradient)
    {
        Plato::ScalarVector tPreviousLocalState;
        Plato::ScalarVector tPreviousGlobalState;
        for(Plato::OrdinalType tCurrentStepIndex = 0; tCurrentStepIndex < mNumPseudoTimeSteps; tCurrentStepIndex++)
        {
            auto tCurrentLocalState = Kokkos::subview(mLocalStates, tCurrentStepIndex, Kokkos::ALL());
            auto tCurrentGlobalState = Kokkos::subview(mGlobalStates, tCurrentStepIndex, Kokkos::ALL());

            // SET PREVIOUS LOCAL STATES
            this->getPreviousState(tCurrentStepIndex, mLocalStates, tPreviousLocalState);
            this->getPreviousState(tCurrentStepIndex, mGlobalStates, tPreviousGlobalState);

            auto tDfDz = aCriterion.gradient_z(tCurrentGlobalState, tPreviousGlobalState,
                                               tCurrentLocalState, tPreviousLocalState,
                                               aControls, tCurrentStepIndex);
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
    void addCriterionPartialDerivativeX(Plato::LocalScalarFunctionInc & aCriterion,
                                        const Plato::ScalarVector & aControls,
                                        Plato::ScalarVector & aTotalGradient)
    {
        Plato::ScalarVector tPreviousLocalState;
        Plato::ScalarVector tPreviousGlobalState;
        for(Plato::OrdinalType tCurrentStepIndex = 0; tCurrentStepIndex < mNumPseudoTimeSteps; tCurrentStepIndex++)
        {
            auto tCurrentLocalState = Kokkos::subview(mLocalStates, tCurrentStepIndex, Kokkos::ALL());
            auto tCurrentGlobalState = Kokkos::subview(mGlobalStates, tCurrentStepIndex, Kokkos::ALL());

            // SET PREVIOUS AND FUTURE LOCAL STATES
            this->getPreviousState(tCurrentStepIndex, mLocalStates, tPreviousLocalState);
            this->getPreviousState(tCurrentStepIndex, mGlobalStates, tPreviousGlobalState);

            auto tDfDX = aCriterion.gradient_x(tCurrentGlobalState, tPreviousGlobalState,
                                               tCurrentLocalState, tPreviousLocalState,
                                               aControls, tCurrentStepIndex);
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
        mAdjointSolver->setNumPseudoTimeSteps(mNumPseudoTimeSteps);
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
        Plato::ForwardStates tCurrentStates(aType);
        Plato::ForwardStates tPreviousStates(aType);
        Plato::AdjointStates tAdjointStates(mGlobalEquation->size(), mLocalEquation->size(), mProjectionEquation->size());
        tAdjointStates.mInvLocalJacT = ScalarArray3D("Inv(DhDc)^T", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);

        this->initializeAdjointSolver();

        // outer loop for pseudo time steps
        auto tLastStepIndex = mNumPseudoTimeSteps - static_cast<Plato::OrdinalType>(1);
        for(tCurrentStates.mCurrentStepIndex = tLastStepIndex; tCurrentStates.mCurrentStepIndex >= 0; tCurrentStates.mCurrentStepIndex--)
        {
            tPreviousStates.mCurrentStepIndex = tCurrentStates.mCurrentStepIndex + 1;
            if(tPreviousStates.mCurrentStepIndex < mNumPseudoTimeSteps)
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
    void cacheStateData(Plato::CurrentStates & aStateData)
    {
        // GET CURRENT STATE
        aStateData.mCurrentLocalState = Kokkos::subview(mLocalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mCurrentGlobalState = Kokkos::subview(mGlobalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mProjectedPressGrad = Kokkos::subview(mProjectedPressGrad, aStateData.mCurrentStepIndex, Kokkos::ALL());

        // GET PREVIOUS STATE
        this->getPreviousState(aStateData.mCurrentStepIndex, mLocalStates, aStateData.mPreviousLocalState);
        this->getPreviousState(aStateData.mCurrentStepIndex, mGlobalStates, aStateData.mPreviousGlobalState);

        // SET ENTRIES IN CURRENT STATES TO ZERO
        Plato::fill(static_cast<Plato::Scalar>(0.0), aStateData.mCurrentLocalState);
        Plato::fill(static_cast<Plato::Scalar>(0.0), aStateData.mCurrentGlobalState);
        Plato::fill(static_cast<Plato::Scalar>(0.0), aStateData.mProjectedPressGrad);
        Plato::fill(static_cast<Plato::Scalar>(0.0), mPressure);
    }

    /***************************************************************************//**
     * \brief Update state data for time step n, i.e. current time step:
     * \param [in] aStateData state data manager
     * \param [in] aZeroEntries flag - zero all entries in current states (default = false)
    *******************************************************************************/
    void updateForwardState(Plato::ForwardStates & aStateData)
    {
        // GET CURRENT STATE
        aStateData.mCurrentLocalState = Kokkos::subview(mLocalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mCurrentGlobalState = Kokkos::subview(mGlobalStates, aStateData.mCurrentStepIndex, Kokkos::ALL());
        aStateData.mProjectedPressGrad = Kokkos::subview(mProjectedPressGrad, aStateData.mCurrentStepIndex, Kokkos::ALL());
        if(aStateData.mPressure.size() <= static_cast<Plato::OrdinalType>(0))
        {
            auto tNumVerts = mGlobalEquation->getMesh().nverts();
            aStateData.mPressure = Plato::ScalarVector("Current Pressure Field", tNumVerts);
        }
        Plato::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(aStateData.mCurrentGlobalState, aStateData.mPressure);

        // GET PREVIOUS STATE.
        this->getPreviousState(aStateData.mCurrentStepIndex, mLocalStates, aStateData.mPreviousLocalState);
        this->getPreviousState(aStateData.mCurrentStepIndex, mGlobalStates, aStateData.mPreviousGlobalState);
    }

    /***************************************************************************//**
     * \brief Update adjoint data for time step n, i.e. current time step:
     * \param [in] aAdjointData adjoint data manager
    *******************************************************************************/
    void updateAdjointState(Plato::AdjointStates& aAdjointStates)
    {
        // NOTE: CURRENT ADJOINT VARIABLES ARE UPDATED AT SOLVE TIME. THERE IS NO NEED TO SET THEM TO ZERO HERE.
        const Plato::Scalar tAlpha = 1.0; const Plato::Scalar tBeta = 0.0;
        Plato::update(tAlpha, aAdjointStates.mCurrentLocalAdjoint, tBeta, aAdjointStates.mPreviousLocalAdjoint);
        Plato::update(tAlpha, aAdjointStates.mCurrentGlobalAdjoint, tBeta, aAdjointStates.mPreviousGlobalAdjoint);
        Plato::update(tAlpha, aAdjointStates.mProjPressGradAdjoint, tBeta, aAdjointStates.mPreviousProjPressGradAdjoint);
    }

    /***************************************************************************//**
     * \brief Allocate objective function interface and adjoint containers
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
    *******************************************************************************/
    void allocateObjectiveFunction(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams)
    {
        if(aInputParams.isType<std::string>("Objective"))
        {
            auto tUserDefinedName = aInputParams.get<std::string>("Objective");
            Plato::PathDependentScalarFunctionFactory<PhysicsT> tObjectiveFunctionFactory;
            mObjective = tObjectiveFunctionFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tUserDefinedName);
        }
        else
        {
            REPORT("Plasticity Problem: Objective Function is disabled for this problem.")
        }
    }

    /***************************************************************************//**
     * \brief Allocate constraint function interface and adjoint containers
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
    *******************************************************************************/
    void allocateConstraintFunction(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams)
    {
        if(aInputParams.isType<std::string>("Constraint"))
        {
            Plato::PathDependentScalarFunctionFactory<PhysicsT> tContraintFunctionFactory;
            auto tUserDefinedName = aInputParams.get<std::string>("Constraint");
            mConstraint = tContraintFunctionFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tUserDefinedName);
        }
        else
        {
            REPORT("Plasticity Problem: Constraint function is disabled for this problem.")
        }
    }
};
// class PlasticityProblem

}
// namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::PlasticityProblem<Plato::InfinitesimalStrainPlasticity<3>>;
#endif


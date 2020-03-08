/*
 * NewtonRaphsonSolver.hpp
 *
 *  Created on: Mar 2, 2020
 */

#pragma once

#include "BLAS2.hpp"
#include "BLAS3.hpp"
#include "ParseTools.hpp"
#include "Plato_Solve.hpp"
#include "ApplyConstraints.hpp"
#include "NewtonRaphsonUtilities.hpp"
#include "LocalVectorFunctionInc.hpp"
#include "GlobalVectorFunctionInc.hpp"
#include "InfinitesimalStrainPlasticity.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Newton-Raphson solver interface.  This interface is responsible for
 * calling the Newton-Raphson solver and providing a new state.  For instance,
 * for infinitesimal strain plasticity problems, it updates the set of global
 * and local states.
 *
 * \tparam PhysicsT physics type, e.g. Plato::InfinitesimalStrainPlasticity
 *
*******************************************************************************/
template<typename PhysicsT>
class NewtonRaphsonSolver
{
// private member data
private:
    static constexpr auto mNumSpatialDims = PhysicsT::mNumSpatialDims;           /*!< spatial dimensions*/
    static constexpr auto mNumGlobalDofsPerCell = PhysicsT::mNumDofsPerCell;     /*!< number of global degrees of freedom per node*/
    static constexpr auto mNumGlobalDofsPerNode = PhysicsT::mNumDofsPerNode;     /*!< number of global degrees of freedom per node*/
    static constexpr auto mNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell; /*!< number of local degrees of freedom per cell (i.e. element)*/

    using LocalPhysicsT = typename Plato::Plasticity<mNumSpatialDims>;
    std::shared_ptr<Plato::GlobalVectorFunctionInc<PhysicsT>> mGlobalEquation;    /*!< global state residual interface */
    std::shared_ptr<Plato::LocalVectorFunctionInc<LocalPhysicsT>> mLocalEquation; /*!< local state residual interface*/
    Plato::WorksetBase<Plato::SimplexPlasticity<mNumSpatialDims>> mWorksetBase;   /*!< interface for assembly routines */

    Plato::Scalar mStoppingTolerance;            /*!< stopping tolerance */
    Plato::Scalar mDirichletValuesMultiplier;    /*!< multiplier for Dirichlet values */
    Plato::Scalar mCurrentResidualNormTolerance; /*!< current residual norm stopping tolerance - avoids unnecessary solves */

    Plato::OrdinalType mMaxNumSolverIter;  /*!< maximum number of iterations */
    Plato::OrdinalType mCurrentSolverIter; /*!< current number of iterations */

    Plato::ScalarVector mDirichletValues;     /*!< Dirichlet boundary conditions values */
    Plato::LocalOrdinalVector mDirichletDofs; /*!< Dirichlet boundary conditions degrees of freedom */

    bool mUseAbsoluteTolerance;   /*!< use absolute stopping tolerance flag */
    bool mWriteSolverDiagnostics; /*!< write solver diagnostics flag */
    std::ofstream mSolverDiagnosticsFile; /*!< output solver diagnostics */

// private functions
private:
    /***************************************************************************//**
     * \brief Open Newton-Raphson solver diagnostics file.
    *******************************************************************************/
    void openDiagnosticsFile()
    {
        if (mWriteSolverDiagnostics == false)
        {
            return;
        }

        mSolverDiagnosticsFile.open("plato_analyze_newton_raphson_diagnostics.txt");
    }

    /***************************************************************************//**
     * \brief Close Newton-Raphson solver diagnostics file.
    *******************************************************************************/
    void closeDiagnosticsFile()
    {
        if (mWriteSolverDiagnostics == false)
        {
            return;
        }

        mSolverDiagnosticsFile.close();
    }

    /***************************************************************************//**
     * \brief Update the inverse of the local Jacobian with respect to the
     * current local states.
     * \param [in]     aControls set of control variables
     * \param [in]     aStates   C++ structure holding the current state variables
     * \param [in/out] Output    inverse Jacobian
    *******************************************************************************/
    void updateInverseLocalJacobian(const Plato::ScalarVector & aControls,
                                    const Plato::CurrentStates & aStates,
                                    Plato::ScalarArray3D& Output)
    {
        auto tNumCells = mLocalEquation->numCells();
        auto tDhDc = mLocalEquation->gradient_c(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                  aStates.mCurrentLocalState , aStates.mPreviousLocalState,
                                                  aControls, aStates.mCurrentStepIndex);
        Plato::inverse_matrix_workset<mNumLocalDofsPerCell, mNumLocalDofsPerCell>(tNumCells, tDhDc, Output);
    }

    /***************************************************************************//**
     * \brief Apply Dirichlet constraints to system of equations
     * \param [in/out] aMatrix   right hand side matrix
     * \param [in/out] aResidual left hand side vector
    *******************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aResidual)
    {
        Plato::ScalarVector tDispControlledDirichletValues("Dirichlet Values", mDirichletValues.size());
        Plato::fill(0.0, tDispControlledDirichletValues);
        if(mCurrentSolverIter == static_cast<Plato::OrdinalType>(0))
        {
            Plato::update(mDirichletValuesMultiplier, mDirichletValues, static_cast<Plato::Scalar>(0.), tDispControlledDirichletValues);
        }

        if(aMatrix->isBlockMatrix())
        {
            Plato::applyBlockConstraints<mNumGlobalDofsPerNode>(aMatrix, aResidual, mDirichletDofs, tDispControlledDirichletValues);
        }
        else
        {
            Plato::applyConstraints<mNumGlobalDofsPerNode>(aMatrix, aResidual, mDirichletDofs, tDispControlledDirichletValues);
        }
    }

    /***************************************************************************//**
     * \brief Update global states, i.e.
     *
     *   1. Solve for \f$ \delta{u} \f$, and
     *   2. Update global states, \f$ u_{i+1} = u_{i} + \delta{u} \f$,
     *
     * \param [in]     aMatrix   right hand side matrix
     * \param [in]     aResidual left hand side vector
     * \param [in/out] aStates   C++ structure with the most recent set of state variables
    *******************************************************************************/
    void updateGlobalStates(Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
                            Plato::ScalarVector aResidual,
                            Plato::CurrentStates &aStates)
    {
        const Plato::Scalar tAlpha = 1.0;
        Plato::fill(static_cast<Plato::Scalar>(0.0), aStates.mDeltaGlobalState);
        Plato::Solve::Consistent<mNumGlobalDofsPerNode>(aMatrix, aStates.mDeltaGlobalState, aResidual, mUseAbsoluteTolerance);
        Plato::update(tAlpha, aStates.mDeltaGlobalState, tAlpha, aStates.mCurrentGlobalState);
    }

    /***************************************************************************//**
     * \brief Compute Schur complement, i.e.
     *
     * \f$ \frac{\partial{R}}{\partial{c}} * \frac{\partial{H}}{\partial{c}}^{-1} *
     *     \frac{\partial{H}}{\partial{u}} \f$,
     *
     * where \f$ R \f$ is the global residual, \f$ H \f$ is the local residual,
     * \f$ u \f$ are the global states, and \f$ c \f$ are the local state variables.
     *
     * \param [in] aControls         current set of design variables
     * \param [in] aStates           C++ structure with the most recent set of state variables
     * \param [in] aInvLocalJacobian inverse of local Jacobian wrt local states
     *
     * \return Schur complement for each cell/element
    *******************************************************************************/
    Plato::ScalarArray3D computeSchurComplement(const Plato::ScalarVector & aControls,
                                                const CurrentStates & aStates,
                                                const Plato::ScalarArray3D & aInvLocalJacobian)
    {
        // Compute cell Jacobian of the local residual with respect to the current global state WorkSet (WS)
        auto tDhDu = mLocalEquation->gradient_u(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                 aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                                 aControls, aStates.mCurrentStepIndex);

        // Compute cell C = (dH/dc)^{-1}*dH/du, where H is the local residual, c are the local states and u are the global states
        Plato::Scalar tBeta = 0.0;
        const Plato::Scalar tAlpha = 1.0;
        auto tNumCells = mLocalEquation->numCells();
        Plato::ScalarArray3D tInvDhDcTimesDhDu("InvDhDc times DhDu", tNumCells, mNumLocalDofsPerCell, mNumGlobalDofsPerCell);
        Plato::multiply_matrix_workset(tNumCells, tAlpha, aInvLocalJacobian, tDhDu, tBeta, tInvDhDcTimesDhDu);

        // Compute cell Jacobian of the global residual with respect to the current local state WorkSet (WS)
        auto tDrDc = mGlobalEquation->gradient_c(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                  aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                                  aStates.mProjectedPressGrad, aControls, aStates.mCurrentStepIndex);

        // Compute cell Schur = dR/dc * (dH/dc)^{-1} * dH/du, where H is the local residual,
        // R is the global residual, c are the local states and u are the global states
        Plato::ScalarArray3D tSchurComplement("Schur Complement", tNumCells, mNumGlobalDofsPerCell, mNumGlobalDofsPerCell);
        Plato::multiply_matrix_workset(tNumCells, tAlpha, tDrDc, tInvDhDcTimesDhDu, tBeta, tSchurComplement);

        return tSchurComplement;
    }

    /***************************************************************************//**
     * \brief Assemble tangent matrix, i.e.
     *
     * \f$ \frac{\partial{R}}{\partial{u}} - \left( \frac{\partial{R}}{\partial{c}}
     *   * \frac{\partial{H}}{\partial{c}}^{-1} * \frac{\partial{H}}{\partial{u}} \right) \f$
     *
     * where \f$ R \f$ is the global residual, \f$ H \f$ is the local residual,
     * \f$ u \f$ are the global states, and \f$ c \f$ are the local state variables.
     *
     * \param [in] aControls         current set of design variables
     * \param [in] aStates           C++ structure with the most recent set of state variables
     * \param [in] aInvLocalJacobian inverse of local Jacobian wrt local states
     *
     * \return Assembled tangent matrix
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    assembleTangentMatrix(const Plato::ScalarVector & aControls,
                          const Plato::CurrentStates & aStates,
                          const Plato::ScalarArray3D& aInvLocalJacobianT)
    {
        // Compute cell Schur Complement, i.e. dR/dc * (dH/dc)^{-1} * dH/du, where H is the local
        // residual, R is the global residual, c are the local states and u are the global states
        auto tSchurComplement = this->computeSchurComplement(aControls, aStates, aInvLocalJacobianT);

        // Compute cell Jacobian of the global residual with respect to the current global state WorkSet (WS)
        auto tDrDu = mGlobalEquation->gradient_u(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                   aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                                   aStates.mProjectedPressGrad, aControls, aStates.mCurrentStepIndex);

        // Add cell Schur complement to dR/du, where R is the global residual and u are the global states
        const Plato::Scalar tBeta = 1.0;
        const Plato::Scalar tAlpha = -1.0;
        auto tNumCells = mGlobalEquation->numCells();
        Plato::update_array_3D(tNumCells, tAlpha, tSchurComplement, tBeta, tDrDu);

        // Assemble full Jacobian
        auto tMesh = mGlobalEquation->getMesh();
        auto tGlobalJacobian = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumGlobalDofsPerNode, mNumGlobalDofsPerNode>(&tMesh);
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumGlobalDofsPerNode> tGlobalJacEntryOrdinal(tGlobalJacobian, &tMesh);
        auto tJacEntries = tGlobalJacobian->entries();
        Plato::assemble_jacobian(tNumCells, mNumGlobalDofsPerCell, mNumGlobalDofsPerCell, tGlobalJacEntryOrdinal, tDrDu, tJacEntries);

        return tGlobalJacobian;
    }

    /***************************************************************************//**
     * \brief Assemble residual vector, i.e.
     *
     * \f$ R - \left( \frac{\partial{R}}{\partial{c} * \frac{\partial{H}}{\partial{c}}^{-1} * H \right) \f$,
     *
     * where \f$ R \f$ is the global residual, \f$ H \f$ is the local residual, and
     * \f$ c \f$ are the local state variables.
     *
     * \param [in] aControls         current set of design variables
     * \param [in] aStates           C++ structure with the most recent set of state variables
     * \param [in] aInvLocalJacobian inverse of local Jacobian wrt local states
     *
     * \return Assembled tangent matrix
    *******************************************************************************/
    Plato::ScalarVector assembleResidual(const Plato::ScalarVector & aControls,
                                         const Plato::CurrentStates & aStates,
                                         const Plato::ScalarArray3D& aInvLocalJacobianT)
    {
        auto tGlobalResidual =
            mGlobalEquation->value(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                     aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                     aStates.mProjectedPressGrad, aControls, aStates.mCurrentStepIndex);

        // compute local residual workset (WS)
        auto tLocalResidualWS =
                mLocalEquation->valueWorkSet(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                               aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                               aControls, aStates.mCurrentStepIndex);

        // compute inv(DhDc)*h, where h is the local residual and DhDc is the local jacobian
        auto tNumCells = mLocalEquation->numCells();
        const Plato::Scalar tAlpha = 1.0; const Plato::Scalar tBeta = 0.0;
        Plato::ScalarMultiVector tInvLocalJacTimesLocalRes("InvLocalJacTimesLocalRes", tNumCells, mNumLocalDofsPerCell);
        Plato::matrix_times_vector_workset("N", tAlpha, aInvLocalJacobianT, tLocalResidualWS, tBeta, tInvLocalJacTimesLocalRes);

        // compute DrDc*inv(DhDc)*h
        Plato::ScalarMultiVector tLocalResidualTerm("LocalResidualTerm", tNumCells, mNumGlobalDofsPerCell);
        auto tDrDc = mGlobalEquation->gradient_c(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                   aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                                   aStates.mProjectedPressGrad, aControls, aStates.mCurrentStepIndex);
        Plato::matrix_times_vector_workset("N", tAlpha, tDrDc, tInvLocalJacTimesLocalRes, tBeta, tLocalResidualTerm);

        // assemble local residual contribution
        const auto tNumNodes = mGlobalEquation->numNodes();
        const auto tTotalNumDofs = mNumGlobalDofsPerNode * tNumNodes;
        Plato::ScalarVector  tLocalResidualContribution("Assembled Local Residual", tTotalNumDofs);
        mWorksetBase.assembleResidual(tLocalResidualTerm, tLocalResidualContribution);

        // add local residual contribution to global residual, i.e. r - DrDc*inv(DhDc)*h
        Plato::axpy(static_cast<Plato::Scalar>(-1.0), tLocalResidualContribution, tGlobalResidual);

        return (tGlobalResidual);
    }

    /***************************************************************************//**
      * \brief Initialize Newton-Raphson solver
      * \param [in\out] aStates C++ structure with the most recent set of state variables
     *******************************************************************************/
    void initializeSolver(Plato::CurrentStates & aStates)
    {
        mCurrentSolverIter = 0;
        Plato::update(1.0, aStates.mPreviousLocalState, 0.0, aStates.mCurrentLocalState);
        Plato::update(1.0, aStates.mPreviousGlobalState, 0.0, aStates.mCurrentGlobalState);
    }

    /***************************************************************************//**
      * \brief Check Newton-Raphson solver stopping criteria
      * \param [in\out] aOutputData C++ structure with the solver output diagnostics
     *******************************************************************************/
    bool checkStoppingCriterion(Plato::NewtonRaphsonOutputData & aOutputData)
    {
        bool tStop = false;

        if(aOutputData.mRelativeNorm < mStoppingTolerance)
        {
            tStop = true;
            aOutputData.mStopingCriterion = Plato::NewtonRaphson::RELATIVE_NORM_TOLERANCE;
        }
        else if(aOutputData.mCurrentNorm < mCurrentResidualNormTolerance)
        {
            tStop = true;
            aOutputData.mStopingCriterion = Plato::NewtonRaphson::CURRENT_NORM_TOLERANCE;
        }
        else if(aOutputData.mCurrentIteration >= mMaxNumSolverIter)
        {
            tStop = true;
            aOutputData.mStopingCriterion = Plato::NewtonRaphson::MAX_NUMBER_ITERATIONS;
        }

        return (tStop);
    }

// public functions
public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh   Omega_h mesh database
     * \param [in] aInputs input parameters database
    *******************************************************************************/
    NewtonRaphsonSolver(Omega_h::Mesh& aMesh, Teuchos::ParameterList& aInputs) :
        mWorksetBase(aMesh),
        mStoppingTolerance(Plato::ParseTools::getSubParam<Plato::Scalar>(aInputs, "Newton-Raphson", "Stopping Tolerance", 1e-6)),
        mDirichletValuesMultiplier(1),
        mCurrentResidualNormTolerance(Plato::ParseTools::getSubParam<Plato::Scalar>(aInputs, "Newton-Raphson", "Current Residual Norm Stopping Tolerance", 1e-10)),
        mMaxNumSolverIter(Plato::ParseTools::getSubParam<Plato::OrdinalType>(aInputs, "Newton-Raphson", "Maximum Number Iterations", 10)),
        mCurrentSolverIter(0),
        mUseAbsoluteTolerance(false),
        mWriteSolverDiagnostics(true)
    {
        this->openDiagnosticsFile();
        auto tInitialNumTimeSteps = Plato::ParseTools::getSubParam<Plato::OrdinalType>(aInputs, "Time Stepping", "Initial Num. Pseudo Time Steps", 20);
        mDirichletValuesMultiplier = static_cast<Plato::Scalar>(1.0) / static_cast<Plato::Scalar>(tInitialNumTimeSteps);
    }

    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh Omega_h mesh database
    *******************************************************************************/
    explicit NewtonRaphsonSolver(Omega_h::Mesh& aMesh) :
        mWorksetBase(aMesh),
        mStoppingTolerance(1e-6),
        mDirichletValuesMultiplier(1),
        mCurrentResidualNormTolerance(1e-10),
        mMaxNumSolverIter(20),
        mCurrentSolverIter(0),
        mUseAbsoluteTolerance(false),
        mWriteSolverDiagnostics(true)
    {
        this->openDiagnosticsFile();
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    ~NewtonRaphsonSolver()
    {
        this->closeDiagnosticsFile();
    }

    /***************************************************************************//**
     * \brief Set multiplier for Dirichlet values
     * \param [in] aInput multiplier
    *******************************************************************************/
    void setDirichletValuesMultiplier(const Plato::Scalar & aInput)
    {
        mDirichletValuesMultiplier = aInput;
    }

    /***************************************************************************//**
     * \brief Append local system of equation interface
     * \param [in] aInput local system of equation interface
    *******************************************************************************/
    void appendLocalEquation(const std::shared_ptr<Plato::LocalVectorFunctionInc<LocalPhysicsT>> & aInput)
    {
        mLocalEquation = aInput;
    }

    /***************************************************************************//**
     * \brief Append global system of equation interface
     * \param [in] aInput global system of equation interface
    *******************************************************************************/
    void appendGlobalEquation(const std::shared_ptr<Plato::GlobalVectorFunctionInc<PhysicsT>> & aInput)
    {
        mGlobalEquation = aInput;
    }

    /***************************************************************************//**
     * \brief Append vector of Dirichlet values
     * \param [in] aInput vector of Dirichlet values
    *******************************************************************************/
    void appendDirichletValues(const Plato::ScalarVector & aInput)
    {
        mDirichletValues = aInput;
    }

    /***************************************************************************//**
     * \brief Append vector of Dirichlet degrees of freedom
     * \param [in] aInput vector of Dirichlet degrees of freedom
    *******************************************************************************/
    void appendDirichletDofs(const Plato::LocalOrdinalVector & aInput)
    {
        mDirichletDofs = aInput;
    }

    /***************************************************************************//**
     * \brief Append output message to Newton-Raphson solver diagnostics file.
     * \param [in] aInput output message
    *******************************************************************************/
    void appendOutputMessage(const std::stringstream & aInput)
    {
        mSolverDiagnosticsFile << aInput.str().c_str();
    }

    /***************************************************************************//**
     * \brief Call Newton-Raphson solver and find new state
     * \param [in] aControls           1-D view of controls, e.g. design variables
     * \param [in] aStateData         data manager with current and previous state data
     * \param [in] aInvLocalJacobianT 3-D container for inverse Jacobian
     * \return Indicates if the Newton-Raphson solver converged (flag)
    *******************************************************************************/
    bool solve(const Plato::ScalarVector & aControls, Plato::CurrentStates & aStates)
    {
        bool tNewtonRaphsonConverged = false;
        Plato::NewtonRaphsonOutputData tOutputData;
        auto tNumCells = mLocalEquation->numCells();
        Plato::ScalarArray3D tInvLocalJacobianT("Inverse Transpose DhDc", tNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);

        tOutputData.mWriteOutput = mWriteSolverDiagnostics;
        Plato::print_newton_raphson_diagnostics_header(tOutputData, mSolverDiagnosticsFile);

        this->initializeSolver(aStates);
        while(true)
        {
            tOutputData.mCurrentIteration = mCurrentSolverIter;

            // update inverse of local Jacobian -> store in tInvLocalJacobianT
            this->updateInverseLocalJacobian(aControls, aStates, tInvLocalJacobianT);

            // assemble residual
            auto tGlobalResidual = this->assembleResidual(aControls, aStates, tInvLocalJacobianT);
            Plato::scale(static_cast<Plato::Scalar>(-1.0), tGlobalResidual);

            // assemble tangent stiffness matrix
            auto tGlobalJacobian = this->assembleTangentMatrix(aControls, aStates, tInvLocalJacobianT);

            // apply Dirichlet boundary conditions
            this->applyConstraints(tGlobalJacobian, tGlobalResidual);

            // check convergence
            Plato::compute_relative_residual_norm_criterion(tGlobalResidual, tOutputData);
            Plato::print_newton_raphson_diagnostics(tOutputData, mSolverDiagnosticsFile);

            const bool tStop = this->checkStoppingCriterion(tOutputData);
            if(tStop == true)
            {
                tNewtonRaphsonConverged = true;
                break;
            }

            // update global states
            this->updateGlobalStates(tGlobalJacobian, tGlobalResidual, aStates);

            // update local states
            mLocalEquation->updateLocalState(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                             aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                             aControls, aStates.mCurrentStepIndex);
            mCurrentSolverIter++;
        }

        Plato::print_newton_raphson_stop_criterion(tOutputData, mSolverDiagnosticsFile);

        return (tNewtonRaphsonConverged);
    }
};
// class NewtonRaphsonSolver

}
// namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::NewtonRaphsonSolver<Plato::InfinitesimalStrainPlasticity<3>>;
#endif


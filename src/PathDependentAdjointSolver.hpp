/*
 * PathDependentAdjointSolver.hpp
 *
 *  Created on: Mar 2, 2020
 */

#pragma once

#include "BLAS1.hpp"
#include "BLAS2.hpp"
#include "BLAS3.hpp"
#include "ParseTools.hpp"
#include "Projection.hpp"
#include "Plato_Solve.hpp"
#include "AnalyzeMacros.hpp"
#include "ApplyConstraints.hpp"
#include "VectorFunctionVMS.hpp"
#include "alg/PlatoAbstractSolver.hpp"
#include "LocalScalarFunctionInc.hpp"
#include "LocalVectorFunctionInc.hpp"
#include "GlobalVectorFunctionInc.hpp"
#include "InfinitesimalStrainPlasticity.hpp"
#include "InfinitesimalStrainThermoPlasticity.hpp"

namespace Plato
{

struct PartialDerivative
{
    enum derivative_t
    {
        CONTROL = 0,
        CONFIGURATION = 1,
    };
};

/***************************************************************************//**
 * \brief C++ structure used to solve path-dependent adjoint problems, e.g.
 * plasticity.  This structure holds the set of forward states at a given time
 * step during the backward time integration.
*******************************************************************************/
struct ForwardStates
{
    Plato::OrdinalType mCurrentStepIndex;     /*!< current time step index */

    Plato::ScalarVector mCurrentLocalState;   /*!< current local state */
    Plato::ScalarVector mPreviousLocalState;  /*!< previous local state */
    Plato::ScalarVector mCurrentGlobalState;  /*!< current global state */
    Plato::ScalarVector mPreviousGlobalState; /*!< previous global state */

    Plato::ScalarVector mPressure;            /*!< projected pressure */
    Plato::ScalarVector mProjectedPressGrad;  /*!< projected pressure gradient at time step k-1, where k is the step index */

    Plato::PartialDerivative::derivative_t mPartialDerivativeType;

    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aType partial derivative type
    *******************************************************************************/
    explicit ForwardStates(const Plato::PartialDerivative::derivative_t &aType) :
        mCurrentStepIndex(0),
        mPartialDerivativeType(aType)
    {
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    ~ForwardStates(){}
};
// struct ForwardStates

/***************************************************************************//**
 * \brief C++ structure holding the current adjoint variables
*******************************************************************************/
struct AdjointStates
{
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aNumGlobalAdjointVars        number of global adjoint variables
     * \param [in] aNumLocalAdjointVars         number of local adjoint variables
     * \param [in] aNumProjPressGradAdjointVars number of projected pressure gradient adjoint variables
    *******************************************************************************/
    AdjointStates(const Plato::OrdinalType & aNumGlobalAdjointVars,
                  const Plato::OrdinalType & aNumLocalAdjointVars,
                  const Plato::OrdinalType & aNumProjPressGradAdjointVars) :
            mCurrentLocalAdjoint(Plato::ScalarVector("Current Local Adjoint", aNumLocalAdjointVars)),
            mPreviousLocalAdjoint(Plato::ScalarVector("Previous Local Adjoint", aNumLocalAdjointVars)),
            mCurrentGlobalAdjoint(Plato::ScalarVector("Current Global Adjoint", aNumGlobalAdjointVars)),
            mPreviousGlobalAdjoint(Plato::ScalarVector("Previous Global Adjoint", aNumGlobalAdjointVars)),
            mProjPressGradAdjoint(Plato::ScalarVector("Current Projected Pressure Gradient Adjoint", aNumProjPressGradAdjointVars)),
            mPreviousProjPressGradAdjoint(Plato::ScalarVector("Previous Projected Pressure Gradient Adjoint", aNumProjPressGradAdjointVars))
    {
    }

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    ~AdjointStates(){}

    Plato::ScalarVector mCurrentLocalAdjoint;          /*!< current local adjoint */
    Plato::ScalarVector mPreviousLocalAdjoint;         /*!< previous local adjoint */
    Plato::ScalarVector mCurrentGlobalAdjoint;         /*!< current global adjoint */
    Plato::ScalarVector mPreviousGlobalAdjoint;        /*!< previous global adjoint */
    Plato::ScalarVector mProjPressGradAdjoint;         /*!< projected pressure adjoint */
    Plato::ScalarVector mPreviousProjPressGradAdjoint; /*!< projected pressure adjoint */

    Plato::ScalarArray3D mInvLocalJacT;                /*!< inverse of local Jacobian with respect to local states */
};
// struct AdjointStates


/***************************************************************************//**
 * \brief Path-dependent adjoint solver manager.  This interface enables the
 * evaluation of the functions responsible for updating the set of path-dependent
 * adjoint variables.
 *
 * \tparam PhysicsT global physics type, e.g. Plato::InfinitesimalStrainPlasticity
 *
*******************************************************************************/
template<typename PhysicsT>
class PathDependentAdjointSolver
{
private:
    static constexpr auto mNumSpatialDims = PhysicsT::mNumSpatialDims;                /*!< spatial dimensions */
    static constexpr auto mNumNodesPerCell = PhysicsT::mNumNodesPerCell;              /*!< number of nodes per cell */
    static constexpr auto mPressureDofOffset = PhysicsT::mPressureDofOffset;          /*!< number of pressure dofs offset */
    static constexpr auto mNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;      /*!< number of local degrees of freedom (dofs) per cell/element */
    static constexpr auto mNumGlobalDofsPerCell = PhysicsT::mNumDofsPerCell;          /*!< number of global degrees of freedom per cell/element */
    static constexpr auto mNumGlobalDofsPerNode = PhysicsT::mNumDofsPerNode;          /*!< number of global degrees of freedom per node */
    static constexpr auto mNumPressGradDofsPerCell = PhysicsT::mNumNodeStatePerCell;  /*!< number of projected pressure gradient dofs per cell */
    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell; /*!< number of configuration (i.e. coordinates) dofs per cell/element */

    using LocalPhysicsT = typename PhysicsT::LocalPhysicsT;
    using ProjectorT  = typename Plato::Projection<mNumSpatialDims, PhysicsT::mNumDofsPerNode, PhysicsT::mPressureDofOffset>;

    std::shared_ptr<Plato::LocalScalarFunctionInc> mCriterion;                    /*!< local criterion interface */ 
    std::shared_ptr<Plato::VectorFunctionVMS<ProjectorT>> mProjectionEquation;    /*!< global pressure gradient projection interface */
    std::shared_ptr<Plato::GlobalVectorFunctionInc<PhysicsT>> mGlobalEquation;    /*!< global equality constraint interface */
    std::shared_ptr<Plato::LocalVectorFunctionInc<LocalPhysicsT>> mLocalEquation; /*!< local equality constraint interface */

    Plato::WorksetBase<PhysicsT> mWorksetBase;   /*!< interface for assembly routines */

    Plato::OrdinalType mNumPseudoTimeSteps;   /*!< current number of pseudo time steps*/
    Plato::LocalOrdinalVector mDirichletDofs; /*!< Dirichlet boundary conditions degrees of freedom */

    std::shared_ptr<Plato::AbstractSolver> mLinearSolver; /*!< linear solver object */

private:
    /***************************************************************************//**
     * \brief Compute Schur complement, i.e.
     *
     * \f$ \frac{\partial{R}}{\partial{c}} * \frac{\partial{H}}{\partial{c}}^{-1} *
     *     \frac{\partial{H}}{\partial{u}} \f$,
     *
     * where \f$ R \f$ is the global residual, \f$ H \f$ is the local residual,
     * \f$ u \f$ are the global states, and \f$ c \f$ are the local state variables.
     *
     * \param [in] aControls    current set of design variables
     * \param [in] aStates      C++ structure that holds the current set of state variables
     * \param [in] aInvLocalJac inverse of local Jacobian wrt local states
     *
     * \return Schur complement for each cell/element
    *******************************************************************************/
    Plato::ScalarArray3D computeSchurComplement(const Plato::ScalarVector & aControls,
                                                const ForwardStates & aStates,
                                                const Plato::ScalarArray3D & aInvLocalJac)
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
        Plato::blas3::multiply(tNumCells, tAlpha, aInvLocalJac, tDhDu, tBeta, tInvDhDcTimesDhDu);

        // Compute cell Jacobian of the global residual with respect to the current local state WorkSet (WS)
        auto tDrDc = mGlobalEquation->gradient_c(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                 aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                                 aStates.mProjectedPressGrad, aControls, aStates.mCurrentStepIndex);

        // Compute cell Schur = dR/dc * (dH/dc)^{-1} * dH/du, where H is the local residual,
        // R is the global residual, c are the local states and u are the global states
        Plato::ScalarArray3D tSchurComplement("Schur Complement", tNumCells, mNumGlobalDofsPerCell, mNumGlobalDofsPerCell);
        Plato::blas3::multiply(tNumCells, tAlpha, tDrDc, tInvDhDcTimesDhDu, tBeta, tSchurComplement);

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
     * \param [in] aControls    current set of design variables
     * \param [in] aStates      C++ structure that holds the current set of state variables
     * \param [in] aInvLocalJac inverse of local Jacobian wrt local states
     *
     * \return Assembled tangent matrix
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    assembleTangentMatrix(const Plato::ScalarVector & aControls,
                          const Plato::ForwardStates & aStates,
                          const Plato::ScalarArray3D& aInvLocalJac)
    {
        // Compute cell Schur Complement, i.e. dR/dc * (dH/dc)^{-1} * dH/du, where H is the local
        // residual, R is the global residual, c are the local states and u are the global states
        auto tSchurComplement = this->computeSchurComplement(aControls, aStates, aInvLocalJac);

        // Compute cell Jacobian of the global residual with respect to the current global state WorkSet (WS)
        auto tDrDu = mGlobalEquation->gradient_u(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                 aStates.mCurrentLocalState, aStates.mPreviousLocalState,
                                                 aStates.mProjectedPressGrad, aControls, aStates.mCurrentStepIndex);

        // Add cell Schur complement to dR/du, where R is the global residual and u are the global states
        const Plato::Scalar tBeta = 1.0;
        const Plato::Scalar tAlpha = -1.0;
        auto tNumCells = mGlobalEquation->numCells();
        Plato::blas3::update(tNumCells, tAlpha, tSchurComplement, tBeta, tDrDu);

        // Assemble full Jacobian
        auto tMesh = mGlobalEquation->getMesh();
        auto tGlobalJacobian = Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumGlobalDofsPerNode, mNumGlobalDofsPerNode>(&tMesh);
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumGlobalDofsPerNode> tGlobalJacEntryOrdinal(tGlobalJacobian, &tMesh);
        auto tJacEntries = tGlobalJacobian->entries();
        Plato::assemble_jacobian(tNumCells, mNumGlobalDofsPerCell, mNumGlobalDofsPerCell, tGlobalJacEntryOrdinal, tDrDu, tJacEntries);

        return tGlobalJacobian;
    }

    /***************************************************************************//**
     * \brief Compute contribution from local residual to global adjoint
     *   right-hand-side vector as follows:
     * \f$
     *  t=k\ \mbox{time step k}
     *   \mathbf{F}_k =
     *     -\left(
     *          \frac{f}{u}_k + \frac{P}{u}_k^T \gamma_k
     *        - \frac{H}{u}_k^T \left( \frac{H}{c}_k^{-T} \left[ \frac{F}{c}_k + \frac{H}{c}_{k+1}^T\mu_{k+1} \right] \right)
     *      \right)
     *  t=N\ \mbox{final time step}
     *   \mathbf{F}_k =
     *     -\left(
     *          \frac{f}{u}_k - \frac{H}{u}_k^T \left( \frac{H}{c}_k^{-T} \frac{F}{c}_k \right)
     *      \right)
     * \f$
     * \param [in] aControls      design variables
     * \param [in] aCurStateVars  C++ structure that holds the current set of state variables
     * \param [in] aPrevStateVars C++ structure that holds the previous set of state variables
     * \param [in] aAdjointVars   C++ structure that holds the current set of adjoint variables
    *******************************************************************************/
    Plato::ScalarMultiVector
    computeLocalAdjointRHS(const Plato::ScalarVector &aControls,
                           const Plato::ForwardStates &aCurStateVars,
                           const Plato::ForwardStates &aPrevStateVars,
                           const Plato::AdjointStates & aAdjointVars)
    {
        // Compute partial derivative of objective with respect to current local states
        auto tDfDc = mCriterion->gradient_c(aCurStateVars.mCurrentGlobalState, aCurStateVars.mPreviousGlobalState,
                                            aCurStateVars.mCurrentLocalState, aCurStateVars.mPreviousLocalState,
                                            aControls, aCurStateVars.mCurrentStepIndex);

        auto tFinalStepIndex = mNumPseudoTimeSteps - static_cast<Plato::OrdinalType>(1);
        if(aCurStateVars.mCurrentStepIndex != tFinalStepIndex)
        {
            // Compute DfDx_k + DfDc_{k+1}, where k denotes the time step index
            const Plato::Scalar tAlpha = 1.0; const Plato::Scalar tBeta = 1.0;
            auto tDfDcp = mCriterion->gradient_cp(aPrevStateVars.mCurrentGlobalState, aPrevStateVars.mPreviousGlobalState,
                                                  aPrevStateVars.mCurrentLocalState, aPrevStateVars.mPreviousLocalState,
                                                  aControls, aCurStateVars.mCurrentStepIndex);
            Plato::blas2::update(tAlpha, tDfDcp, tBeta, tDfDc);

            // Compute DfDc_k + DfDc_{k+1} + ( DhDc_{k+1}^T * mu_{k+1} )
            auto tNumCells = mLocalEquation->numCells();
            Plato::ScalarMultiVector tPrevMu("Previous Local Adjoint Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aAdjointVars.mPreviousLocalAdjoint, tPrevMu);
            auto tDhDcp = mLocalEquation->gradient_cp(aPrevStateVars.mCurrentGlobalState, aPrevStateVars.mPreviousGlobalState,
                                                      aPrevStateVars.mCurrentLocalState , aPrevStateVars.mPreviousLocalState,
                                                      aControls, aCurStateVars.mCurrentStepIndex);
            Plato::blas2::matrix_times_vector("T", tAlpha, tDhDcp, tPrevMu, tBeta, tDfDc);

            // Compute DfDc_k + DfDc_{k+1} + ( DhDc_{k+1}^T * mu_{k+1} ) + ( DrDc_{k+1}^T * lambda_{k+1} )
            Plato::ScalarMultiVector tPrevLambda("Previous Global Adjoint Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aAdjointVars.mPreviousGlobalAdjoint, tPrevLambda);
            auto tDrDcp = mGlobalEquation->gradient_cp(aPrevStateVars.mCurrentGlobalState, aPrevStateVars.mPreviousGlobalState,
                                                       aPrevStateVars.mCurrentLocalState , aPrevStateVars.mPreviousLocalState,
                                                       aPrevStateVars.mProjectedPressGrad, aControls, aCurStateVars.mCurrentStepIndex);
            Plato::blas2::matrix_times_vector("T", tAlpha, tDrDcp, tPrevLambda, tBeta, tDfDc);
        }

        // Compute Inv(tDhDc_k^T) * [ DfDc_k + DfDc_{k+1} + ( DhDc_{k+1}^T * mu_{k+1} ) + ( DrDc_{k+1}^T * lambda_{k+1} ) ]
        auto tNumCells = mLocalEquation->numCells();
        const Plato::Scalar tAlpha = 1.0; const Plato::Scalar tBeta = 0.0;
        Plato::ScalarMultiVector tLocalStateWorkSet("InvLocalJacobianTimesLocalVec", tNumCells, mNumLocalDofsPerCell);
        Plato::blas2::matrix_times_vector("T", tAlpha, aAdjointVars.mInvLocalJacT, tDfDc, tBeta, tLocalStateWorkSet);

        // Compute local RHS <- tDhDu_k^T * { Inv(tDhDc_k^T) * [ DfDc_k + DfDc_{k+1} + ( DhDc_{k+1}^T * mu_{k+1} ) + ( DrDc_{k+1}^T * lambda_{k+1} ) }
        auto tDhDu = mLocalEquation->gradient_u(aCurStateVars.mCurrentGlobalState, aCurStateVars.mPreviousGlobalState,
                                                aCurStateVars.mCurrentLocalState , aCurStateVars.mPreviousLocalState,
                                                aControls, aCurStateVars.mCurrentStepIndex);
        Plato::ScalarMultiVector tLocalRHS("Local Adjoint RHS", tNumCells, mNumGlobalDofsPerCell);
        Plato::blas2::matrix_times_vector("T", tAlpha, tDhDu, tLocalStateWorkSet, tBeta, tLocalRHS);

        return (tLocalRHS);
    }

    /***************************************************************************//**
     * \brief Compute the right hand side vector needed to solve for the projected
     * pressure gradient adjoint variables.  The projected pressure gradient solve
     * is defined by:
     *
     * \f$  \left( \frac{\partial{P}_{k+1}}{\partial{\Pi}_k} \right)^{-1} *
     *       \frac{\partial{R}_{k+1}}{\partial{\Pi}_{k}} \lambda_{k+1} \f$,
     *
     * where \f$ P \f$ is the projected pressure gradient residual, \f$ \Pi \f$
     * is the projected pressure gradient, \f$ R \f$ is the global residual,
     * \f$ \lambda \f$ is the global adjoint field, and \f$ k \f$ is the time
     * step index.
     *
     * \param [in] aControls    current set of design variables
     * \param [in] aStateVars   C++ structure that holds the current set of state variables
     * \param [in] aAdjointVars C++ structure that holds the current set of adjoint variables
    *******************************************************************************/
    Plato::ScalarMultiVector
    computeProjPressGradAdjointRHS(const Plato::ScalarVector & aControls,
                                   const Plato::ForwardStates & aStateVars,
                                   const Plato::AdjointStates & aAdjointVars)
    {
        // Compute partial derivative of projected pressure gradient residual wrt pressure field, i.e. DpDn
        auto tDpDn = mProjectionEquation->gradient_n_workset(aStateVars.mProjectedPressGrad, aStateVars.mPressure,
                                                              aControls, aStateVars.mCurrentStepIndex);

        // Compute projected pressure gradient adjoint workset
        auto tNumCells = mProjectionEquation->numCells();
        Plato::ScalarMultiVector tGamma("Projected Pressure Gradient Adjoint", tNumCells, mNumPressGradDofsPerCell);
        mWorksetBase.worksetNodeState(aAdjointVars.mProjPressGradAdjoint, tGamma);

        // Compute DpDn_k^T * gamma_k
        const Plato::Scalar tAlpha = 1.0; const Plato::Scalar tBeta = 1.0;
        const auto tNumPressureDofsPerCell = mProjectionEquation->numNodeStatePerCell();
        Plato::ScalarMultiVector tOuput("DpDn_{k+1}^T * gamma_{k+1}", tNumCells, tNumPressureDofsPerCell);
        Plato::blas2::matrix_times_vector("T", tAlpha, tDpDn, tGamma, tBeta, tOuput);

        return (tOuput);
    }

    /***************************************************************************//**
     * \brief Assemble global adjoint right hand side vector, which is given by:
     *
     * \f$ \mathbf{f} = \left(\frac{\partial{f}}{\partial{u}}\right)_{t=n} - \left(
     * \frac{\partial{H}}{\partial{u}} \right)_{t=n}^{T} * \left[ \left( \left(
     * \frac{\partial{H}}{\partial{c}}\right)_{t=n}^{T} \right)^{-1} * \left(
     * \frac{\partial{f}}{\partial{c}} + \frac{\partial{H}}{\partial{v}}
     * \right)_{t=n+1}^{T} \gamma_{n+1} \right] \f$,
     *
     * where R is the global residual, H is the local residual, u is the global state,
     * c is the local state, f is the performance criterion (e.g. objective function),
     * and \f$\gamma\f$ is the local adjoint vector. The pseudo time is denoted by t,
     * where n denotes the current step index and n+1 is the previous time step index.
     *
     * \param [in] aControls      1D view of control variables, i.e. design variables
     * \param [in] aCurStateVars  C++ structure that holds the current set of state variables
     * \param [in] aPrevStateVars C++ structure that holds the previous set of state variables
     * \param [in] aAdjointVars   C++ structure that holds the current set of adjoint variables
    *******************************************************************************/
    Plato::ScalarVector
    assembleGlobalAdjointRHS(const Plato::ScalarVector & aControls,
                             const Plato::ForwardStates & aCurStateVars,
                             const Plato::ForwardStates & aPrevStateVars,
                             const Plato::AdjointStates & aAdjointVars)
    {
        // Compute partial derivative of objective with respect to current global states
        auto tDfDu = mCriterion->gradient_u(aCurStateVars.mCurrentGlobalState, aCurStateVars.mPreviousGlobalState,
                                            aCurStateVars.mCurrentLocalState, aCurStateVars.mPreviousLocalState,
                                            aControls, aCurStateVars.mCurrentStepIndex);

        // Compute previous adjoint states contribution to global adjoint rhs
        auto tFinalStepIndex = mNumPseudoTimeSteps - static_cast<Plato::OrdinalType>(1);
        if(aCurStateVars.mCurrentStepIndex != tFinalStepIndex)
        {
            // Compute partial derivative of objective with respect to previous global states, i.e. DfDu_{k+1}
            const Plato::Scalar tAlpha = 1.0; const Plato::Scalar tBeta = 1.0;
            auto tDfDup = mCriterion->gradient_up(aPrevStateVars.mCurrentGlobalState, aPrevStateVars.mPreviousGlobalState,
                                                  aPrevStateVars.mCurrentLocalState, aPrevStateVars.mPreviousLocalState,
                                                  aControls, aCurStateVars.mCurrentStepIndex);
            Plato::blas2::update(tAlpha, tDfDup, tBeta, tDfDu);

            // Compute projected pressure gradient contribution to global adjoint rhs, i.e. DpDu_{k+1}^T * gamma_{k+1}
            auto tProjPressGradAdjointRHS = this->computeProjPressGradAdjointRHS(aControls, aPrevStateVars, aAdjointVars);
            Plato::blas2::axpy<mNumGlobalDofsPerNode, mPressureDofOffset>(tAlpha, tProjPressGradAdjointRHS, tDfDu);

            // Compute global residual contribution to global adjoint RHS, i.e. DrDu_{k+1}^T * lambda_{k+1}
            auto tNumCells = mGlobalEquation->numCells();
            Plato::ScalarMultiVector tPrevLambda("Previous Global Adjoint Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aAdjointVars.mPreviousGlobalAdjoint, tPrevLambda);
            auto tDrDup = mGlobalEquation->gradient_up(aPrevStateVars.mCurrentGlobalState, aPrevStateVars.mPreviousGlobalState,
                                                       aPrevStateVars.mCurrentLocalState , aPrevStateVars.mPreviousLocalState,
                                                       aPrevStateVars.mProjectedPressGrad, aControls, aCurStateVars.mCurrentStepIndex);
            Plato::blas2::matrix_times_vector("T", tAlpha, tDrDup, tPrevLambda, tBeta, tDfDu);

            // Compute local residual contribution to global adjoint RHS, i.e. DhDu_{k+1}^T * mu_{k+1}
            Plato::ScalarMultiVector tPrevMu("Previous Local Adjoint Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aAdjointVars.mPreviousLocalAdjoint, tPrevMu);
            auto tDhDup = mLocalEquation->gradient_up(aPrevStateVars.mCurrentGlobalState, aPrevStateVars.mPreviousGlobalState,
                                                      aPrevStateVars.mCurrentLocalState , aPrevStateVars.mPreviousLocalState,
                                                      aControls, aCurStateVars.mCurrentStepIndex);
            Plato::blas2::matrix_times_vector("T", tAlpha, tDhDup, tPrevMu, tBeta, tDfDu);
        }

        // Compute and add local contribution to global adjoint rhs, i.e. tDfDu_k - F_k^{local}
        auto tLocalStateAdjointRHS = this->computeLocalAdjointRHS(aControls, aCurStateVars, aPrevStateVars, aAdjointVars);
        const Plato::Scalar  tAlpha = -1.0; const Plato::Scalar tBeta = 1.0;
        Plato::blas2::update(tAlpha, tLocalStateAdjointRHS, tBeta, tDfDu);

        // Assemble -( DfDu_k + DfDup + (DpDup_T * gamma_{k+1}) - F_k^{local} )
        auto tNumGlobalAdjointVars = mGlobalEquation->size();
        Plato::ScalarVector tGlobalResidual("global adjoint residual", tNumGlobalAdjointVars);
        Plato::blas1::fill(static_cast<Plato::Scalar>(0), tGlobalResidual);
        mWorksetBase.assembleVectorGradientU(tDfDu, tGlobalResidual);
        Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tGlobalResidual);

        return (tGlobalResidual);
    }

    /***************************************************************************//**
     * \brief Apply Dirichlet constraints for adjoint problem
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    *******************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    {
        if(mDirichletDofs.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("Path-Dependent Adjoint Solver: Essential Boundary Conditions are empty.")
        }

        Plato::ScalarVector tDirichletValues("Dirichlet Values", mDirichletDofs.size());
        Plato::blas1::scale(static_cast<Plato::Scalar>(0.0), tDirichletValues);

        if(aMatrix->isBlockMatrix())
        {
            Plato::applyBlockConstraints<mNumGlobalDofsPerNode>(aMatrix, aVector, mDirichletDofs, tDirichletValues);
        }
        else
        {
            Plato::applyConstraints<mNumGlobalDofsPerNode>(aMatrix, aVector, mDirichletDofs, tDirichletValues);
        }
    }

    /***************************************************************************//**
     * \brief Compute the contibution from the partial derivative of partial differential
     *   equation (PDE) with respect to the control degrees of freedom (dofs).  The PDE
     *   contribution to the total gradient with respect to the control dofs is given by:
     *
     *  \f$ \left(\frac{df}{dz}\right)_{t=n} = \left(\frac{\partial{f}}{\partial{z}}\right)_{t=n}
     *         + \left(\frac{\partial{R}}{\partial{z}}\right)_{t=n}^{T}\lambda_{t=n}
     *         + \left(\frac{\partial{H}}{\partial{z}}\right)_{t=n}^{T}\gamma_{t=n}
     *         + \left(\frac{\partial{P}}{\partial{z}}\right)_{t=n}^{T}\mu_{t=n} \f$,
     *
     * where R is the global residual, H is the local residual, P is the projection residual,
     * and \f$\lambda\f$ is the global adjoint vector, \f$\gamma\f$ is the local adjoint vector,
     * and \f$\mu\f$ is the projection adjoint vector. The pseudo time is denoted by t, where n
     * denotes the current step index.
     *
     * \param [in] aControls     1D view of control variables, i.e. design variables
     * \param [in] aStateVars    C++ structure that holds the current set of state variables
     * \param [in] aAdjointVars  C++ structure that holds the current set of adjoint variables
     * \param [in/out] aGradient total derivative wrt controls
    *********************************************************************************/
    void addPDEpartialDerivativeZ(const Plato::ScalarVector &aControls,
                                  const Plato::ForwardStates &aStateVars,
                                  const Plato::AdjointStates &aAdjointVars,
                                  Plato::ScalarVector &aTotalDerivative)
    {
        auto tNumCells = mGlobalEquation->numCells();
        Plato::ScalarMultiVector tGradientControl("Gradient WRT Control", tNumCells, mNumNodesPerCell);

        // add global adjoint contribution to total gradient, i.e. DfDz += (DrDz)^T * lambda
        Plato::ScalarMultiVector tCurrentLambda("Current Global State Adjoint", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aAdjointVars.mCurrentGlobalAdjoint, tCurrentLambda);
        auto tDrDz = mGlobalEquation->gradient_z(aStateVars.mCurrentGlobalState, aStateVars.mPreviousGlobalState,
                                                 aStateVars.mCurrentLocalState, aStateVars.mPreviousLocalState,
                                                 aStateVars.mProjectedPressGrad, aControls, aStateVars.mCurrentStepIndex);
        const Plato::Scalar tAlpha = 1.0; Plato::Scalar tBeta = 0.0;
        Plato::blas2::matrix_times_vector("T", tAlpha, tDrDz, tCurrentLambda, tBeta, tGradientControl);

        // add projected pressure gradient adjoint contribution to total gradient, i.e. DfDz += (DpDz)^T * gamma
        Plato::ScalarMultiVector tCurrentGamma("Current Projected Pressure Gradient Adjoint", tNumCells, mNumPressGradDofsPerCell);
        mWorksetBase.worksetNodeState(aAdjointVars.mProjPressGradAdjoint, tCurrentGamma);
        auto tDpDz = mProjectionEquation->gradient_z_workset(aStateVars.mProjectedPressGrad, aStateVars.mPressure,
                                                             aControls, aStateVars.mCurrentStepIndex);
        tBeta = 1.0;
        Plato::blas2::matrix_times_vector("T", tAlpha, tDpDz, tCurrentGamma, tBeta, tGradientControl);

        // compute local adjoint contribution to total gradient, i.e. (DhDz)^T * mu
        Plato::ScalarMultiVector tCurrentMu("Current Local State Adjoint", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aAdjointVars.mCurrentLocalAdjoint, tCurrentMu);
        auto tDhDz = mLocalEquation->gradient_z(aStateVars.mCurrentGlobalState, aStateVars.mPreviousGlobalState,
                                                aStateVars.mCurrentLocalState, aStateVars.mPreviousLocalState,
                                                aControls, aStateVars.mCurrentStepIndex);
        Plato::blas2::matrix_times_vector("T", tAlpha, tDhDz, tCurrentMu, tBeta, tGradientControl);

        mWorksetBase.assembleScalarGradientZ(tGradientControl, aTotalDerivative);
    }

    /***************************************************************************//**
     * \brief Compute the contibution from the partial derivative of partial differential
     *   equation (PDE) with respect to the configuration degrees of freedom (dofs).  The
     *   PDE contribution to the total gradient with respect to the configuration dofs is
     *   given by:
     *
     *  \f$ \left(\frac{df}{dz}\right)_{t=n} = \left(\frac{\partial{f}}{\partial{x}}\right)_{t=n}
     *         + \left(\frac{\partial{R}}{\partial{x}}\right)_{t=n}^{T}\lambda_{t=n}
     *         + \left(\frac{\partial{H}}{\partial{x}}\right)_{t=n}^{T}\gamma_{t=n}
     *         + \left(\frac{\partial{P}}{\partial{x}}\right)_{t=n}^{T}\mu_{t=n} \f$,
     *
     * where R is the global residual, H is the local residual, P is the projection residual,
     * and \f$\lambda\f$ is the global adjoint vector, \f$\gamma\f$ is the local adjoint vector,
     * x denotes the configuration variables, and \f$\mu\f$ is the projection adjoint vector.
     * The pseudo time is denoted by t, where n denotes the current step index.
     *
     * \param [in] aControls     1D view of control variables, i.e. design variables
     * \param [in] aStateVars    C++ structure that holds the current set of state variables
     * \param [in] aAdjointVars  C++ structure that holds the current set of adjoint variables
     * \param [in/out] aGradient total derivative wrt configuration
    *******************************************************************************/
    void addPDEpartialDerivativeX(const Plato::ScalarVector &aControls,
                                  const Plato::ForwardStates &aStateVars,
                                  const Plato::AdjointStates &aAdjointVars,
                                  Plato::ScalarVector &aTotalDerivative)
    {
        // Allocate return gradient
        auto tNumCells = mGlobalEquation->numCells();
        Plato::ScalarMultiVector tGradientConfiguration("Gradient WRT Configuration", tNumCells, mNumConfigDofsPerCell);

        // add global adjoint contribution to total gradient, i.e. DfDx += (DrDx)^T * lambda
        Plato::ScalarMultiVector tCurrentLambda("Current Global State Adjoint", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aAdjointVars.mCurrentGlobalAdjoint, tCurrentLambda);
        auto tDrDx = mGlobalEquation->gradient_x(aStateVars.mCurrentGlobalState, aStateVars.mPreviousGlobalState,
                                                 aStateVars.mCurrentLocalState, aStateVars.mPreviousLocalState,
                                                 aStateVars.mProjectedPressGrad, aControls, aStateVars.mCurrentStepIndex);
        const Plato::Scalar tAlpha = 1.0; Plato::Scalar tBeta = 0.0;
        Plato::blas2::matrix_times_vector("T", tAlpha, tDrDx, tCurrentLambda, tBeta, tGradientConfiguration);

        // add projected pressure gradient adjoint contribution to total gradient, i.e. DfDx += (DpDx)^T * gamma
        Plato::ScalarMultiVector tCurrentGamma("Current Projected Pressure Gradient Adjoint", tNumCells, mNumPressGradDofsPerCell);
        mWorksetBase.worksetNodeState(aAdjointVars.mProjPressGradAdjoint, tCurrentGamma);
        auto tDpDx = mProjectionEquation->gradient_x_workset(aStateVars.mProjectedPressGrad, aStateVars.mPressure,
                                                             aControls, aStateVars.mCurrentStepIndex);
        tBeta = 1.0;
        Plato::blas2::matrix_times_vector("T", tAlpha, tDpDx, tCurrentGamma, tBeta, tGradientConfiguration);

        // compute local contribution to total gradient, i.e. (DhDx)^T * mu
        Plato::ScalarMultiVector tCurrentMu("Current Local State Adjoint", tNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aAdjointVars.mCurrentLocalAdjoint, tCurrentMu);
        auto tDhDx = mLocalEquation->gradient_x(aStateVars.mCurrentGlobalState, aStateVars.mPreviousGlobalState,
                                                aStateVars.mCurrentLocalState, aStateVars.mPreviousLocalState,
                                                aControls, aStateVars.mCurrentStepIndex);
        Plato::blas2::matrix_times_vector("T", tAlpha, tDhDx, tCurrentMu, tBeta, tGradientConfiguration);

        mWorksetBase.assembleVectorGradientX(tGradientConfiguration, aTotalDerivative);
    }

public:
    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh   mesh database
     * \param [in] aInputs input parameters list
     * \param [in] aLinearSolver linear solver object
    *******************************************************************************/
    PathDependentAdjointSolver(Omega_h::Mesh & aMesh, Teuchos::ParameterList & aInputs, std::shared_ptr<Plato::AbstractSolver> &aLinearSolver) :
        mWorksetBase(aMesh),
        mNumPseudoTimeSteps(Plato::ParseTools::getSubParam<Plato::OrdinalType>(aInputs, "Time Stepping", "Initial Num. Pseudo Time Steps", 20)),
        mLinearSolver(aLinearSolver)
    {}

    /***************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh database
    *******************************************************************************/
    explicit PathDependentAdjointSolver(Omega_h::Mesh & aMesh) :
        mWorksetBase(aMesh),
        mNumPseudoTimeSteps(20),
        mLinearSolver(nullptr)
    {}

    /***************************************************************************//**
     * \brief Destructor
    *******************************************************************************/
    ~PathDependentAdjointSolver(){}

    /***************************************************************************//**
     * \brief Set number of pseudo time steps
     * \param [in] aInput number of pseudo time steps
    *******************************************************************************/
    void setNumPseudoTimeSteps(const Plato::OrdinalType & aInput)
    {
        mNumPseudoTimeSteps = aInput;
    }

    /***************************************************************************//**
     * \brief Append scalar function interface
     * \param [in] aInput scalar function interface
    *******************************************************************************/
    void appendScalarFunction(const std::shared_ptr<Plato::LocalScalarFunctionInc> & aInput)
    {
        mCriterion = aInput;
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
     * \brief Append projection system of equation interface
     * \param [in] aInput projection system of equation interface
    *******************************************************************************/
    void appendProjectionEquation(const std::shared_ptr<Plato::VectorFunctionVMS<ProjectorT>> & aInput)
    {
        mProjectionEquation = aInput;
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
     * \brief Update inverse of local Jacobian with respect to local states, i.e.
     *
     * \f$ \left[ \left( \frac{\partial{H}}{\partial{c}} \right)_{\Delta{t}=n} \right]^{-1}, \f$:
     *
     * where \f$ H \f$ is the local residual and \f$ c \f$ is the local state vector.
     * The pseudo time step is denoted by \f$ \Delta{t} \f$, where \f$ n \f$ denotes
     * the current time step index.
     *
     * \param [in]     aControls    1D view of control variables, i.e. design variables
     * \param [in]     aStates      C++ structure holding the most recent state data
     * \param [in\out] aInvLocalJac inverse of local Jacobian wrt local states
    *******************************************************************************/
    void updateInverseLocalJacobian(const Plato::ScalarVector & aControls,
                                    const Plato::ForwardStates & aStates,
                                    Plato::ScalarArray3D& aInvLocalJac)
    {
        auto tNumCells = mLocalEquation->numCells();
        auto tDhDc = mLocalEquation->gradient_c(aStates.mCurrentGlobalState, aStates.mPreviousGlobalState,
                                                aStates.mCurrentLocalState , aStates.mPreviousLocalState,
                                                aControls, aStates.mCurrentStepIndex);
        Plato::blas3::inverse<mNumLocalDofsPerCell, mNumLocalDofsPerCell>(tNumCells, tDhDc, aInvLocalJac);
    }

    /***************************************************************************//**
     * \brief Update projected pressure gradient adjoint variables, \f$ \gamma_k \f$
     *   as follows:
     *  \f$
     *    \gamma_{k} =
     *      -\left(
     *          \left(\frac{\partial{P}}{\partial{\pi}}\right)_{t=k}^{T}
     *       \right)^{-1}
     *       \left[
     *          \left(\frac{\partial{R}}{\partial{\pi}}\right)_{t=k+1}^{T}\lambda_{k+1}
     *       \right]
     *  \f$,
     * where R is the global residual, P is the projected pressure gradient residual,
     * and \f$\pi\f$ is the projected pressure gradient. The pseudo time step index is
     * denoted by k.
     *
     * \param [in] aControls    1D view of control variables, i.e. design variables
     * \param [in] aStateVars   C++ structure that holds current state variables
     * \param [in] aAdjointVars C++ structure that holds current adjoint variables
    *******************************************************************************/
    void updateProjPressGradAdjointVars(const Plato::ScalarVector & aControls,
                                        const Plato::ForwardStates & aStateVars,
                                        Plato::AdjointStates & aAdjointVars)
    {
        auto tLastStepIndex = mNumPseudoTimeSteps - static_cast<Plato::OrdinalType>(1);
        if(aStateVars.mCurrentStepIndex == tLastStepIndex)
        {
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), aAdjointVars.mProjPressGradAdjoint);
            return;
        }

        // Compute Jacobian tDrDp_{k+1}^T, i.e. transpose of Jacobian with respect to projected pressure gradient
        auto tDrDp_T =
            mGlobalEquation->gradient_n_T_assembled(aStateVars.mCurrentGlobalState, aStateVars.mPreviousGlobalState,
                                                    aStateVars.mCurrentLocalState , aStateVars.mPreviousLocalState,
                                                    aStateVars.mProjectedPressGrad, aControls, aStateVars.mCurrentStepIndex);

        // Compute tDrDp_{k+1}^T * lambda_{k+1}
        auto tNumProjPressGradDofs = mProjectionEquation->size();
        Plato::ScalarVector tResidual("Projected Pressure Gradient Residual", tNumProjPressGradDofs);
        Plato::MatrixTimesVectorPlusVector(tDrDp_T, aAdjointVars.mPreviousGlobalAdjoint, tResidual);
        Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tResidual);

        // Solve for current projected pressure gradient adjoint, i.e.
        //   gamma_k =  INV(tDpDp_k^T) * (tDrDp_{k+1}^T * lambda_{k+1})
        auto tProjJacobian = mProjectionEquation->gradient_u_T(aStateVars.mProjectedPressGrad, aStateVars.mPressure,
                                                               aControls, aStateVars.mCurrentStepIndex);

        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), aAdjointVars.mProjPressGradAdjoint);
        Plato::Solve::RowSummed<PhysicsT::mNumSpatialDims>(tProjJacobian, aAdjointVars.mProjPressGradAdjoint, tResidual);
    }

    /***************************************************************************//**
     * \brief Update current global adjoint variables, i.e. \f$ \lambda_k \f$
     * \param [in]     aControls      1D view of control variables, i.e. design variables
     * \param [in]     aCurStateVars  C++ structure that holds current state variables
     * \param [in]     aPrevStateVars C++ structure that holds previous state variables
     * \param [in\out] aAdjointVars   C++ structure that holds current adjoint variables
    *******************************************************************************/
    void updateGlobalAdjointVars(const Plato::ScalarVector & aControls,
                                 const Plato::ForwardStates & aCurStateVars,
                                 const Plato::ForwardStates & aPrevStateVars,
                                 Plato::AdjointStates & aAdjointVars)
    {
        // Assemble Jacobian
        auto tJacobian = this->assembleTangentMatrix(aControls, aCurStateVars, aAdjointVars.mInvLocalJacT);

        // Assemble residual
        auto tResidual = this->assembleGlobalAdjointRHS(aControls, aCurStateVars, aPrevStateVars, aAdjointVars);

        // Apply Dirichlet conditions for adjoint problem
        this->applyConstraints(tJacobian, tResidual);

        // Solve for lambda_k = (K_{tangent})_k^{-T} * F_k^{adjoint}
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), aAdjointVars.mCurrentGlobalAdjoint);
        //Plato::Solve::Consistent<mNumGlobalDofsPerNode>(tJacobian, aAdjointVars.mCurrentGlobalAdjoint, tResidual, false);
        if (mLinearSolver == nullptr)
            THROWERR("Linear solver object not initialized.")
        mLinearSolver->solve(*tJacobian, aAdjointVars.mCurrentGlobalAdjoint, tResidual);
    }

    /***************************************************************************//**
     * \brief Update local adjoint vector using the following equation:
     *
     *  \f$ \mu_k =
     *      -\left(
     *          \left(\frac{\partial{H}}{\partial{c}}\right)_{t=k}^{T}
     *       \right)^{-1}
     *       \left[
     *           \left( \frac{\partial{R}}{\partial{c}} \right)_{t=k}^{T}\lambda_k +
     *           \left( \frac{\partial{f}}{\partial{c}}_{k} \right)
     *                + \left( \frac{\partial{H}}{\partial{c}} \right)_{t=k+1}^{T} \mu_{k+1}
     *       \right]
     *  \f$,
     *
     * where R is the global residual, H is the local residual, u is the global state,
     * c is the local state, f is the performance criterion (e.g. objective function),
     * and \f$\gamma\f$ is the local adjoint vector. The pseudo time is denoted by t,
     * where n denotes the current step index and n+1 is the previous time step index.
     *
     * \param [in]     aControls      1D view of control variables, i.e. design variables
     * \param [in]     aCurStateVars  C++ structure that holds current state variables
     * \param [in]     aPrevStateVars C++ structure that holds previous state variables
     * \param [in/out] aAdjointVars   C++ structure that holds current adjoint variables
    *******************************************************************************/
    void updateLocalAdjointVars(const Plato::ScalarVector & aControls,
                                const Plato::ForwardStates& aCurStateVars,
                                const Plato::ForwardStates& aPrevStateVars,
                                Plato::AdjointStates & aAdjointVars)
    {
        // Compute DfDc_{k}
        auto tDfDc = mCriterion->gradient_c(aCurStateVars.mCurrentGlobalState, aCurStateVars.mPreviousGlobalState,
                                            aCurStateVars.mCurrentLocalState, aCurStateVars.mPreviousLocalState,
                                            aControls, aCurStateVars.mCurrentStepIndex);

        // Compute DfDc_k + ( DrDc_k^T * lambda_k )
        auto tNumCells = mLocalEquation->numCells();
        Plato::ScalarMultiVector tCurrentLambda("Current Global Adjoint Workset", tNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aAdjointVars.mCurrentGlobalAdjoint, tCurrentLambda);
        auto tDrDc = mGlobalEquation->gradient_c(aCurStateVars.mCurrentGlobalState, aCurStateVars.mPreviousGlobalState,
                                                 aCurStateVars.mCurrentLocalState , aCurStateVars.mPreviousLocalState,
                                                 aCurStateVars.mProjectedPressGrad, aControls, aCurStateVars.mCurrentStepIndex);
        Plato::Scalar tAlpha = 1.0; Plato::Scalar tBeta = 1.0;
        Plato::blas2::matrix_times_vector("T", tAlpha, tDrDc, tCurrentLambda, tBeta, tDfDc);

        auto tFinalStepIndex = mNumPseudoTimeSteps - static_cast<Plato::OrdinalType>(1);
        if(aCurStateVars.mCurrentStepIndex != tFinalStepIndex)
        {
            // Compute DfDc_k + ( DrDc_k^T * lambda_k ) + DfDc_{k+1}
            const Plato::Scalar tAlpha = 1.0; const Plato::Scalar tBeta = 1.0;
            auto tDfDcp = mCriterion->gradient_cp(aPrevStateVars.mCurrentGlobalState, aPrevStateVars.mPreviousGlobalState,
                                                  aPrevStateVars.mCurrentLocalState, aPrevStateVars.mPreviousLocalState,
                                                  aControls, aCurStateVars.mCurrentStepIndex);
            Plato::blas2::update(tAlpha, tDfDcp, tBeta, tDfDc);

            // Compute DfDc_k + ( DrDc_k^T * lambda_k ) + DfDc_{k+1} + ( DhDc_{k+1}^T * mu_{k+1} )
            Plato::ScalarMultiVector tPreviousMu("Previous Local Adjoint Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aAdjointVars.mPreviousLocalAdjoint, tPreviousMu);
            auto tDhDcp = mLocalEquation->gradient_cp(aPrevStateVars.mCurrentGlobalState, aPrevStateVars.mPreviousGlobalState,
                                                      aPrevStateVars.mCurrentLocalState , aPrevStateVars.mPreviousLocalState,
                                                      aControls, aCurStateVars.mCurrentStepIndex);
            Plato::blas2::matrix_times_vector("T", tAlpha, tDhDcp, tPreviousMu, tBeta, tDfDc);

            // Compute RHS_{local} = DfDc_k + ( DrDc_k^T * lambda_k ) + DfDc_{k+1} + ( DhDc_{k+1}^T * mu_{k+1} ) + ( DrDc_{k+1}^T * lambda_{k+1} )
            Plato::ScalarMultiVector tPrevLambda("Previous Global Adjoint Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aAdjointVars.mPreviousGlobalAdjoint, tPrevLambda);
            auto tDrDcp = mGlobalEquation->gradient_cp(aPrevStateVars.mCurrentGlobalState, aPrevStateVars.mPreviousGlobalState,
                                                       aPrevStateVars.mCurrentLocalState , aPrevStateVars.mPreviousLocalState,
                                                       aCurStateVars.mProjectedPressGrad, aControls, aCurStateVars.mCurrentStepIndex);
            Plato::blas2::matrix_times_vector("T", tAlpha, tDrDcp, tPrevLambda, tBeta, tDfDc);
        }

        // Solve for current local adjoint variables, i.e. mu_k = -Inv(tDhDc_k^T) * RHS_{local}
        tAlpha = -1.0; tBeta = 0.0;
        Plato::ScalarMultiVector tCurrentMu("Current Local Adjoint Workset", tNumCells, mNumLocalDofsPerCell);
        Plato::blas2::matrix_times_vector("T", tAlpha, aAdjointVars.mInvLocalJacT, tDfDc, tBeta, tCurrentMu);
        Plato::flatten_vector_workset<mNumLocalDofsPerCell>(tNumCells, tCurrentMu, aAdjointVars.mCurrentLocalAdjoint);
    }

    /***************************************************************************//**
     * \brief Update path-dependent adjoint variables.
     * \param [in]     aControls          1D view of current control variables, i.e. design variables
     * \param [in]     aCurrentStateVars  C++ structure that holds current state variables
     * \param [in]     aPreviousStateVars C++ structure that holds previous state variables
     * \param [in/out] aAdjointVars       C++ structure that holds current adjoint variables
    *******************************************************************************/
    void updateAdjointVariables(const Plato::ScalarVector & aControls,
                                const Plato::ForwardStates & aCurrentStateVars,
                                const Plato::ForwardStates & aPreviousStateVars,
                                Plato::AdjointStates & aAdjointVars)
    {
        this->updateInverseLocalJacobian(aControls, aCurrentStateVars, aAdjointVars.mInvLocalJacT);
        this->updateProjPressGradAdjointVars(aControls, aCurrentStateVars, aAdjointVars);
        this->updateGlobalAdjointVars(aControls, aCurrentStateVars, aPreviousStateVars, aAdjointVars);
        this->updateLocalAdjointVars(aControls, aCurrentStateVars, aPreviousStateVars, aAdjointVars);
    }

    /***************************************************************************//**
     * \brief Add contribution from partial differential equation to total derivative.
     * \param [in]     aControls    1D view of control variables, i.e. design variables
     * \param [in]     aStateVars   C++ structure that holds current state variables
     * \param [in]     aAdjointVars C++ structure that holds current adjoint variables
     * \param [in/out] aOutput      total derivative
    *******************************************************************************/
    void addContributionFromPDE(const Plato::ScalarVector &aControls,
                                const Plato::ForwardStates &aStateVars,
                                const Plato::AdjointStates &aAdjointVars,
                                Plato::ScalarVector &aOutput)
    {
        switch(aStateVars.mPartialDerivativeType)
        {
            case Plato::PartialDerivative::CONTROL:
            {
                this->addPDEpartialDerivativeZ(aControls, aStateVars, aAdjointVars, aOutput);
                break;
            }
            case Plato::PartialDerivative::CONFIGURATION:
            {
                this->addPDEpartialDerivativeX(aControls, aStateVars, aAdjointVars, aOutput);
                break;
            }
            default:
            {
                PRINTERR("PARTIAL DERIVATIVE IS NOT DEFINED. OPTIONS ARE CONTROL AND CONFIGURATION")
            }
        }
    }
};
// class PathDependentAdjointSolver

}
// namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainPlasticity<1>>;
extern template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainThermoPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainPlasticity<2>>;
extern template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainThermoPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainPlasticity<3>>;
extern template class Plato::PathDependentAdjointSolver<Plato::InfinitesimalStrainThermoPlasticity<3>>;
#endif

#pragma once

#include "PlatoUtilities.hpp"

#include <memory>
#include <sstream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "BLAS1.hpp"
#include "Solutions.hpp"
#include "NaturalBCs.hpp"
#include "EssentialBCs.hpp"
#include "AnalyzeOutput.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyConstraints.hpp"
#include "SpatialModel.hpp"
#include "PlatoSequence.hpp"
#include "ToMap.hpp"

#include "ParseTools.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoAbstractProblem.hpp"

#include "Geometrical.hpp"
#include "geometric/ScalarFunctionBase.hpp"
#include "geometric/ScalarFunctionBaseFactory.hpp"

#include "elliptic/updated_lagrangian/VectorFunction.hpp"
#include "elliptic/updated_lagrangian/LagrangianUpdate.hpp"
#include "elliptic/updated_lagrangian/ScalarFunctionBaseFactory.hpp"
#include "AnalyzeMacros.hpp"

#include "alg/ParallelComm.hpp"
#include "alg/PlatoSolverFactory.hpp"

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename PhysicsT>
class Problem: public Plato::AbstractProblem
{
private:

    using Criterion       = std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::ScalarFunctionBase>;
    using Criteria        = std::map<std::string, Criterion>;

    using LinearCriterion = std::shared_ptr<Plato::Geometric::ScalarFunctionBase>;
    using LinearCriteria  = std::map<std::string, LinearCriterion>;

    static constexpr Plato::OrdinalType SpatialDim = PhysicsT::mNumSpatialDims;
    static constexpr Plato::OrdinalType NumVoigtTerms = PhysicsT::mNumVoigtTerms;

    using VectorFunctionType = Plato::Elliptic::UpdatedLagrangian::VectorFunction<PhysicsT>;

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    Plato::Sequence<SpatialDim> mSequence;

    std::shared_ptr<VectorFunctionType> mPDE; /*!< equality constraint interface */

    LinearCriteria mLinearCriteria;
    Criteria       mCriteria;

    Plato::OrdinalType mNumNewtonSteps;
    Plato::Scalar      mNewtonResTol, mNewtonIncTol;

    bool mSaveState;

    Plato::ScalarMultiVector mGlobalAdjoints;
    Plato::ScalarMultiVector mLocalAdjoints;
    Plato::ScalarVector mResidual;

    Plato::ScalarMultiVector mGlobalStates;
    Plato::ScalarMultiVector mTotalStates;
    Plato::ScalarMultiVector mLocalStates;

    bool mIsSelfAdjoint; /*!< indicates if problem is self-adjoint */

    Teuchos::RCP<Plato::CrsMatrixType> mGlobalJacobian; /*!< Global jacobian matrix */
    Teuchos::RCP<Plato::CrsMatrixType> mLocalJacobian; /*!< Global jacobian matrix */

    Plato::LocalOrdinalVector mBcDofs; /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */
    Plato::ScalarVector mBcValues; /*!< values associated with the Dirichlet boundary conditions */

    rcp<Plato::AbstractSolver> mSolver;

    Plato::LagrangianUpdate<PhysicsT> mLagrangianUpdate;

    std::string mPDEType; /*!< partial differential equation type */
    std::string mPhysics; /*!< physics used for the simulation */

public:
    /******************************************************************************//**
     * \brief PLATO problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    Problem(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Teuchos::ParameterList& aProblemParams,
      Comm::Machine aMachine
    ) :
      mSpatialModel  (aMesh, aMeshSets, aProblemParams),
      mSequence      (mSpatialModel, aProblemParams),
      mPDE(std::make_shared<VectorFunctionType>(mSpatialModel, mDataMap, aProblemParams, aProblemParams.get<std::string>("PDE Constraint"))),
      mNumNewtonSteps(Plato::ParseTools::getSubParam<int>   (aProblemParams, "Newton Iteration", "Maximum Iterations",  1  )),
      mNewtonIncTol  (Plato::ParseTools::getSubParam<double>(aProblemParams, "Newton Iteration", "Increment Tolerance", 0.0)),
      mNewtonResTol  (Plato::ParseTools::getSubParam<double>(aProblemParams, "Newton Iteration", "Residual Tolerance",  0.0)),
      mSaveState     (aProblemParams.sublist("Updated Lagrangian Elliptic").isType<Teuchos::Array<std::string>>("Plottable")),
      mResidual      ("MyResidual", mPDE->size()),
      mGlobalStates  ("Global states", mSequence.getNumSteps(), mPDE->size()),
      mTotalStates   ("Total states",  mSequence.getNumSteps(), mPDE->size()),
      mLocalStates   ("Local states",  mSequence.getNumSteps(), mPDE->stateSize()),
      mGlobalJacobian(Teuchos::null),
      mIsSelfAdjoint (aProblemParams.get<bool>("Self-Adjoint", false)),
      mLagrangianUpdate(mSpatialModel),
      mPDEType       (aProblemParams.get<std::string>("PDE Constraint")),
      mPhysics       (aProblemParams.get<std::string>("Physics"))
    {
        this->initialize(aProblemParams);

        Plato::SolverFactory tSolverFactory(aProblemParams.sublist("Linear Solver"));
        mSolver = tSolverFactory.create(aMesh, aMachine, PhysicsT::mNumDofsPerNode);
    }

    ~Problem(){}

    Plato::OrdinalType numNodes() const
    {
        const auto tNumNodes = mPDE->numNodes();
        return (tNumNodes);
    }

    Plato::OrdinalType numCells() const
    {
        const auto tNumCells = mPDE->numCells();
        return (tNumCells);
    }
    
    Plato::OrdinalType numDofsPerCell() const
    {
        const auto tNumDofsPerCell = mPDE->numDofsPerCell();
        return (tNumDofsPerCell);
    }

    Plato::OrdinalType numNodesPerCell() const
    {
        const auto tNumNodesPerCell = mPDE->numNodesPerCell();
        return (tNumNodesPerCell);
    }

    Plato::OrdinalType numDofsPerNode() const
    {
        const auto tNumDofsPerNode = mPDE->numDofsPerNode();
        return (tNumDofsPerNode);
    }

    Plato::OrdinalType numControlsPerNode() const
    {
        const auto tNumControlsPerNode = mPDE->numControlsPerNode();
        return (tNumControlsPerNode);
    }

    /******************************************************************************//**
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    **********************************************************************************/
    void applyStateConstraints(
      const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
      const Plato::ScalarVector & aVector,
            Plato::Scalar aScale
    )
    {
        if(mGlobalJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<PhysicsT::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues, aScale);
        }
        else
        {
            Plato::applyConstraints<PhysicsT::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues, aScale);
        }
    }
    
    /******************************************************************************//**
     * \brief Output solution to visualization file.
     * \param [in] aFilepath output/visualizaton file path
    **********************************************************************************/
    void output(const std::string & aFilepath) override
    {
        auto tDataMap = this->getDataMap();
        auto tSolution = this->getSolution();
        Plato::output<SpatialDim>(aFilepath, tSolution, tDataMap, mSpatialModel.Mesh);
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aGlobalState 2D container of state variables
     * \param [in] aControl 1D container of control variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aSolution)
    {
        auto tState = aSolution.get("State");
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(tState, tTIME_STEP_INDEX, Kokkos::ALL());

        for( auto tCriterion : mCriteria )
        {
            tCriterion.second->updateProblem(tStatesSubView, aControl);
        }
        for( auto tCriterion : mLinearCriteria )
        {
            tCriterion.second->updateProblem(aControl);
        }
    }

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return Plato::Solution composed of state variables    
     ***********************************************************************************/
    Plato::Solutions
    solution
    (const Plato::ScalarVector & aControl)
    {
        mDataMap.clearAll();

        auto& tSequenceSteps = mSequence.getSteps();
        auto tNumSequenceSteps = tSequenceSteps.size(); 

        for (Plato::OrdinalType tStepIndex=0; tStepIndex<tNumSequenceSteps; tStepIndex++)
        {
            Plato::ScalarVector tGlobalState = Kokkos::subview(mGlobalStates, tStepIndex, Kokkos::ALL());
            Plato::ScalarVector tTotalState  = Kokkos::subview(mTotalStates, tStepIndex, Kokkos::ALL());

            const auto& tSequenceStep = tSequenceSteps[tStepIndex];

            mSpatialModel.applyMask(tSequenceStep.getMask());

            Plato::ScalarVector tLocalState;
            if (tStepIndex > 0)
            {
                tLocalState  = Kokkos::subview(mLocalStates, tStepIndex-1, Kokkos::ALL());

                // copy forward the previous total state
                Kokkos::deep_copy(tTotalState, Kokkos::subview(mTotalStates, tStepIndex-1, Kokkos::ALL()));
            }
            else
            {
                // kokkos initializes new views to zero.
                tLocalState  = Plato::ScalarVector("initial local state",  mPDE->stateSize());
            }

            // inner loop for non-linear models
            for(Plato::OrdinalType tNewtonIndex = 0; tNewtonIndex < mNumNewtonSteps; tNewtonIndex++)
            {
                mResidual = mPDE->value(tGlobalState, tLocalState, aControl);
                Plato::blas1::scale(-1.0, mResidual);

                if (mNumNewtonSteps > 1) {
                    auto tResidualNorm = Plato::blas1::norm(mResidual);
                    std::cout << " Residual norm: " << tResidualNorm << std::endl;
                    if (tResidualNorm < mNewtonResTol) {
                        std::cout << " Residual norm tolerance satisfied." << std::endl;
                        break;
                    }
                }

                mGlobalJacobian = mPDE->gradient_u(tGlobalState, tLocalState, aControl);

                Plato::OrdinalType tScale = (tNewtonIndex == 0) ? 1.0 : 0.0;
                this->applyStateConstraints(mGlobalJacobian, mResidual, tScale);

                Plato::ScalarVector tDeltaD("increment", tGlobalState.extent(0));
                Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaD);

                tSequenceStep.template constrainInactiveNodes<PhysicsT::mNumDofsPerNode>(mGlobalJacobian, mResidual);

                mSolver->solve(*mGlobalJacobian, tDeltaD, mResidual);
                Plato::blas1::axpy(1.0, tDeltaD, tGlobalState);

                if (mNumNewtonSteps > 1) {
                    auto tIncrementNorm = Plato::blas1::norm(tDeltaD);
                    std::cout << " Delta norm: " << tIncrementNorm << std::endl;
                    if (tIncrementNorm < mNewtonIncTol) {
                        std::cout << " Solution increment norm tolerance satisfied." << std::endl;
                        break;
                    }
                }
            }

            // compute residual at end state
            mResidual  = mPDE->value(tGlobalState, tLocalState, aControl);

            Plato::ScalarVector tUpdatedLocalState = Kokkos::subview(mLocalStates, tStepIndex, Kokkos::ALL());
            mDataMap.scalarMultiVectors["total strain"] = Plato::ScalarMultiVector("total strain", numCells(), NumVoigtTerms);
            mLagrangianUpdate(mDataMap, tLocalState, tUpdatedLocalState);

            Plato::blas1::axpy(1.0, tGlobalState, tTotalState);
            mDataMap.vectorNodeFields["total displacement"] = tTotalState;
            mDataMap.scalarNodeFields["topology"] = aControl;
            Plato::toMap(mDataMap.scalarNodeFields, tSequenceStep.getMask()->nodeMask(), "node_mask");

            mDataMap.saveState();

        } // end sequence loop

        auto tSolution = this->getSolution();
        return tSolution;
    }

    /******************************************************************************//**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return criterion function value
    **********************************************************************************/
    Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override
    {
        if( mCriteria.count(aName) )
        {
            Plato::Solutions tSolutions;
            tSolutions.set("State", mGlobalStates);
            Criterion tCriterion = mCriteria[aName];
            return tCriterion->value(tSolutions, mLocalStates, aControl);
        }
        else
        if( mLinearCriteria.count(aName) )
        {
            LinearCriterion tCriterion = mLinearCriteria[aName];
            return tCriterion->value(aControl);
        }
        else
        {
            THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
    }

    /******************************************************************************//**
     * \brief Evaluate criterion function
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aName Name of criterion.
     * \return criterion function value
    **********************************************************************************/
    Plato::Scalar
    criterionValue(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {

// TODO input argument 'aSolution' and member datum 'mLocalStates' must be consistent
// (i.e., computed from a single call to solution()), so taking 'aSolution' as an 
// argument and not 'mLocalStates' is inviting abuse and adds nothing to the utility
// of this function. Should this function be removed?

        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];
            return tCriterion->value(aSolution, mLocalStates, aControl);
        }
        else
        if( mLinearCriteria.count(aName) )
        {
            LinearCriterion tCriterion = mLinearCriteria[aName];
            return tCriterion->value(aControl);
        }
        else
        {
            THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
    }

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution Plato::Solution containing state
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];
            return criterionGradient(aControl, aSolution, tCriterion);
        }
        else
        if( mLinearCriteria.count(aName) )
        {
            LinearCriterion tCriterion = mLinearCriteria[aName];
            return tCriterion->gradient_z(aControl);
        }
        else
        {
            THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
    }


    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \param [in] aCriterion criterion to be evaluated
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
              Criterion             aCriterion
    )
    {

// TODO input argument 'aSolution' and member datum 'mLocalStates' must be consistent
// (i.e., computed from a single call to solution()), so taking 'aSolution' as an 
// argument and not 'mLocalStates' is inviting abuse and adds nothing to the utility
// of this function. Should this function be removed?


        if(aCriterion == nullptr)
        {
            THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }

        // F_{,z}
        auto t_dFdz = aCriterion->gradient_z(aSolution, mLocalStates, aControl);

        auto& tSequenceSteps = mSequence.getSteps();
        auto tLastStepIndex  = tSequenceSteps.size()-1; 
        for (Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex >= 0; tStepIndex--)
        {

            const auto& tSequenceStep = tSequenceSteps[tStepIndex];

            mSpatialModel.applyMask(tSequenceStep.getMask());

            auto tU = Kokkos::subview(mGlobalStates, tStepIndex, Kokkos::ALL());
            auto tC = Kokkos::subview(mLocalStates, tStepIndex, Kokkos::ALL());

            Plato::ScalarVector tAdjoint_U = Kokkos::subview(mGlobalAdjoints, tStepIndex, Kokkos::ALL());
            Plato::ScalarVector tAdjoint_C = Kokkos::subview(mLocalAdjoints, tStepIndex, Kokkos::ALL());


            // F_{,c^k}
            auto t_dFdc = aCriterion->gradient_c(aSolution, mLocalStates, aControl, tStepIndex);
            // F_{,u^k}
            auto t_dFdu = aCriterion->gradient_u(aSolution, mLocalStates, aControl, tStepIndex);

            if(tStepIndex != tLastStepIndex)
            {

                auto tU_next = Kokkos::subview(mGlobalStates, tStepIndex+1, Kokkos::ALL());

                // \lambda^{k+1}
                Plato::ScalarVector tAdjoint_U_next = Kokkos::subview(mGlobalAdjoints, tStepIndex+1, Kokkos::ALL());

                // R_{,c^{k-1}}^{k+1}
                mLocalJacobian = mPDE->gradient_cp_T(tU_next, tC, aControl);

                // f_{,c^k} += \R_{,c^{k-1}}^{k+1} \lambda^{k+1}
                Plato::MatrixTimesVectorPlusVector(mLocalJacobian, tAdjoint_U_next, t_dFdc);

                // \mu^{k+1}
                Plato::ScalarVector tAdjoint_C_next = Kokkos::subview(mLocalAdjoints, tStepIndex+1, Kokkos::ALL());

                // f_{,c^k} -= \mu^{k+1}
                Plato::blas1::axpy(-1.0, tAdjoint_C_next, t_dFdc);
            }
            Plato::blas1::scale(-1.0, t_dFdc);
            Kokkos::deep_copy(tAdjoint_C, t_dFdc);

            Plato::ScalarVector tC_prev;
            if (tStepIndex > 0)
            {
                tC_prev = Kokkos::subview(mLocalStates, tStepIndex-1, Kokkos::ALL());
            }
            else
            {
                // kokkos initializes new views to zero.
                tC_prev = Plato::ScalarVector("initial local state",  mPDE->stateSize());
            }

            // H_{,u^k}^{k}
            auto t_dHdu = mLagrangianUpdate.gradient_u_T(tU, tC, tC_prev);

            // f_{,u^k} += \H_{,u^k}^{k}^T \mu^{k}
            Plato::MatrixTimesVectorPlusVector(t_dHdu, tAdjoint_C, t_dFdu);

            Plato::blas1::scale(-1.0, t_dFdu);

            // R_{,u^k}^T
            mGlobalJacobian = mPDE->gradient_u_T(tU, tC_prev, aControl);

            this->applyAdjointConstraints(mGlobalJacobian, t_dFdu);

            tSequenceStep.template constrainInactiveNodes<PhysicsT::mNumDofsPerNode>(mGlobalJacobian, t_dFdu);

            mSolver->solve(*mGlobalJacobian, tAdjoint_U, t_dFdu);

            // R_{,z}^T
            // dRdz is returned transposed, nxm.  n=z.size() and m=u.size().
            auto t_dRdz = mPDE->gradient_z(tU, tC_prev, aControl);

            // F_{,z} += \lambda^k R_{,z}^k
            Plato::MatrixTimesVectorPlusVector(t_dRdz, tAdjoint_U, t_dFdz);
        }
        return t_dFdz;
    }

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution Plato::Solution containing state
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
        const std::string         & aName
    ) override
    {
        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];
            return criterionGradientX(aControl, aSolution, tCriterion);
        }
        else
        if( mLinearCriteria.count(aName) )
        {
            LinearCriterion tCriterion = mLinearCriteria[aName];
            return tCriterion->gradient_x(aControl);
        }
        else
        {
            THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
    }


    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \param [in] aCriterion criterion to be evaluated
     * \return 1D view - criterion gradient wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const Plato::Solutions    & aSolution,
              Criterion             aCriterion)
    {
        if(aCriterion == nullptr)
        {
            THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }

        // F_{,z}
        auto t_dFdx = aCriterion->gradient_x(aSolution, mLocalStates, aControl);

        auto& tSequenceSteps = mSequence.getSteps();
        auto tLastStepIndex  = tSequenceSteps.size()-1; 
        for (Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex >= 0; tStepIndex--)
        {

            const auto& tSequenceStep = tSequenceSteps[tStepIndex];

            mSpatialModel.applyMask(tSequenceStep.getMask());

            auto tU = Kokkos::subview(mGlobalStates, tStepIndex, Kokkos::ALL());
            auto tC = Kokkos::subview(mLocalStates, tStepIndex, Kokkos::ALL());

            Plato::ScalarVector tAdjoint_U = Kokkos::subview(mGlobalAdjoints, tStepIndex, Kokkos::ALL());
            Plato::ScalarVector tAdjoint_C = Kokkos::subview(mLocalAdjoints, tStepIndex, Kokkos::ALL());


            // F_{,c^k}
            auto t_dFdc = aCriterion->gradient_c(aSolution, mLocalStates, aControl, tStepIndex);
            // F_{,u^k}
            auto t_dFdu = aCriterion->gradient_u(aSolution, mLocalStates, aControl, tStepIndex);

            if(tStepIndex != tLastStepIndex)
            {

                auto tU_next = Kokkos::subview(mGlobalStates, tStepIndex+1, Kokkos::ALL());

                // \lambda^{k+1}
                Plato::ScalarVector tAdjoint_U_next = Kokkos::subview(mGlobalAdjoints, tStepIndex+1, Kokkos::ALL());

                // R_{,c^{k-1}}^{k+1}
                mLocalJacobian = mPDE->gradient_cp_T(tU_next, tC, aControl);

                // f_{,c^k} += \R_{,c^{k-1}}^{k+1} \lambda^{k+1}
                Plato::MatrixTimesVectorPlusVector(mLocalJacobian, tAdjoint_U_next, t_dFdc);

                // \mu^{k+1}
                Plato::ScalarVector tAdjoint_C_next = Kokkos::subview(mLocalAdjoints, tStepIndex+1, Kokkos::ALL());

                // f_{,c^k} -= \mu^{k+1}
                Plato::blas1::axpy(-1.0, tAdjoint_C_next, t_dFdc);
            }
            Plato::blas1::scale(-1.0, t_dFdc);
            Kokkos::deep_copy(tAdjoint_C, t_dFdc);

            Plato::ScalarVector tC_prev;
            if (tStepIndex > 0)
            {
                tC_prev = Kokkos::subview(mLocalStates, tStepIndex-1, Kokkos::ALL());
            }
            else
            {
                // kokkos initializes new views to zero.
                tC_prev = Plato::ScalarVector("initial local state",  mPDE->stateSize());
            }

            // H_{,u^k}^{k}
            auto t_dHdu = mLagrangianUpdate.gradient_u_T(tU, tC, tC_prev);

            // f_{,u^k} += \H_{,u^k}^{k}^T \mu^{k}
            Plato::MatrixTimesVectorPlusVector(t_dHdu, tAdjoint_C, t_dFdu);

            Plato::blas1::scale(-1.0, t_dFdu);

            // R_{,u^k}^T
            mGlobalJacobian = mPDE->gradient_u_T(tU, tC_prev, aControl);

            this->applyAdjointConstraints(mGlobalJacobian, t_dFdu);

            tSequenceStep.template constrainInactiveNodes<PhysicsT::mNumDofsPerNode>(mGlobalJacobian, t_dFdu);

            mSolver->solve(*mGlobalJacobian, tAdjoint_U, t_dFdu);

            // R_{,x}^T
            // dRdx is returned transposed, nxm.  n=x.size() and m=u.size().
            auto t_dRdx = mPDE->gradient_x(tU, tC_prev, aControl);

            // F_{,x} += \lambda^k R_{,x}^k
            Plato::MatrixTimesVectorPlusVector(t_dRdx, tAdjoint_U, t_dFdx);

            // H_{,x}^T
            // dHdx is returned transposed, nxm.  n=x.size() and m=c.size().
            auto t_dHdx = mLagrangianUpdate.gradient_x(tU, tC, tC_prev);

            // F_{,x} += \mu^k R_{,x}^k
            Plato::MatrixTimesVectorPlusVector(t_dHdx, tAdjoint_C, t_dFdx);
        }
        return t_dFdx;
    }

    /******************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradient(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override
    {
        if( mCriteria.count(aName) )
        {
            Plato::Solutions tSolution;
            tSolution.set("State", mGlobalStates);
            Criterion tCriterion = mCriteria[aName];
            return criterionGradient(aControl, tSolution, tCriterion);
        }
        else
        if( mLinearCriteria.count(aName) )
        {
            LinearCriterion tCriterion = mLinearCriteria[aName];
            return tCriterion->gradient_z(aControl);
        }
        else
        {
            THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
    }

    /******************************************************************************//**
     * \brief Evaluate criterion partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aName Name of criterion.
     * \return 1D view - criterion partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector
    criterionGradientX(
        const Plato::ScalarVector & aControl,
        const std::string         & aName
    ) override
    {
        if( mCriteria.count(aName) )
        {
            Plato::Solutions Solutions;
            Solutions.set("State", mGlobalStates);
            Criterion tCriterion = mCriteria[aName];
            return criterionGradientX(aControl, Solutions, tCriterion);
        }
        else
        if( mLinearCriteria.count(aName) )
        {
            LinearCriterion tCriterion = mLinearCriteria[aName];
            return tCriterion->gradient_x(aControl);
        }
        else
        {
            THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
    }

    /***************************************************************************//**
     * \brief Read essential (Dirichlet) boundary conditions from the Exodus file.
     * \param [in] aProblemParams input parameters database
    *******************************************************************************/
    void readEssentialBoundaryConditions(Teuchos::ParameterList& aProblemParams)
    {
        if(aProblemParams.isSublist("Essential Boundary Conditions") == false)
        {
            THROWERR("ESSENTIAL BOUNDARY CONDITIONS SUBLIST IS NOT DEFINED IN THE INPUT FILE.")
        }
        Plato::EssentialBCs<PhysicsT>
        tEssentialBoundaryConditions(aProblemParams.sublist("Essential Boundary Conditions", false), mSpatialModel.MeshSets);
        tEssentialBoundaryConditions.get(mBcDofs, mBcValues);
    }

    /***************************************************************************//**
     * \brief Set essential (Dirichlet) boundary conditions
     * \param [in] aDofs   degrees of freedom associated with Dirichlet boundary conditions
     * \param [in] aValues values associated with Dirichlet degrees of freedom
    *******************************************************************************/
    void setEssentialBoundaryConditions(const Plato::LocalOrdinalVector & aDofs, const Plato::ScalarVector & aValues)
    {
        if(aDofs.size() != aValues.size())
        {
            std::ostringstream tError;
            tError << "DIMENSION MISMATCH: THE NUMBER OF ELEMENTS IN INPUT DOFS AND VALUES ARRAY DO NOT MATCH."
                << "DOFS SIZE = " << aDofs.size() << " AND VALUES SIZE = " << aValues.size();
            THROWERR(tError.str())
        }
        mBcDofs = aDofs;
        mBcValues = aValues;
    }

private:
    /******************************************************************************//**
     * \brief Initialize member data
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList& aProblemParams)
    {

        readEssentialBoundaryConditions(aProblemParams);

        if(aProblemParams.isSublist("Criteria"))
        {
            Plato::Geometric::ScalarFunctionBaseFactory<Plato::Geometrical<SpatialDim>> tLinearFunctionBaseFactory;
            Plato::Elliptic::UpdatedLagrangian::ScalarFunctionBaseFactory<PhysicsT> tNonlinearFunctionBaseFactory;

            auto tCriteriaParams = aProblemParams.sublist("Criteria");
            for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); tIndex != tCriteriaParams.end(); ++tIndex)
            {
                const Teuchos::ParameterEntry & tEntry = tCriteriaParams.entry(tIndex);
                std::string tName = tCriteriaParams.name(tIndex);

                TEUCHOS_TEST_FOR_EXCEPTION(!tEntry.isList(), std::logic_error,
                  " Parameter in Criteria block not valid.  Expect lists only.");

                if( tCriteriaParams.sublist(tName).get<bool>("Linear", false) == true )
                {
                    auto tCriterion = tLinearFunctionBaseFactory.create(mSpatialModel, mDataMap, aProblemParams, tName);
                    if( tCriterion != nullptr )
                    {
                        mLinearCriteria[tName] = tCriterion;
                    }
                }
                else
                {
                    auto tCriterion = tNonlinearFunctionBaseFactory.create(mSpatialModel, mSequence, mDataMap, aProblemParams, tName);
                    if( tCriterion != nullptr )
                    {
                        mCriteria[tName] = tCriterion;
                    }
                }
            }
            if( mCriteria.size() )
            {
                mGlobalAdjoints = Plato::ScalarMultiVector("Global Adjoint Variables", mSequence.getNumSteps(), mPDE->size());
                mLocalAdjoints = Plato::ScalarMultiVector("Local Adjoint Variables", mSequence.getNumSteps(), mPDE->stateSize());
            }
        }
    }

    void applyAdjointConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    {
        Plato::ScalarVector tDirichletValues("Dirichlet Values For Adjoint Problem", mBcValues.size());
        Plato::blas1::scale(static_cast<Plato::Scalar>(0.0), tDirichletValues);
        if(mGlobalJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<PhysicsT::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tDirichletValues);
        }
        else
        {
            Plato::applyConstraints<PhysicsT::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tDirichletValues);
        }
    }

    /******************************************************************************/ /**
    * \brief Return solution database.
    * \return solution database
    **********************************************************************************/
    Plato::Solutions 
    getSolution() const
    {
        Plato::Solutions tSolution(mPhysics, mPDEType);
        tSolution.set("State", mGlobalStates);
        tSolution.set("Local State", mLocalStates);
        return tSolution;
    }

};
// class Problem

} // namespace UpdatedLagrangian

} // namespace Elliptic

} // namespace Plato

#include "elliptic/updated_lagrangian/Mechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Elliptic::UpdatedLagrangian::Problem<::Plato::Elliptic::UpdatedLagrangian::Mechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Elliptic::UpdatedLagrangian::Problem<::Plato::Elliptic::UpdatedLagrangian::Mechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Elliptic::UpdatedLagrangian::Problem<::Plato::Elliptic::UpdatedLagrangian::Mechanics<3>>;
#endif

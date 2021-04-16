#ifndef PLATO_PROBLEM_HPP
#define PLATO_PROBLEM_HPP

#include "PlatoUtilities.hpp"

#include <memory>
#include <sstream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "BLAS1.hpp"
#include "Solutions.hpp"
#include "NaturalBCs.hpp"
#include "EssentialBCs.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyConstraints.hpp"
#include "SpatialModel.hpp"

#include "ParseTools.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoAbstractProblem.hpp"

#include "Geometrical.hpp"
#include "geometric/ScalarFunctionBase.hpp"
#include "geometric/ScalarFunctionBaseFactory.hpp"

#include "elliptic/VectorFunction.hpp"
#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include "AnalyzeMacros.hpp"

#include "alg/ParallelComm.hpp"
#include "alg/PlatoSolverFactory.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename SimplexPhysics>
class Problem: public Plato::AbstractProblem
{
private:

    using Criterion       = std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>;
    using Criteria        = std::map<std::string, Criterion>;

    using LinearCriterion = std::shared_ptr<Plato::Geometric::ScalarFunctionBase>;
    using LinearCriteria  = std::map<std::string, LinearCriterion>;

    static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::mNumSpatialDims; /*!< spatial dimensions */

    using VectorFunctionType = Plato::Elliptic::VectorFunction<SimplexPhysics>;

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    // required
    std::shared_ptr<VectorFunctionType> mPDE; /*!< equality constraint interface */

    LinearCriteria mLinearCriteria;
    Criteria       mCriteria;

    Plato::OrdinalType mNumNewtonSteps;
    Plato::Scalar      mNewtonResTol, mNewtonIncTol;

    bool mSaveState;

    Plato::ScalarMultiVector mAdjoint;
    Plato::ScalarVector mResidual;

    Plato::ScalarMultiVector mStates; /*!< state variables */

    bool mIsSelfAdjoint; /*!< indicates if problem is self-adjoint */

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian; /*!< Jacobian matrix */

    Plato::LocalOrdinalVector mBcDofs; /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */
    Plato::ScalarVector mBcValues; /*!< values associated with the Dirichlet boundary conditions */

    rcp<Plato::AbstractSolver> mSolver;
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
      mPDE(std::make_shared<VectorFunctionType>(mSpatialModel, mDataMap, aProblemParams, aProblemParams.get<std::string>("PDE Constraint"))),
      mNumNewtonSteps(Plato::ParseTools::getSubParam<int>   (aProblemParams, "Newton Iteration", "Maximum Iterations",  1  )),
      mNewtonIncTol  (Plato::ParseTools::getSubParam<double>(aProblemParams, "Newton Iteration", "Increment Tolerance", 0.0)),
      mNewtonResTol  (Plato::ParseTools::getSubParam<double>(aProblemParams, "Newton Iteration", "Residual Tolerance",  0.0)),
      mSaveState     (aProblemParams.sublist("Elliptic").isType<Teuchos::Array<std::string>>("Plottable")),
      mResidual      ("MyResidual", mPDE->size()),
      mStates        ("States", static_cast<Plato::OrdinalType>(1), mPDE->size()),
      mJacobian      (Teuchos::null),
      mIsSelfAdjoint (aProblemParams.get<bool>("Self-Adjoint", false)),
      mPhysics       (aProblemParams.get<std::string>("Physics"))
    {
        this->initialize(aProblemParams);

        Plato::SolverFactory tSolverFactory(aProblemParams.sublist("Linear Solver"));
        mSolver = tSolverFactory.create(aMesh, aMachine, SimplexPhysics::mNumDofsPerNode);
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

    void applyConstraints(
      const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
      const Plato::ScalarVector & aVector
    ){}
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
    //**********************************************************************************/
    {
        if(mJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues, aScale);
        }
        else
        {
            Plato::applyConstraints<SimplexPhysics::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues, aScale);
        }
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
     * \return solution database
    **********************************************************************************/
    Plato::Solutions
    solution(const Plato::ScalarVector & aControl)
    {
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        Plato::ScalarVector tStatesSubView = Kokkos::subview(mStates, tTIME_STEP_INDEX, Kokkos::ALL());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tStatesSubView);

        mDataMap.clearStates();

        // inner loop for non-linear models
        for(Plato::OrdinalType tNewtonIndex = 0; tNewtonIndex < mNumNewtonSteps; tNewtonIndex++)
        {
            mResidual = mPDE->value(tStatesSubView, aControl);
            Plato::blas1::scale(-1.0, mResidual);

            if (mNumNewtonSteps > 1) {
                auto tResidualNorm = Plato::blas1::norm(mResidual);
                std::cout << " Residual norm: " << tResidualNorm << std::endl;
                if (tResidualNorm < mNewtonResTol) {
                    std::cout << " Residual norm tolerance satisfied." << std::endl;
                    break;
                }
            }

            mJacobian = mPDE->gradient_u(tStatesSubView, aControl);

            Plato::OrdinalType tScale = (tNewtonIndex == 0) ? 1.0 : 0.0;
            this->applyStateConstraints(mJacobian, mResidual, tScale);

            Plato::ScalarVector tDeltaD("increment", tStatesSubView.extent(0));
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tDeltaD);

            mSolver->solve(*mJacobian, tDeltaD, mResidual);
            Plato::blas1::axpy(1.0, tDeltaD, tStatesSubView);

            if (mNumNewtonSteps > 1) {
                auto tIncrementNorm = Plato::blas1::norm(tDeltaD);
                std::cout << " Delta norm: " << tIncrementNorm << std::endl;
                if (tIncrementNorm < mNewtonIncTol) {
                    std::cout << " Solution increment norm tolerance satisfied." << std::endl;
                    break;
                }
            }
        }

        if ( mSaveState )
        {
            // evaluate at new state
            mResidual  = mPDE->value(tStatesSubView, aControl);
            mDataMap.saveState();
        }

        Plato::Solutions tSolution(mPhysics);
        tSolution.set("State", mStates);
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
            Plato::Solutions tSolution(mPhysics);
            tSolution.set("State", mStates);
            Criterion tCriterion = mCriteria[aName];
            return tCriterion->value(tSolution, aControl);
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
     * \param [in] aSolution solution database
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
        if( mCriteria.count(aName) )
        {
            Criterion tCriterion = mCriteria[aName];
            return tCriterion->value(aSolution, aControl);
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
     * \param [in] aSolution solution database
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
     * \param [in] aSolution solution database
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
        if(aCriterion == nullptr)
        {
            THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }
        if(aSolution.empty())
        {
            THROWERR("SOLUTION DATABASE IS EMPTY")
        }

        if(static_cast<Plato::OrdinalType>(mAdjoint.size()) <= static_cast<Plato::OrdinalType>(0))
        {
            const auto tLength = mPDE->size();
            mAdjoint = Plato::ScalarMultiVector("Adjoint Variables", 1, tLength);
        }

        // compute dfdz: partial of criterion wrt z
        auto tPartialCriterionWRT_Control = aCriterion->gradient_z(aSolution, aControl);
        if(mIsSelfAdjoint)
        {
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialCriterionWRT_Control);
        }
        else
        {
            // compute dfdu: partial of criterion wrt u
            auto tPartialCriterionWRT_State = aCriterion->gradient_u(aSolution, aControl, /*stepIndex=*/0);
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialCriterionWRT_State);

            // compute dgdu: partial of PDE wrt state
            auto tState = aSolution.get("State");
            const Plato::OrdinalType tTIME_STEP_INDEX = 0;
            auto tStatesSubView = Kokkos::subview(tState, tTIME_STEP_INDEX, Kokkos::ALL());
            mJacobian = mPDE->gradient_u_T(tStatesSubView, aControl);

            this->applyAdjointConstraints(mJacobian, tPartialCriterionWRT_State);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

            Plato::ScalarVector tAdjointSubView = Kokkos::subview(mAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());

            mSolver->solve(*mJacobian, tAdjointSubView, tPartialCriterionWRT_State);

            // compute dgdz: partial of PDE wrt state.
            // dgdz is returned transposed, nxm.  n=z.size() and m=u.size().
            auto tPartialPDE_WRT_Control = mPDE->gradient_z(tStatesSubView, aControl);

            // compute dgdz . adjoint + dfdz
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Control, tAdjointSubView, tPartialCriterionWRT_Control);
        }
        return tPartialCriterionWRT_Control;
    }

    /******************************************************************************//**
     * \brief Evaluate criterion gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution solution database
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
     * \param [in] aSolution solution database
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
        if(aSolution.empty())
        {
            THROWERR("SOLUTION DATABASE IS EMPTY")
        }

        // compute partial derivative wrt x
        auto tState = aSolution.get("State");
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(tState, tTIME_STEP_INDEX, Kokkos::ALL());

        auto tPartialCriterionWRT_Config  = aCriterion->gradient_x(aSolution, aControl);

        if(mIsSelfAdjoint)
        {
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialCriterionWRT_Config);
        }
        else
        {
            // compute dfdu: partial of criterion wrt u
            auto tPartialCriterionWRT_State = aCriterion->gradient_u(aSolution, aControl, /*stepIndex=*/0);
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialCriterionWRT_State);

            // compute dgdu: partial of PDE wrt state
            mJacobian = mPDE->gradient_u(tStatesSubView, aControl);
            this->applyStateConstraints(mJacobian, tPartialCriterionWRT_State, 1.0);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

            Plato::ScalarVector
              tAdjointSubView = Kokkos::subview(mAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());

            mSolver->solve(*mJacobian, tAdjointSubView, tPartialCriterionWRT_State);

            // compute dgdx: partial of PDE wrt config.
            // dgdx is returned transposed, nxm.  n=x.size() and m=u.size().
            auto tPartialPDE_WRT_Config = mPDE->gradient_x(tStatesSubView, aControl);

            // compute dgdx . adjoint + dfdx
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Config, tAdjointSubView, tPartialCriterionWRT_Config);
        }
        return tPartialCriterionWRT_Config;
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
            Plato::Solutions tSolution(mPhysics);
            tSolution.set("State", mStates);
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
            Plato::Solutions tSolution(mPhysics);
            tSolution.set("State", mStates);
            Criterion tCriterion = mCriteria[aName];
            return criterionGradientX(aControl, tSolution, tCriterion);
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
        Plato::EssentialBCs<SimplexPhysics>
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
        auto tName = aProblemParams.get<std::string>("PDE Constraint");
        mPDE = std::make_shared<Plato::Elliptic::VectorFunction<SimplexPhysics>>(mSpatialModel, mDataMap, aProblemParams, tName);

        if(aProblemParams.isSublist("Criteria"))
        {
            Plato::Geometric::ScalarFunctionBaseFactory<Plato::Geometrical<SpatialDim>> tLinearFunctionBaseFactory;
            Plato::Elliptic::ScalarFunctionBaseFactory<SimplexPhysics> tNonlinearFunctionBaseFactory;

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
                    auto tCriterion = tNonlinearFunctionBaseFactory.create(mSpatialModel, mDataMap, aProblemParams, tName);
                    if( tCriterion != nullptr )
                    {
                        mCriteria[tName] = tCriterion;
                    }
                }
            }
            if( mCriteria.size() )
            {
                auto tLength = mPDE->size();
                mAdjoint = Plato::ScalarMultiVector("Adjoint Variables", 1, tLength);
            }
        }
    }

    void applyAdjointConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    {
        Plato::ScalarVector tDirichletValues("Dirichlet Values For Adjoint Problem", mBcValues.size());
        Plato::blas1::scale(static_cast<Plato::Scalar>(0.0), tDirichletValues);
        if(mJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tDirichletValues);
        }
        else
        {
            Plato::applyConstraints<SimplexPhysics::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tDirichletValues);
        }
    }

};
// class Problem

} // namespace Elliptic

} // namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Elliptic::Problem<::Plato::Thermal<1>>;
extern template class Plato::Elliptic::Problem<::Plato::Mechanics<1>>;
extern template class Plato::Elliptic::Problem<::Plato::Electromechanics<1>>;
extern template class Plato::Elliptic::Problem<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATOANALYZE_2D
extern template class Plato::Elliptic::Problem<::Plato::Thermal<2>>;
extern template class Plato::Elliptic::Problem<::Plato::Mechanics<2>>;
extern template class Plato::Elliptic::Problem<::Plato::Electromechanics<2>>;
extern template class Plato::Elliptic::Problem<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATOANALYZE_3D
extern template class Plato::Elliptic::Problem<::Plato::Thermal<3>>;
extern template class Plato::Elliptic::Problem<::Plato::Mechanics<3>>;
extern template class Plato::Elliptic::Problem<::Plato::Electromechanics<3>>;
extern template class Plato::Elliptic::Problem<::Plato::Thermomechanics<3>>;
#endif

#endif // PLATO_PROBLEM_HPP

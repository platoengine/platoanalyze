#ifndef PLATO_PROBLEM_HPP
#define PLATO_PROBLEM_HPP

#include "PlatoUtilities.hpp"

#include <memory>
#include <sstream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "BLAS1.hpp"
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

/* Notes:
 1.  The updateProblem function should send the MultiVector into objective.
 2.  Some of the objective and constraint functions dont use the solution arg.
*/

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

    static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::mNumSpatialDims; /*!< spatial dimensions */

    using VectorFunctionType = Plato::Elliptic::VectorFunction<SimplexPhysics>;

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    // required
    std::shared_ptr<VectorFunctionType> mPDE; /*!< equality constraint interface */

    // optional
//TODO    std::shared_ptr<Plato::Geometric::ScalarFunctionBase> mConstraint; /*!< constraint constraint interface */
    std::shared_ptr<Plato::Elliptic::ScalarFunctionBase> mObjective; /*!< objective constraint interface */

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

public:
    /******************************************************************************//**
     * \brief PLATO problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    Problem(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Teuchos::ParameterList& aInputParams,
      Comm::Machine aMachine
    ) :
      mSpatialModel  (aMesh, aMeshSets, aInputParams),
      mPDE(std::make_shared<VectorFunctionType>(mSpatialModel, mDataMap, aInputParams, aInputParams.get<std::string>("PDE Constraint"))),
//TODO      mConstraint    (nullptr),
      mObjective     (nullptr),
      mNumNewtonSteps(Plato::ParseTools::getSubParam<int>   (aInputParams, "Newton Iteration", "Maximum Iterations",  1  )),
      mNewtonIncTol  (Plato::ParseTools::getSubParam<double>(aInputParams, "Newton Iteration", "Increment Tolerance", 0.0)),
      mNewtonResTol  (Plato::ParseTools::getSubParam<double>(aInputParams, "Newton Iteration", "Residual Tolerance",  0.0)),
      mSaveState     (aInputParams.sublist("Elliptic").isType<Teuchos::Array<std::string>>("Plottable")),
      mResidual      ("MyResidual", mPDE->size()),
      mStates        ("States", static_cast<Plato::OrdinalType>(1), mPDE->size()),
      mJacobian      (Teuchos::null),
      mIsSelfAdjoint (aInputParams.get<bool>("Self-Adjoint", false))
    {
        this->initialize(aMesh, aMeshSets, aInputParams);

        Plato::SolverFactory tSolverFactory(aInputParams.sublist("Linear Solver"));
        mSolver = tSolverFactory.create(aMesh, aMachine, SimplexPhysics::mNumDofsPerNode);
    }

    virtual ~Problem(){}

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

    void appendResidual(const std::shared_ptr<VectorFunctionType>& aPDE)
    {
        mPDE = aPDE;
    }

    void appendObjective(const std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>& aObjective)
    {
        mObjective = aObjective;
    }

    void appendConstraint(const std::shared_ptr<Plato::Geometric::ScalarFunctionBase>& aConstraint)
    {
//TODO        mConstraint = aConstraint;
    }

    /******************************************************************************//**
     * \brief Return number of degrees of freedom in solution.
     * \return Number of degrees of freedom
    **********************************************************************************/
    Plato::OrdinalType getNumSolutionDofs()
    {
        return SimplexPhysics::mNumDofsPerNode;
    }

    /******************************************************************************//**
     * \brief Set state variables
     * \param [in] aGlobalState 2D view of state variables
    **********************************************************************************/
    void setGlobalSolution(const Plato::Solution & aSolution)
    {
        auto tState = aSolution.State;
        assert(tState.extent(0) == mStates.extent(0));
        assert(tState.extent(1) == mStates.extent(1));
        Kokkos::deep_copy(mStates, tState);
    }

    /******************************************************************************//**
     * \brief Return 2D view of state variables
     * \return aGlobalState 2D view of state variables
    **********************************************************************************/
    Plato::Solution getGlobalSolution()
    {
        return Plato::Solution(mStates);
    }

    /******************************************************************************//**
     * \brief Return 2D view of adjoint variables
     * \return 2D view of adjoint variables
    **********************************************************************************/
    Plato::Adjoint getAdjoint()
    {
        return Plato::Adjoint(mAdjoint);
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

    void applyBoundaryLoads(const Plato::ScalarVector & aForce){}

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aGlobalState 2D container of state variables
     * \param [in] aControl 1D container of control variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solution & aSolution)
    {
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aSolution.State, tTIME_STEP_INDEX, Kokkos::ALL());
        mObjective->updateProblem(tStatesSubView, aControl);
    }

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return Plato::Solution composed of state variables
    **********************************************************************************/
    Plato::Solution
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

        return Plato::Solution(mStates);
    }

    /******************************************************************************//**
     * \brief Evaluate objective function
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \return objective function value
    **********************************************************************************/
    Plato::Scalar
    objectiveValue(const Plato::ScalarVector & aControl, const Plato::Solution & aSolution)
    {
        assert(aSolution.State.extent(0) == mStates.extent(0));
        assert(aSolution.State.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
        }

        auto tObjFuncValue = mObjective->value(aSolution, aControl);
        return tObjFuncValue;
    }

    /******************************************************************************//**
     * \brief Evaluate objective function
     * \param [in] aControl 1D view of control variables
     * \return objective function value
    **********************************************************************************/
    Plato::Scalar
    objectiveValue(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
        }

        Plato::Solution tSolution = solution(aControl);
        return mObjective->value(tSolution, aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate constraint function
     * \param [in] aControl 1D view of control variables
     * \return constraint function value
    **********************************************************************************/
    Plato::Scalar
    constraintValue(const Plato::ScalarVector & aControl)
    {
//TODO        if(mConstraint == nullptr)
//TODO        {
//TODO            THROWERR("CONSTRAINT REQUESTED BUT NOT DEFINED BY USER.");
//TODO        }
//TODO        return mConstraint->value(aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate objective gradient wrt control variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aSolution Plato::Solution composed of state variables
     * \return 1D view - objective gradient wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    objectiveGradient(const Plato::ScalarVector & aControl, const Plato::Solution & aSolution)
    {
        assert(aSolution.State.extent(0) == mStates.extent(0));
        assert(aSolution.State.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
        }

        if(static_cast<Plato::OrdinalType>(mAdjoint.size()) <= static_cast<Plato::OrdinalType>(0))
        {
            const auto tLength = mPDE->size();
            mAdjoint = Plato::ScalarMultiVector("Adjoint Variables", 1, tLength);
        }

        // compute dfdz: partial of objective wrt z
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aSolution.State, tTIME_STEP_INDEX, Kokkos::ALL());

        auto tPartialObjectiveWRT_Control = mObjective->gradient_z(aSolution, aControl);
        if(mIsSelfAdjoint)
        {
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_Control);
        }
        else
        {
            // compute dfdu: partial of objective wrt u
            auto tPartialObjectiveWRT_State = mObjective->gradient_u(aSolution, aControl, /*stepIndex=*/0);
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_State);

            // compute dgdu: partial of PDE wrt state
            mJacobian = mPDE->gradient_u_T(tStatesSubView, aControl);

            this->applyAdjointConstraints(mJacobian, tPartialObjectiveWRT_State);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

            Plato::ScalarVector tAdjointSubView = Kokkos::subview(mAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());

            mSolver->solve(*mJacobian, tAdjointSubView, tPartialObjectiveWRT_State);

            // compute dgdz: partial of PDE wrt state.
            // dgdz is returned transposed, nxm.  n=z.size() and m=u.size().
            auto tPartialPDE_WRT_Control = mPDE->gradient_z(tStatesSubView, aControl);

            // compute dgdz . adjoint + dfdz
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Control, tAdjointSubView, tPartialObjectiveWRT_Control);
        }
        return tPartialObjectiveWRT_Control;
    }

    /******************************************************************************//**
     * \brief Evaluate objective gradient wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \param [in] aGlobalState 2D view of state variables
     * \return 1D view - objective gradient wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector
    objectiveGradientX(const Plato::ScalarVector & aControl, const Plato::Solution & aSolution)
    {
        assert(aSolution.State.extent(0) == mStates.extent(0));
        assert(aSolution.State.extent(1) == mStates.extent(1));

        if(mObjective == nullptr)
        {
            THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");

        }

        // compute partial derivative wrt x
        const Plato::OrdinalType tTIME_STEP_INDEX = 0;
        auto tStatesSubView = Kokkos::subview(aSolution.State, tTIME_STEP_INDEX, Kokkos::ALL());

        auto tPartialObjectiveWRT_Config  = mObjective->gradient_x(aSolution, aControl);

        if(mIsSelfAdjoint)
        {
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_Config);
        }
        else
        {
            // compute dfdu: partial of objective wrt u
            auto tPartialObjectiveWRT_State = mObjective->gradient_u(aSolution, aControl, /*stepIndex=*/0);
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), tPartialObjectiveWRT_State);

            // compute dgdu: partial of PDE wrt state
            mJacobian = mPDE->gradient_u(tStatesSubView, aControl);
            this->applyStateConstraints(mJacobian, tPartialObjectiveWRT_State, 1.0);

            // adjoint problem uses transpose of global stiffness, but we're assuming the constrained
            // system is symmetric.

            Plato::ScalarVector
              tAdjointSubView = Kokkos::subview(mAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());

            mSolver->solve(*mJacobian, tAdjointSubView, tPartialObjectiveWRT_State);

            // compute dgdx: partial of PDE wrt config.
            // dgdx is returned transposed, nxm.  n=x.size() and m=u.size().
            auto tPartialPDE_WRT_Config = mPDE->gradient_x(tStatesSubView, aControl);

            // compute dgdx . adjoint + dfdx
            Plato::MatrixTimesVectorPlusVector(tPartialPDE_WRT_Config, tAdjointSubView, tPartialObjectiveWRT_Config);
        }
        return tPartialObjectiveWRT_Config;
    }

    /******************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - constraint partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    constraintGradient(const Plato::ScalarVector & aControl)
    {
//TODO        if(mConstraint == nullptr)
//TODO        {
//TODO            THROWERR("CONSTRAINT REQUESTED BUT NOT DEFINED BY USER.");
//TODO        }
//TODO        return mConstraint->gradient_z(aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate objective partial derivative wrt control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - objective partial derivative wrt control variables
    **********************************************************************************/
    Plato::ScalarVector
    objectiveGradient(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
        }

        return mObjective->gradient_z(Plato::Solution(mStates), aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate objective partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - objective partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector
    objectiveGradientX(const Plato::ScalarVector & aControl)
    {
        if(mObjective == nullptr)
        {
            THROWERR("OBJECTIVE REQUESTED BUT NOT DEFINED BY USER.");
        }

        return mObjective->gradient_x(Plato::Solution(mStates), aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate constraint partial derivative wrt configuration variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view - constraint partial derivative wrt configuration variables
    **********************************************************************************/
    Plato::ScalarVector
    constraintGradientX(const Plato::ScalarVector & aControl)
    {
//TODO        if(mConstraint == nullptr)
//TODO        {
//TODO            THROWERR("CONSTRAINT REQUESTED BUT NOT DEFINED BY USER.");
//TODO        }
//TODO        return mConstraint->gradient_x(aControl);
    }

    /***************************************************************************//**
     * \brief Read essential (Dirichlet) boundary conditions from the Exodus file.
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
    *******************************************************************************/
    void readEssentialBoundaryConditions(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams)
    {
        if(aInputParams.isSublist("Essential Boundary Conditions") == false)
        {
            THROWERR("ESSENTIAL BOUNDARY CONDITIONS SUBLIST IS NOT DEFINED IN THE INPUT FILE.")
        }
        Plato::EssentialBCs<SimplexPhysics> tEssentialBoundaryConditions(aInputParams.sublist("Essential Boundary Conditions", false));
        tEssentialBoundaryConditions.get(aMeshSets, mBcDofs, mBcValues);
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
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Omega_h::Mesh& aMesh, Omega_h::MeshSets& aMeshSets, Teuchos::ParameterList& aInputParams)
    {
//        auto tName = aInputParams.get<std::string>("PDE Constraint");
//        mPDE = std::make_shared<VectorFunctionType>(aMesh, aMeshSets, mDataMap, aInputParams, tName);

        if(aInputParams.isType<std::string>("Constraint"))
        {
//TODO            Plato::Geometric::ScalarFunctionBaseFactory<Plato::Geometrical<SpatialDim>> tFunctionBaseFactory;
//TODO            std::string tName = aInputParams.get<std::string>("Constraint");
//TODO            mConstraint = tFunctionBaseFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tName);
        }

        if(aInputParams.isType<std::string>("Objective"))
        {
            Plato::Elliptic::ScalarFunctionBaseFactory<SimplexPhysics> tFunctionBaseFactory;
            std::string tName = aInputParams.get<std::string>("Objective");
            mObjective = tFunctionBaseFactory.create(aMesh, aMeshSets, mDataMap, aInputParams, tName);

            auto tLength = mPDE->size();
            mAdjoint = Plato::ScalarMultiVector("Adjoint Variables", 1, tLength);
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

//#include "Thermal.hpp"
#include "Mechanics.hpp"
//#include "Electromechanics.hpp"
//#include "Thermomechanics.hpp"

#ifdef PLATOANALYZE_1D
//extern template class Plato::Elliptic::Problem<::Plato::Thermal<1>>;
extern template class Plato::Elliptic::Problem<::Plato::Mechanics<1>>;
//extern template class Plato::Elliptic::Problem<::Plato::Electromechanics<1>>;
//extern template class Plato::Elliptic::Problem<::Plato::Thermomechanics<1>>;
#endif
#ifdef PLATOANALYZE_2D
//extern template class Plato::Elliptic::Problem<::Plato::Thermal<2>>;
extern template class Plato::Elliptic::Problem<::Plato::Mechanics<2>>;
//extern template class Plato::Elliptic::Problem<::Plato::Electromechanics<2>>;
//extern template class Plato::Elliptic::Problem<::Plato::Thermomechanics<2>>;
#endif
#ifdef PLATOANALYZE_3D
//extern template class Plato::Elliptic::Problem<::Plato::Thermal<3>>;
extern template class Plato::Elliptic::Problem<::Plato::Mechanics<3>>;
//extern template class Plato::Elliptic::Problem<::Plato::Electromechanics<3>>;
//extern template class Plato::Elliptic::Problem<::Plato::Thermomechanics<3>>;
#endif

#endif // PLATO_PROBLEM_HPP

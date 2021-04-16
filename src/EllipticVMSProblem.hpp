#ifndef ELLIPTIC_VMS_PROBLEM_HPP
#define ELLIPTIC_VMS_PROBLEM_HPP

#include <memory>
#include <sstream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "BLAS1.hpp"
#include "BLAS2.hpp"
#include "NaturalBCs.hpp"
#include "UtilsOmegaH.hpp"
#include "EssentialBCs.hpp"
#include "ImplicitFunctors.hpp"
#include "ApplyConstraints.hpp"

#include "Solutions.hpp"
#include "VectorFunctionVMS.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoAbstractProblem.hpp"
#include "ParseTools.hpp"
#include "Plato_Solve.hpp"
#include "alg/PlatoSolverFactory.hpp"

#include "Geometrical.hpp"
#include "geometric/ScalarFunctionBase.hpp"
#include "geometric/ScalarFunctionBaseFactory.hpp"

#include "elliptic/ScalarFunctionBase.hpp"
#include "elliptic/ScalarFunctionBaseFactory.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename SimplexPhysics>
class EllipticVMSProblem: public Plato::AbstractProblem
{
private:

    using Criterion       = std::shared_ptr<Plato::Elliptic::ScalarFunctionBase>;
    using Criteria        = std::map<std::string, Criterion>;

    using LinearCriterion = std::shared_ptr<Plato::Geometric::ScalarFunctionBase>;
    using LinearCriteria  = std::map<std::string, LinearCriterion>;

    static constexpr auto mSpaceDim = SimplexPhysics::SpaceDim; /*!< spatial dimensions*/
    static constexpr auto mPressureDofOffset = SimplexPhysics::mPressureDofOffset;          /*!< number of pressure dofs offset*/
    static constexpr auto mNumGlobalDofsPerNode = SimplexPhysics::mNumDofsPerNode;          /*!< number of global degrees of freedom per node*/

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    // required
    Plato::VectorFunctionVMS<SimplexPhysics> mPDEConstraint; /*!< equality constraint interface */
    Plato::VectorFunctionVMS<typename SimplexPhysics::ProjectorT> mStateProjection; /*!< projection interface */

    // optional
    LinearCriteria mLinearCriteria;
    Criteria mCriteria;

    Plato::OrdinalType mNumSteps, mNumNewtonSteps, mCurrentNewtonStep;
    Plato::Scalar mTimeStep;

    Plato::ScalarVector      mResidual;
    Plato::ScalarMultiVector mGlobalState; /*!< state variables */
    Plato::ScalarMultiVector mLambda;
    Teuchos::RCP<Plato::CrsMatrixType> mJacobian; /*!< Jacobian matrix */

    Plato::ScalarVector mProjResidual;
    Plato::ScalarVector mProjPGrad;
    Plato::ScalarVector mProjectState;
    Plato::ScalarVector mEta;
    Teuchos::RCP<Plato::CrsMatrixType> mProjJacobian; /*!< Jacobian matrix */

    Plato::LocalOrdinalVector mBcDofs; /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */
    Plato::ScalarVector mBcValues; /*!< values associated with the Dirichlet boundary conditions */

    rcp<Plato::AbstractSolver> mSolver;
    std::string mPhysics; /*!< physics used for the simulation */

public:
    /******************************************************************************//**
     * \brief PLATO problem constructor
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    EllipticVMSProblem(
      Omega_h::Mesh& aMesh,
      Omega_h::MeshSets& aMeshSets,
      Teuchos::ParameterList& aInputParams,
      Comm::Machine aMachine
    ) :
      mSpatialModel    (aMesh, aMeshSets, aInputParams),
      mPDEConstraint   (mSpatialModel, mDataMap, aInputParams, aInputParams.get<std::string>("PDE Constraint")),
      mStateProjection (mSpatialModel, mDataMap, aInputParams, std::string("State Gradient Projection")),
      mNumSteps        (Plato::ParseTools::getSubParam<int>   (aInputParams, "Time Stepping", "Number Time Steps",    1  )),
      mTimeStep        (Plato::ParseTools::getSubParam<Plato::Scalar>(aInputParams, "Time Stepping", "Time Step",     1.0)),
      mNumNewtonSteps  (Plato::ParseTools::getSubParam<int>   (aInputParams, "Newton Iteration", "Number Iterations", 2  )),
      mResidual        ("MyResidual", mPDEConstraint.size()),
      mGlobalState     ("States", mNumSteps, mPDEConstraint.size()),
      mJacobian        (Teuchos::null),
      mProjResidual    ("MyProjResidual", mStateProjection.size()),
      mProjPGrad       ("Projected PGrad", mStateProjection.size()),
      mProjectState    ("Project State", aMesh.nverts()),
      mProjJacobian    (Teuchos::null),
      mPhysics         (aInputParams.get<std::string>("Physics"))
    {
        this->initialize(aInputParams);

        Plato::SolverFactory tSolverFactory(aInputParams.sublist("Linear Solver"));
        mSolver = tSolverFactory.create(aMesh, aMachine, SimplexPhysics::mNumDofsPerNode);
    }

    /***************************************************************************//**
     * \brief Save states to visualization file
     * \param [in] aFilepath output/viz directory path
     * \param [in] aMesh     Omega_h mesh database
    *******************************************************************************/
    void saveStates(const std::string& aFilepath, Omega_h::Mesh& aMesh)
    {
        auto tNumNodes = mPDEConstraint.numNodes();
        Plato::ScalarMultiVector tPressure("Pressure", mGlobalState.extent(0), tNumNodes);
        Plato::ScalarMultiVector tDisplacements("Displacements", mGlobalState.extent(0), tNumNodes*mSpaceDim);
        Plato::blas2::extract<mNumGlobalDofsPerNode, mPressureDofOffset>(mGlobalState, tPressure);
        Plato::blas2::extract<mNumGlobalDofsPerNode, mSpaceDim>(tNumNodes, mGlobalState, tDisplacements);

        Omega_h::vtk::Writer tWriter = Omega_h::vtk::Writer(aFilepath.c_str(), &aMesh, mSpaceDim);
        for(Plato::OrdinalType tSnapshot = 1; tSnapshot < tDisplacements.extent(0); tSnapshot++)
        {
            auto tPressSubView = Kokkos::subview(tPressure, tSnapshot, Kokkos::ALL());
            auto tPressSubViewDefaultMirror = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(),tPressSubView);
            auto tDispSubView = Kokkos::subview(tDisplacements, tSnapshot, Kokkos::ALL());
            auto tDispSubViewDefaultMirror = Kokkos::create_mirror_view(Kokkos::DefaultExecutionSpace(),tDispSubView);
            aMesh.add_tag(Omega_h::VERT, "Pressure", 1, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tPressSubViewDefaultMirror)));
            aMesh.add_tag(Omega_h::VERT, "Displacements", mSpaceDim, Omega_h::Reals(Omega_h::Write<Omega_h::Real>(tDispSubViewDefaultMirror)));
            Plato::omega_h::add_element_state_tags(aMesh, mDataMap, tSnapshot);
            auto tTags = Omega_h::vtk::get_all_vtk_tags(&aMesh, mSpaceDim);
            auto tTime = mTimeStep * static_cast<Plato::Scalar>(tSnapshot);
            tWriter.write(tSnapshot, tTime, tTags);
        }
    }

    /***************************************************************************//**
     * \brief Read essential (Dirichlet) boundary conditions from the Exodus file.
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aInputParams input parameters database
    *******************************************************************************/
    void readEssentialBoundaryConditions(Teuchos::ParameterList& aInputParams)
    {
        if(aInputParams.isSublist("Essential Boundary Conditions") == false)
        {
            THROWERR("ESSENTIAL BOUNDARY CONDITIONS SUBLIST IS NOT DEFINED IN THE INPUT FILE.")
        }
        Plato::EssentialBCs<SimplexPhysics>
        tEssentialBoundaryConditions(aInputParams.sublist("Essential Boundary Conditions", false), mSpatialModel.MeshSets);
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

    /******************************************************************************//**
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    **********************************************************************************/
    void applyConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    {
        if(mBcValues.size() <= static_cast<Plato::OrdinalType>(0))
        { THROWERR("Elliptic VMS Problem: Essential Boundary Conditions Values array is empty.") }

        if(mBcDofs.size() <= static_cast<Plato::OrdinalType>(0))
        { THROWERR("Elliptic VMS Problem: Essential Boundary Conditions Dofs array is empty.") }

        Plato::ScalarVector tBcValues("Dirichlet Values", mBcValues.size());
        Plato::blas1::fill(0.0, tBcValues);
        if(mCurrentNewtonStep == static_cast<Plato::OrdinalType>(0))
        { Plato::blas1::update(static_cast<Plato::Scalar>(1.), mBcValues, static_cast<Plato::Scalar>(0.), tBcValues); }

        if(mJacobian->isBlockMatrix())
        {
            Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tBcValues);
        }
        else
        {
            Plato::applyConstraints<SimplexPhysics::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tBcValues);
        }
    }

    /******************************************************************************//**
     * \brief Apply Dirichlet constraints for adjoint problem
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    **********************************************************************************/
    void applyAdjointConstraints(const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, const Plato::ScalarVector & aVector)
    {
        if(mBcValues.size() <= static_cast<Plato::OrdinalType>(0))
        { THROWERR("Elliptic VMS Problem: Essential Boundary Conditions Values array is empty.") }

        if(mBcDofs.size() <= static_cast<Plato::OrdinalType>(0))
        { THROWERR("Elliptic VMS Problem: Essential Boundary Conditions Dofs array is empty.") }

        Plato::ScalarVector tBcValues("Dirichlet Values", mBcValues.size());
        Plato::blas1::scale(static_cast<Plato::Scalar>(0.0), tBcValues);

        if(aMatrix->isBlockMatrix())
        {
            Plato::applyBlockConstraints<SimplexPhysics::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tBcValues);
        }
        else
        {
            Plato::applyConstraints<SimplexPhysics::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, tBcValues);
        }
    }

    void applyBoundaryLoads(const Plato::ScalarVector & aForce){}

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D container of control variables
     * \param [in] aGlobalState 2D container of state variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aGlobalSolution)
    { return; }

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return Plato::Solutions composed of state variables
    **********************************************************************************/
    Plato::Solutions
    solution(
        const Plato::ScalarVector & aControl
    ) override
    {

        Plato::ScalarVector tStateIncrement("State increment", mGlobalState.extent(1));

        // outer loop for load/time steps
        for(Plato::OrdinalType tStepIndex = 1; tStepIndex < mNumSteps; tStepIndex++)
        {
            // compute the projected pressure gradient
            Plato::ScalarVector tState = Kokkos::subview(mGlobalState, tStepIndex, Kokkos::ALL());
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tState);
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mProjPGrad);
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mProjectState);

            // inner loop for load/time steps
            for(Plato::OrdinalType tNewtonIndex = 0; tNewtonIndex < mNumNewtonSteps; tNewtonIndex++)
            {
                mCurrentNewtonStep = tNewtonIndex;
                mProjResidual = mStateProjection.value      (mProjPGrad, mProjectState, aControl);
                mProjJacobian = mStateProjection.gradient_u (mProjPGrad, mProjectState, aControl);

                Plato::Solve::RowSummed<SimplexPhysics::mNumSpatialDims>(mProjJacobian, mProjPGrad, mProjResidual);

                // compute the state solution
                mResidual = mPDEConstraint.value      (tState, mProjPGrad, aControl);
                Plato::blas1::scale(static_cast<Plato::Scalar>(-1.0), mResidual);
                mJacobian = mPDEConstraint.gradient_u (tState, mProjPGrad, aControl);

                this->applyConstraints(mJacobian, mResidual);

                mSolver->solve(*mJacobian, tStateIncrement, mResidual);

                // update the state with the new increment
                Plato::blas1::update(static_cast<Plato::Scalar>(1.0), tStateIncrement, static_cast<Plato::Scalar>(1.0), tState);

                // copy projection state
                Plato::blas1::extract<SimplexPhysics::mNumDofsPerNode,
                                      SimplexPhysics::ProjectorT::SimplexT::mProjectionDof>(tState, mProjectState);
            }

            mResidual = mPDEConstraint.value(tState, mProjPGrad, aControl);

        }

        Plato::Solutions tSolution(mPhysics);
        tSolution.set("State", mGlobalState);
        return tSolution;
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
            tSolution.set("State", mGlobalState);
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
            tSolution.set("State", mGlobalState);
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
     * \param [in] aSolution solution databse
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
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(aSolution.empty())
        {
            THROWERR("\nSolutions database is empty.\n");
        }
        if(aCriterion == nullptr)
        {
            THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }

        // compute dfdz: partial of criterion wrt z
        auto t_df_dz = aCriterion->gradient_z(aSolution, aControl, mTimeStep);

        // outer loop for load/time steps
        auto tState = aSolution.get("State");
        auto tLastStepIndex = mNumSteps - 1;
        for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--)
        {
            // compute dfdu: partial of criterion wrt u
            auto t_df_du = aCriterion->gradient_u(aSolution, aControl, tStepIndex, mTimeStep);
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), t_df_du);

            // compute nodal projection of pressure gradient
            Plato::ScalarVector tStateAtStepK = Kokkos::subview(tState, tStepIndex, Kokkos::ALL());
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mProjPGrad);
            // extract projection state
            Plato::blas1::extract<SimplexPhysics::mNumDofsPerNode,
                                  SimplexPhysics::ProjectorT::SimplexT::mProjectionDof>(tStateAtStepK, mProjectState);
            mProjResidual = mStateProjection.value      (mProjPGrad, mProjectState, aControl);
            mProjJacobian = mStateProjection.gradient_u (mProjPGrad, mProjectState, aControl);
            Plato::Solve::RowSummed<SimplexPhysics::mNumSpatialDims>(mProjJacobian, mProjPGrad, mProjResidual);

            // compute dgdu^T: Transpose of partial of PDE wrt state
            mJacobian = mPDEConstraint.gradient_u_T(tStateAtStepK, mProjPGrad, aControl);

            // compute dPdn^T: Transpose of partial of projection residual wrt state
            auto t_dP_dn_T = mStateProjection.gradient_n_T(mProjPGrad, mProjectState, aControl);

            // compute dgdPI^T: Transpose of partial of PDE wrt projected pressure gradient
            auto t_dg_dPI_T = mPDEConstraint.gradient_n_T(tStateAtStepK, mProjPGrad, aControl);

            // compute dgdu^T - dP_dn_T X (mProjJacobian)^-1 X t_dg_dPI_T
            auto tRow = SimplexPhysics::ProjectorT::SimplexT::mProjectionDof;
            Plato::Condense(mJacobian, t_dP_dn_T, mProjJacobian,  t_dg_dPI_T, tRow);

            this->applyAdjointConstraints(mJacobian, t_df_du);

            Plato::ScalarVector tLambda = Kokkos::subview(mLambda, tStepIndex, Kokkos::ALL());
            mSolver->solve(*mJacobian, tLambda, t_df_du);

            // compute adjoint variable for projection equation
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mProjResidual);
            Plato::MatrixTimesVectorPlusVector(t_dg_dPI_T, tLambda, mProjResidual);
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), mProjResidual);
            Plato::Solve::RowSummed<SimplexPhysics::mNumSpatialDims>(mProjJacobian, mEta, mProjResidual);

            // compute dgdz: partial of PDE wrt state.
            // dgdz is returned transposed, nxm.  n=z.size() and m=u.size().
            auto t_dg_dz = mPDEConstraint.gradient_z(tStateAtStepK, mProjPGrad, aControl);

            // compute dfdz += dgdz . lambda
            // dPdz is returned transposed, nxm.  n=z.size() and m=u.size().
            Plato::MatrixTimesVectorPlusVector(t_dg_dz, tLambda, t_df_dz);

            // compute dPdz: partial of projection wrt state.
            // dPdz is returned transposed, nxm.  n=z.size() and m=PI.size().
            auto t_dP_dz = mStateProjection.gradient_z(mProjPGrad, tStateAtStepK, aControl);

            // compute dfdz += dPdz . eta
            Plato::MatrixTimesVectorPlusVector(t_dP_dz, mEta, t_df_dz);
        }

        return t_df_dz;
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
            tSolution.set("State", mGlobalState);
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
        if(aControl.size() <= static_cast<Plato::OrdinalType>(0))
        {
            THROWERR("\nCONTROL 1D VIEW IS EMPTY.\n");
        }
        if(aSolution.empty())
        {
            THROWERR("\nSolution database is empty.\n");
        }
        if(aCriterion == nullptr)
        {
            THROWERR("REQUESTED CRITERION NOT DEFINED BY USER.");
        }

        // compute dfdx: partial of criterion wrt x
        auto t_df_dx = aCriterion->gradient_x(aSolution, aControl, mTimeStep);

        // outer loop for load/time steps
        auto tState = aSolution.get("State");
        auto tLastStepIndex = mNumSteps - 1;
        for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--)
        {
            // compute dfdu: partial of criterion wrt u
            auto t_df_du = aCriterion->gradient_u(aSolution, aControl, tStepIndex, mTimeStep);
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), t_df_du);

            // compute nodal projection of pressure gradient
            Plato::ScalarVector tStateAtStepK = Kokkos::subview(tState, tStepIndex, Kokkos::ALL());
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mProjPGrad);
            auto mProjResidual = mStateProjection.value      (mProjPGrad, tStateAtStepK, aControl);
            auto mProjJacobian = mStateProjection.gradient_u (mProjPGrad, tStateAtStepK, aControl);
            // extract projection state
            Plato::blas1::extract<SimplexPhysics::mNumDofsPerNode,
                                  SimplexPhysics::ProjectorT::SimplexT::mProjectionDof>(tStateAtStepK, mProjectState);
            mProjResidual = mStateProjection.value      (mProjPGrad, mProjectState, aControl);
            mProjJacobian = mStateProjection.gradient_u (mProjPGrad, mProjectState, aControl);
            Plato::Solve::RowSummed<SimplexPhysics::mNumSpatialDims>(mProjJacobian, mProjPGrad, mProjResidual);

            // compute dgdu^T: Transpose of partial of PDE wrt state
            mJacobian = mPDEConstraint.gradient_u_T(tStateAtStepK, mProjPGrad, aControl);

            // compute dPdn^T: Transpose of partial of projection residual wrt state
            auto t_dP_dn_T = mStateProjection.gradient_n_T(mProjPGrad, mProjectState, aControl);

            // compute dgdPI: Transpose of partial of PDE wrt projected pressure gradient
            auto t_dg_dPI_T = mPDEConstraint.gradient_n_T(tStateAtStepK, mProjPGrad, aControl);

            // compute dgdu^T - dP_dn_T X (mProjJacobian)^-1 X t_dg_dPI_T
            auto tRow = SimplexPhysics::ProjectorT::SimplexT::mProjectionDof;
            Plato::Condense(mJacobian, t_dP_dn_T, mProjJacobian,  t_dg_dPI_T, tRow);

            this->applyAdjointConstraints(mJacobian, t_df_du);

            Plato::ScalarVector tLambda = Kokkos::subview(mLambda, tStepIndex, Kokkos::ALL());
            mSolver->solve(*mJacobian, tLambda, t_df_du);

            // compute adjoint variable for projection equation
            Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), mProjResidual);
            Plato::MatrixTimesVectorPlusVector(t_dg_dPI_T, tLambda, mProjResidual);
            Plato::blas1::scale(static_cast<Plato::Scalar>(-1), mProjResidual);
            Plato::Solve::RowSummed<SimplexPhysics::mNumSpatialDims>(mProjJacobian, mEta, mProjResidual);

            // compute dgdx: partial of PDE wrt configuration
            // dgdx is returned transposed, nxm.  n=z.size() and m=u.size().
            auto t_dg_dx = mPDEConstraint.gradient_x(tStateAtStepK, mProjPGrad, aControl);

            // compute dfdx += dgdx . lambda
            // dPdx is returned transposed, nxm.  n=z.size() and m=u.size().
            Plato::MatrixTimesVectorPlusVector(t_dg_dx, tLambda, t_df_dx);

            // compute dPdx: partial of projection wrt configuration
            // dPdx is returned transposed, nxm.  n=z.size() and m=PI.size().
            auto t_dP_dx = mStateProjection.gradient_x(mProjPGrad, tStateAtStepK, aControl);

            // compute dfdx += dPdx . eta
            Plato::MatrixTimesVectorPlusVector(t_dP_dx, mEta, t_df_dx);
        }

        return t_df_dx;
    }

private:
    /******************************************************************************//**
     * \brief Initialize member data
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void
    initialize(
        Teuchos::ParameterList& aProblemParams
    )
    {
        if(aProblemParams.isSublist("Time Stepping") == true)
        {
            mNumSteps = aProblemParams.sublist("Time Stepping").get<int>("Number Time Steps");
            mTimeStep = aProblemParams.sublist("Time Stepping").get<Plato::Scalar>("Time Step");
        } 
        else
        {
            mNumSteps = 1;
            mTimeStep = 1.0;
        }

        if(aProblemParams.isSublist("Newton Iteration") == true)
        {
            mNumNewtonSteps = aProblemParams.sublist("Newton Iteration").get<int>("Number Iterations");
        } 
        else
        {
            mNumNewtonSteps = 2;
        }

        if(aProblemParams.isSublist("Criteria"))
        {
            Plato::Geometric::ScalarFunctionBaseFactory<Plato::Geometrical<mSpaceDim>> tLinearFunctionBaseFactory;
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
                auto tLength = mPDEConstraint.size();
                mLambda = Plato::ScalarMultiVector("Lambda", mNumSteps, tLength);
                tLength = mStateProjection.size();
                mEta = Plato::ScalarVector("Eta", tLength);
            }
        }
    }
};
// class EllipticVMSProblem

} // namespace Plato

#include "StabilizedMechanics.hpp"
#include "StabilizedThermomechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::EllipticVMSProblem<::Plato::StabilizedMechanics<1>>;
extern template class Plato::EllipticVMSProblem<::Plato::StabilizedThermomechanics<1>>;
#endif
#ifdef PLATOANALYZE_2D
extern template class Plato::EllipticVMSProblem<::Plato::StabilizedMechanics<2>>;
extern template class Plato::EllipticVMSProblem<::Plato::StabilizedThermomechanics<2>>;
#endif
#ifdef PLATOANALYZE_3D
extern template class Plato::EllipticVMSProblem<::Plato::StabilizedMechanics<3>>;
extern template class Plato::EllipticVMSProblem<::Plato::StabilizedThermomechanics<3>>;
#endif

#endif // PLATO_PROBLEM_HPP

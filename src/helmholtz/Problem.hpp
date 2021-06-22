#pragma once

#include "PlatoUtilities.hpp"

#include <memory>
#include <sstream>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "BLAS1.hpp"
#include "Solutions.hpp"
/* #include "NaturalBCs.hpp" */
/* #include "EssentialBCs.hpp" */
#include "AnalyzeOutput.hpp"
#include "ImplicitFunctors.hpp"
/* #include "ApplyConstraints.hpp" */
#include "SpatialModel.hpp"

#include "ParseTools.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoStaticsTypes.hpp"
#include "PlatoAbstractProblem.hpp"
#include "PlatoUtilities.hpp"

#include "Geometrical.hpp"
#include "geometric/ScalarFunctionBase.hpp"
#include "geometric/ScalarFunctionBaseFactory.hpp"

#include "helmholtz/VectorFunction.hpp"
#include "AnalyzeMacros.hpp"

#include "alg/ParallelComm.hpp"
#include "alg/PlatoSolverFactory.hpp"

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************//**
 * \brief Manage scalar and vector function evaluations
**********************************************************************************/
template<typename PhysicsT>
class Problem: public Plato::AbstractProblem
{
private:

    static constexpr Plato::OrdinalType mSpatialDim = PhysicsT::mNumSpatialDims; /*!< spatial dimensions */

    using VectorFunctionType = Plato::Helmholtz::VectorFunction<PhysicsT>;

    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    // required
    std::shared_ptr<VectorFunctionType> mPDE; /*!< equality constraint interface */

    Plato::ScalarVector mResidual;

    Plato::ScalarMultiVector mStates; /*!< state variables */

    Teuchos::RCP<Plato::CrsMatrixType> mJacobian; /*!< Jacobian matrix */

    Plato::LocalOrdinalVector mBcDofs; /*!< list of degrees of freedom associated with the Dirichlet boundary conditions */
    Plato::ScalarVector mBcValues; /*!< values associated with the Dirichlet boundary conditions */

    rcp<Plato::AbstractSolver> mSolver;

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
      mPDE(std::make_shared<VectorFunctionType>(mSpatialModel, mDataMap, aProblemParams, aProblemParams.get<std::string>("PDE Constraint"))),
      mResidual      ("MyResidual", mPDE->size()),
      mStates        ("States", static_cast<Plato::OrdinalType>(1), mPDE->size()),
      mJacobian      (Teuchos::null),
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
     * \brief Output solution to visualization file.
     * \param [in] aFilepath output/visualizaton file path
    **********************************************************************************/
    void output(const std::string & aFilepath) override
    {
        THROWERR("OUTPUT: NOT SURE YET IF NEEDED FOR HELMHOLTZ")
        /* auto tDataMap = this->getDataMap(); */
        /* auto tSolution = this->getSolution(); */
        /* auto tSolutionOutput = mPDE->getSolutionStateOutputData(tSolution); */
        /* Plato::universal_solution_output<mSpatialDim>(aFilepath, tSolutionOutput, tDataMap, mSpatialModel.Mesh); */
    }

    /******************************************************************************//**
     * \brief Apply Dirichlet constraints
     * \param [in] aMatrix Compressed Row Storage (CRS) matrix
     * \param [in] aVector 1D view of Right-Hand-Side forces
    **********************************************************************************/
    /* void applyStateConstraints( */
    /*   const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix, */
    /*   const Plato::ScalarVector & aVector, */
    /*         Plato::Scalar aScale */
    /* ) */
    //**********************************************************************************/
    /* { */
    /*     if(mJacobian->isBlockMatrix()) */
    /*     { */
    /*         Plato::applyBlockConstraints<PhysicsT::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues, aScale); */
    /*     } */
    /*     else */
    /*     { */
    /*         Plato::applyConstraints<PhysicsT::mNumDofsPerNode>(aMatrix, aVector, mBcDofs, mBcValues, aScale); */
    /*     } */
    /* } */

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aGlobalState 2D container of state variables
     * \param [in] aControl 1D container of control variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl, const Plato::Solutions & aSolution)
    {
        THROWERR("UPDATE PROBLEM: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /******************************************************************************//**
     * \brief Solve system of equations
     * \param [in] aControl 1D view of control variables
     * \return solution database
    **********************************************************************************/
    Plato::Solutions
    solution(const Plato::ScalarVector & aControl)
    {
        Plato::ScalarVector tStatesSubView = Kokkos::subview(mStates, 0, Kokkos::ALL());
        Plato::blas1::fill(static_cast<Plato::Scalar>(0.0), tStatesSubView);

        mDataMap.clearStates();
        mDataMap.scalarNodeFields["Topology"] = aControl;

        mResidual = mPDE->value(tStatesSubView, aControl);
        Plato::blas1::scale(-1.0, mResidual);

        mJacobian = mPDE->gradient_u(tStatesSubView, aControl);

        /* this->applyStateConstraints(mJacobian, mResidual); */

        mSolver->solve(*mJacobian, tStatesSubView, mResidual);

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
        THROWERR("CRITERION VALUE: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
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
        THROWERR("CRITERION VALUE: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
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
        THROWERR("CRITERION GRADIENT: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
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
        THROWERR("CRITERION GRADIENT X: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
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
        THROWERR("CRITERION GRADIENT: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
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
        THROWERR("CRITERION GRADIENT X: NO CRITERION ASSOCIATED WITH HELMHOLTZ FILTER PROBLEM.")
    }

    /***************************************************************************//**
     * \brief Read essential (Dirichlet) boundary conditions from the Exodus file.
     * \param [in] aProblemParams input parameters database
    *******************************************************************************/
    /* void readEssentialBoundaryConditions(Teuchos::ParameterList& aProblemParams) */
    /* { */
    /*     if(aProblemParams.isSublist("Essential Boundary Conditions") == false) */
    /*     { */
    /*         THROWERR("ESSENTIAL BOUNDARY CONDITIONS SUBLIST IS NOT DEFINED IN THE INPUT FILE.") */
    /*     } */
    /*     Plato::EssentialBCs<PhysicsT> */
    /*     tEssentialBoundaryConditions(aProblemParams.sublist("Essential Boundary Conditions", false), mSpatialModel.MeshSets); */
    /*     tEssentialBoundaryConditions.get(mBcDofs, mBcValues); */
    /* } */

    /***************************************************************************//**
     * \brief Set essential (Dirichlet) boundary conditions
     * \param [in] aDofs   degrees of freedom associated with Dirichlet boundary conditions
     * \param [in] aValues values associated with Dirichlet degrees of freedom
    *******************************************************************************/
    /* void setEssentialBoundaryConditions(const Plato::LocalOrdinalVector & aDofs, const Plato::ScalarVector & aValues) */
    /* { */
    /*     if(aDofs.size() != aValues.size()) */
    /*     { */
    /*         std::ostringstream tError; */
    /*         tError << "DIMENSION MISMATCH: THE NUMBER OF ELEMENTS IN INPUT DOFS AND VALUES ARRAY DO NOT MATCH." */
    /*             << "DOFS SIZE = " << aDofs.size() << " AND VALUES SIZE = " << aValues.size(); */
    /*         THROWERR(tError.str()) */
    /*     } */
    /*     mBcDofs = aDofs; */
    /*     mBcValues = aValues; */
    /* } */

private:
    /******************************************************************************//**
     * \brief Initialize member data
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList& aProblemParams)
    {
        auto tName = aProblemParams.get<std::string>("PDE Constraint");
        mPDE = std::make_shared<Plato::Helmholtz::VectorFunction<PhysicsT>>(mSpatialModel, mDataMap, aProblemParams, tName);
    }

    /******************************************************************************/ /**
    * \brief Return solution database.
    * \return solution database
    **********************************************************************************/
    Plato::Solutions getSolution() const
    {
        Plato::Solutions tSolution(mPhysics, mPDEType);
        tSolution.set("State", mStates);
        return tSolution;
    }
};
// class Problem

} // namespace Helmholtz

} // namespace Plato

#include "Helmholtz.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<1>>;
#endif
#ifdef PLATOANALYZE_2D
extern template class Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<2>>;
#endif
#ifdef PLATOANALYZE_3D
extern template class Plato::Helmholtz::Problem<::Plato::HelmholtzFilter<3>>;
#endif


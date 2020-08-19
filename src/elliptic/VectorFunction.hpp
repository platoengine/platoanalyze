#ifndef ELLIPTIC_VECTOR_FUNCTION_HPP
#define ELLIPTIC_VECTOR_FUNCTION_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "WorksetBase.hpp"
#include "NaturalBCs.hpp"
#include "elliptic/AbstractVectorFunction.hpp"
#include "elliptic/EllipticSimplexFadTypes.hpp"

namespace Plato
{

namespace Elliptic
{

/******************************************************************************/
/*! constraint class

   This class takes as a template argument a vector function in the form:

   and manages the evaluation of the function and derivatives wrt state
   and control.
  
*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunction : public Plato::WorksetBase<PhysicsT>
{
  private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode;
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims;
    using Plato::WorksetBase<PhysicsT>::mNumControl;
    using Plato::WorksetBase<PhysicsT>::mNumNodes;
    using Plato::WorksetBase<PhysicsT>::mNumCells;

    using Plato::WorksetBase<PhysicsT>::mGlobalStateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal;

    using Residual  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using Jacobian  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian;
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    using ResidualFunction  = std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<Residual>>;
    using JacobianFunction  = std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<Jacobian>>;
    using GradientXFunction = std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Elliptic::AbstractVectorFunction<GradientZ>>;

    std::shared_ptr<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>> mBoundaryLoads;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;

    std::map<std::string, ResidualFunction>  mResidualFunctions;
    std::map<std::string, JacobianFunction>  mJacobianFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap      & mDataMap;

  public:

    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap problem-specific data map
    * \param [in] aProblemParams Teuchos parameter list with input data
    * \param [in] aProblemType problem type
    *
    ******************************************************************************/
    VectorFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aProblemType
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mBoundaryLoads (nullptr),
        mSpatialModel  (aSpatialModel),
        mDataMap       (aDataMap)
    {
        typename PhysicsT::FunctionFactory tFunctionFactory;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
          auto tName = tDomain.getDomainName();
          mResidualFunctions [tName] = tFunctionFactory.template createVectorFunction<Residual> (tDomain, aDataMap, aProblemParams, aProblemType);
          mJacobianFunctions [tName] = tFunctionFactory.template createVectorFunction<Jacobian> (tDomain, aDataMap, aProblemParams, aProblemType);
          mGradientZFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientZ>(tDomain, aDataMap, aProblemParams, aProblemType);
          mGradientXFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientX>(tDomain, aDataMap, aProblemParams, aProblemType);
        }

        // parse boundary Conditions
        // 
        if(aProblemParams.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<mNumSpatialDims, mNumDofsPerNode>>(aProblemParams.sublist("Natural Boundary Conditions"));
        }
    }

    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aMesh mesh data base
    * \param [in] aDataMap problem-specific data map
    ******************************************************************************/
    VectorFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap)
    {
    }

    /**************************************************************************//**
    * \brief Return number of nodes on the mesh
    * \return number of nodes
    ******************************************************************************/
    Plato::OrdinalType numNodes() const
    {
        return (mNumNodes);
    }

    /**************************************************************************//**
    * \brief Return number of elements/cells on the mesh
    * \return number of elements
    ******************************************************************************/
    Plato::OrdinalType numCells() const
    {
        return (mNumCells);
    }

    /**************************************************************************//**
    * \brief Return total number of global degrees of freedom
    * \return total number of global degrees of freedom
    ******************************************************************************/
    Plato::OrdinalType numDofsPerCell() const
    {
        return (mNumDofsPerCell);
    }

    /**************************************************************************//**
    * \brief Return total number of nodes per cell/element
    * \return total number of nodes per cell/element
    ******************************************************************************/
    Plato::OrdinalType numNodesPerCell() const
    {
        return (mNumNodesPerCell);
    }

    /**************************************************************************//**
    * \brief Return number of degrees of freedom per node
    * \return number of degrees of freedom per node
    ******************************************************************************/
    Plato::OrdinalType numDofsPerNode() const
    {
        return (mNumDofsPerNode);
    }

    /**************************************************************************//**
    * \brief Return number of control vectors/fields, e.g. number of materials.
    * \return number of control vectors
    ******************************************************************************/
    Plato::OrdinalType numControlsPerNode() const
    {
        return (mNumControl);
    }

    /**************************************************************************//**
    *
    * \brief Allocate residual evaluator
    * \param [in] aResidual residual evaluator
    * \param [in] aJacobian Jacobian evaluator
    * \param [in] aName Name of the mesh domain
    *
    ******************************************************************************/
    void
    setEvaluator(
        const ResidualFunction & aResidual,
        const JacobianFunction & aJacobian,
              std::string        aName
    )
    {
        mResidualFunctions[aName] = aResidual;
        mJacobianFunctions[aName] = aJacobian;
    }

    /**************************************************************************//**
    *
    * \brief Allocate partial derivative with respect to control evaluator
    * \param [in] aGradientZ partial derivative with respect to control evaluator
    * \param [in] aName Name of the mesh domain
    *
    ******************************************************************************/
    void
    setEvaluator(
        const GradientZFunction & aGradientZ,
              std::string         aName
    )
    {
        mGradientZFunctions[aName] = aGradientZ; 
    }

    /**************************************************************************//**
    *
    * \brief Allocate partial derivative with respect to configuration evaluator
    * \param [in] GradientX partial derivative with respect to configuration evaluator
    * \param [in] aName Name of the mesh domain
    *
    ******************************************************************************/
    void
    setEvaluator(
        const GradientXFunction & aGradientX,
              std::string         aName
    )
    {
        mGradientXFunctions[aName] = aGradientX; 
    }

    /**************************************************************************//**
    *
    * \brief Return local number of degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return mNumNodes*mNumDofsPerNode;
    }

    /**************************************************************************/
    Plato::ScalarVector
    value(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    )
    {
        using ConfigScalar  = typename Residual::ConfigScalarType;
        using StateScalar   = typename Residual::StateScalarType;
        using ControlScalar = typename Residual::ControlScalarType;
        using ResultScalar  = typename Residual::ResultScalarType;


        Plato::ScalarVector  tReturnValue("Assembled Residual",mNumDofsPerNode*mNumNodes);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mResidualFunctions[tName]->evaluate( tStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

            // create and assemble to return view
            //
            Plato::WorksetBase<PhysicsT>::assembleResidual( tResidual, tReturnValue, tDomain );

        }

        if( mBoundaryLoads != nullptr )
        {
            auto tNumCells = mSpatialModel.Mesh.nelems();

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoads->get(mSpatialModel, tStateWS, tControlWS, tConfigWS, tResidual, /*Scale=*/-1.0 );

            // create and assemble to return view
            //
            Plato::WorksetBase<PhysicsT>::assembleResidual( tResidual, tReturnValue);

        }

        return tReturnValue;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    )
    {
        using ConfigScalar  = typename GradientX::ConfigScalarType;
        using StateScalar   = typename GradientX::StateScalarType;
        using ControlScalar = typename GradientX::ControlScalarType;
        using ResultScalar  = typename GradientX::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumDofsPerNode>(&tMesh);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientXFunctions[tName]->evaluate(tStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode>
                tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);

        }

        if( mBoundaryLoads != nullptr )
        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoads->get(mSpatialModel, tStateWS, tControlWS, tConfigWS, tJacobian, /*Scale=*/1.0 );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode>
                tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }

        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u_T(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    )
    {
        using ConfigScalar  = typename Jacobian::ConfigScalarType;
        using StateScalar   = typename Jacobian::StateScalarType;
        using ControlScalar = typename Jacobian::ControlScalarType;
        using ResultScalar  = typename Jacobian::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( &tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianFunctions[tName]->evaluate( tStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixTransposeEntryOrdinal<mNumSpatialDims, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        if( mBoundaryLoads != nullptr )
        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoads->get(mSpatialModel, tStateWS, tControlWS, tConfigWS, tJacobian, /*Scale=*/1.0 );

            // assembly to return matrix
            Plato::BlockMatrixTransposeEntryOrdinal<mNumSpatialDims, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }

        return tJacobianMat;
    }
    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    )
    /**************************************************************************/
    {
        using ConfigScalar  = typename Jacobian::ConfigScalarType;
        using StateScalar   = typename Jacobian::StateScalarType;
        using ControlScalar = typename Jacobian::ControlScalarType;
        using ResultScalar  = typename Jacobian::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( &tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianFunctions[tName]->evaluate( tStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        if( mBoundaryLoads != nullptr )
        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoads->get(mSpatialModel, tStateWS, tControlWS, tConfigWS, tJacobian, /*Scale=*/1.0 );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    )
    {
        using ConfigScalar  = typename GradientZ::ConfigScalarType;
        using StateScalar   = typename GradientZ::StateScalarType;
        using ControlScalar = typename GradientZ::ControlScalarType;
        using ResultScalar  = typename GradientZ::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumDofsPerNode>( &tMesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar>
                tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);
 
            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS, tDomain);

            // create result 
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", tNumCells, mNumDofsPerCell);

            // evaluate function 
            //
            mGradientZFunctions[tName]->evaluate( tStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
              tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        if( mBoundaryLoads != nullptr )
        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar>
                tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);
 
            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

            // create result 
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", mNumCells, mNumDofsPerCell);

            // evaluate function 
            //
            mBoundaryLoads->get(mSpatialModel, tStateWS, tControlWS, tConfigWS, tJacobian, /*Scale=*/1.0 );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
              tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }
};
// class VectorFunction

} // namespace Elliptic

} // namespace Plato

#endif

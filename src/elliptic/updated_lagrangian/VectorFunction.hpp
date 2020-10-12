#pragma once

#include <memory>

#include "WorksetBase.hpp"
#include "NaturalBCs.hpp"
#include "elliptic/updated_lagrangian/AbstractVectorFunction.hpp"
#include "elliptic/updated_lagrangian/EllipticUpLagSimplexFadTypes.hpp"

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

/******************************************************************************/
/*! VectorFunction class

   This class defines a vector function of the form:

   \f[
     F(\phi, U^k, c^{k-1})
   \f]

   where \f$ \phi \f$ is the control, \f$ U^k \f$ is the nodal state at the
   current step, and \f$ c^{k-1} \f$ is the element state at the previous step.

   This class is intended for use in an updated lagrangian formulation where
   the reference configuration is updated by the displacement at each step.
  
*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunction : public Plato::WorksetBase<PhysicsT>
{
  private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumLocalDofsPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode;
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims;
    using Plato::WorksetBase<PhysicsT>::mNumControl;
    using Plato::WorksetBase<PhysicsT>::mNumNodes;
    using Plato::WorksetBase<PhysicsT>::mNumCells;

    using Plato::WorksetBase<PhysicsT>::mGlobalStateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal;

    using Residual  = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using Jacobian  = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<typename PhysicsT::SimplexT>::Jacobian;
    using GradientC = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<typename PhysicsT::SimplexT>::GradientC;
    using GradientX = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    using ResidualFunction  = std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractVectorFunction<Residual>>;
    using JacobianFunction  = std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractVectorFunction<Jacobian>>;
    using GradientCFunction = std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractVectorFunction<GradientC>>;
    using GradientXFunction = std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractVectorFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Elliptic::UpdatedLagrangian::AbstractVectorFunction<GradientZ>>;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;

    std::map<std::string, ResidualFunction>  mResidualFunctions;
    std::map<std::string, JacobianFunction>  mJacobianFunctions;
    std::map<std::string, GradientCFunction> mGradientCFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap & mDataMap;

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
        mSpatialModel  (aSpatialModel),
        mDataMap       (aDataMap)
    {
        typename PhysicsT::FunctionFactory tFunctionFactory;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
          auto tName = tDomain.getDomainName();
          mResidualFunctions [tName] = tFunctionFactory.template createVectorFunction<Residual> (tDomain, aDataMap, aProblemParams, aProblemType);
          mJacobianFunctions [tName] = tFunctionFactory.template createVectorFunction<Jacobian> (tDomain, aDataMap, aProblemParams, aProblemType);
          mGradientCFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientC>(tDomain, aDataMap, aProblemParams, aProblemType);
          mGradientZFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientZ>(tDomain, aDataMap, aProblemParams, aProblemType);
          mGradientXFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientX>(tDomain, aDataMap, aProblemParams, aProblemType);
        }
    }

    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap problem-specific data map
    ******************************************************************************/
    VectorFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel  (aSpatialModel),
        mDataMap       (aDataMap)
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
    * \param [in] aGradientC partial derivative with respect to control evaluator
    * \param [in] aName Name of the mesh domain
    *
    ******************************************************************************/
    void
    setEvaluator(
        const GradientCFunction & aGradientC,
              std::string         aName
    )
    {
        mGradientCFunctions[aName] = aGradientC; 
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
    * \brief Return number of global state degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return mNumNodes*mNumDofsPerNode;
    }

    /**************************************************************************//**
    *
    * \brief Return number of local state degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType stateSize() const
    {
      return mNumCells*mNumLocalDofsPerCell;
    }

    /**************************************************************************/
    Plato::ScalarVector
    value(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename Residual::ConfigScalarType;
        using GlobalStateScalar = typename Residual::GlobalStateScalarType;
        using LocalStateScalar  = typename Residual::LocalStateScalarType;
        using ControlScalar     = typename Residual::ControlScalarType;
        using ResultScalar      = typename Residual::ResultScalarType;


        Plato::ScalarVector  tReturnValue("Assembled Residual", mNumDofsPerNode*mNumNodes);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

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
            mResidualFunctions.at(tName)->evaluate( tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

            // assemble to return view
            //
            Plato::WorksetBase<PhysicsT>::assembleResidual( tResidual, tReturnValue, tDomain );

        }

        {
            auto tNumCells = mSpatialModel.Mesh.nelems();

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS);

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
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mResidualFunctions.at(tFirstBlockName)->evaluate_boundary
                (mSpatialModel, tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

            // create and assemble to return view
            //
            Plato::WorksetBase<PhysicsT>::assembleResidual( tResidual, tReturnValue);
        }

        return tReturnValue;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename GradientX::ConfigScalarType;
        using GlobalStateScalar = typename GradientX::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientX::LocalStateScalarType;
        using ControlScalar     = typename GradientX::ControlScalarType;
        using ResultScalar      = typename GradientX::ResultScalarType;

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

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientXFunctions.at(tName)->evaluate(tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode>
                tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);

        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mGradientXFunctions.at(tFirstBlockName)->evaluate_boundary
                (mSpatialModel, tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

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
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename Jacobian::ConfigScalarType;
        using GlobalStateScalar = typename Jacobian::GlobalStateScalarType;
        using LocalStateScalar  = typename Jacobian::LocalStateScalarType;
        using ControlScalar     = typename Jacobian::ControlScalarType;
        using ResultScalar      = typename Jacobian::ResultScalarType;

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

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianFunctions.at(tName)->evaluate( tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixTransposeEntryOrdinal<mNumSpatialDims, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mJacobianFunctions.at(tFirstBlockName)->evaluate_boundary
                (mSpatialModel, tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

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
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        using ConfigScalar      = typename Jacobian::ConfigScalarType;
        using GlobalStateScalar = typename Jacobian::GlobalStateScalarType;
        using LocalStateScalar  = typename Jacobian::LocalStateScalarType;
        using ControlScalar     = typename Jacobian::ControlScalarType;
        using ResultScalar      = typename Jacobian::ResultScalarType;

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

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mJacobianFunctions.at(tName)->evaluate( tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mJacobianFunctions.at(tFirstBlockName)->evaluate_boundary
                (mSpatialModel, tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

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
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename GradientZ::ConfigScalarType;
        using GlobalStateScalar = typename GradientZ::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientZ::LocalStateScalarType;
        using ControlScalar     = typename GradientZ::ControlScalarType;
        using ResultScalar      = typename GradientZ::ResultScalarType;

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
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);
 
            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // create result 
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", tNumCells, mNumDofsPerCell);

            // evaluate function 
            //
            mGradientZFunctions.at(tName)->evaluate( tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
              tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

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
 
            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", mNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS);

            // create result 
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", mNumCells, mNumDofsPerCell);

            // evaluate function 
            //
            auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
            mGradientZFunctions.at(tFirstBlockName)->evaluate_boundary
                (mSpatialModel, tGlobalStateWS, tLocalStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

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

} // namespace UpdatedLagrangian

} // namespace Elliptic

} // namespace Plato

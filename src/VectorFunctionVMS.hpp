#ifndef VECTOR_FUNCTION_VMS_HPP
#define VECTOR_FUNCTION_VMS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "WorksetBase.hpp"
#include "AbstractVectorFunctionVMS.hpp"
#include "SimplexFadTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Stabilized Partial Differential Equation (PDE) constraint workset manager

   This class takes as a template argument a vector function in the form:

   and manages the evaluation of the function and Jacobians wrt state, node
   state, control, and configuration.

   NOTES:
   1. The use case will define the node state: a) If the stabilized
      mechanics residual is used, the node state is denoted by the projected
      pressure gradient. 2) If the projected gradient residual is used, the
      node state is denoted by the projected pressure field.
   2. The use case will define the state: a) If the stabilized mechanics
      residual is used, the states are displacement+pressure. 2) If the
      projected gradient residual is used, the state is denoted by the
      projected pressure gradient.
*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunctionVMS
{
private:
    static constexpr auto mNumControl      = PhysicsT::mNumControl;      /*!< number of control fields, e.g. number of material fields */
    static constexpr auto mNumSpatialDims  = PhysicsT::mNumSpatialDims;  /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell = PhysicsT::mNumNodesPerCell; /*!< number of nodes, i.e. vertices, per cell, i.e. element */

    static constexpr auto mNumStateDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of global state degrees of freedom per node */
    static constexpr auto mNumStateDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of global state degrees of freedom per cell */
    static constexpr auto mNumNodeStateDofsPerNode = PhysicsT::mNumNodeStatePerNode; /*!< number of node state degrees of freedom per node */
    static constexpr auto mNumNodeStateDofsPerCell = PhysicsT::mNumNodeStatePerCell; /*!< number of node state degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell; /*!< number of configuration degrees of freedom per cell */

    const Plato::OrdinalType mNumNodes; /*!< total number of nodes */
    const Plato::OrdinalType mNumCells; /*!< total number of cells (i.e. elements)*/

    using Residual  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using Jacobian  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian;
    using JacobianN = typename Plato::Evaluation<typename PhysicsT::SimplexT>::JacobianN;
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    using ResidualFunction  = std::shared_ptr<Plato::AbstractVectorFunctionVMS<Residual>>;
    using JacobianUFunction = std::shared_ptr<Plato::AbstractVectorFunctionVMS<Jacobian>>;
    using JacobianNFunction = std::shared_ptr<Plato::AbstractVectorFunctionVMS<JacobianN>>;
    using JacobianXFunction = std::shared_ptr<Plato::AbstractVectorFunctionVMS<GradientX>>;
    using JacobianZFunction = std::shared_ptr<Plato::AbstractVectorFunctionVMS<GradientZ>>;

    std::map<std::string, ResidualFunction>  mResidualFunctions;
    std::map<std::string, JacobianUFunction> mJacobianUFunctions;
    std::map<std::string, JacobianNFunction> mJacobianNFunctions;
    std::map<std::string, JacobianXFunction> mJacobianXFunctions;
    std::map<std::string, JacobianZFunction> mJacobianZFunctions;

    ResidualFunction  mBoundaryLoadsResidualFunction;
    JacobianUFunction mBoundaryLoadsJacobianUFunction;
    JacobianNFunction mBoundaryLoadsJacobianNFunction;
    JacobianXFunction mBoundaryLoadsJacobianXFunction;
    JacobianZFunction mBoundaryLoadsJacobianZFunction;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap; /*!< output data map */
    Plato::WorksetBase<PhysicsT> mWorksetBase; /*!< assembly routine interface */

public:
    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aMesh mesh data base
    * \param [in] aMeshSets mesh sets data base
    * \param [in] aDataMap problem-specific data map
    * \param [in] aParamList Teuchos parameter list with input data
    * \param [in] aProblemType problem type
    *
    ******************************************************************************/
    VectorFunctionVMS(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
        const std::string            & aProblemType
    ) :
        mNumNodes     (aSpatialModel.Mesh.nverts()),
        mNumCells     (aSpatialModel.Mesh.nelems()),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mWorksetBase  (aSpatialModel.Mesh)
    {
        typename PhysicsT::FunctionFactory tFunctionFactory;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mResidualFunctions[tName]  = tFunctionFactory.template createVectorFunction<Residual> (tDomain, aDataMap, aParamList, aProblemType);
            mJacobianUFunctions[tName] = tFunctionFactory.template createVectorFunction<Jacobian> (tDomain, aDataMap, aParamList, aProblemType);
            mJacobianNFunctions[tName] = tFunctionFactory.template createVectorFunction<JacobianN>(tDomain, aDataMap, aParamList, aProblemType);
            mJacobianZFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientZ>(tDomain, aDataMap, aParamList, aProblemType);
            mJacobianXFunctions[tName] = tFunctionFactory.template createVectorFunction<GradientX>(tDomain, aDataMap, aParamList, aProblemType);
        }

        // any block can compute the boundary terms for the entire mesh.  We'll use the first block.
        auto tFirstBlockName = aSpatialModel.Domains.front().getDomainName();

        mBoundaryLoadsResidualFunction  = mResidualFunctions[tFirstBlockName];
        mBoundaryLoadsJacobianUFunction = mJacobianUFunctions[tFirstBlockName];
        mBoundaryLoadsJacobianNFunction = mJacobianNFunctions[tFirstBlockName];
        mBoundaryLoadsJacobianZFunction = mJacobianZFunctions[tFirstBlockName];
        mBoundaryLoadsJacobianXFunction = mJacobianXFunctions[tFirstBlockName];
    }


    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aMesh mesh data base
    * \param [in] aDataMap problem-specific data map
    *
    ******************************************************************************/
    VectorFunctionVMS(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            mDataMap(aDataMap),
            mWorksetBase(aMesh)
    {
    }

    /***************************************************************************//**
     * \brief Return reference to Omega_h mesh database
     * \return Omega_h mesh database
    *******************************************************************************/
    Omega_h::Mesh& getMesh() const
    {
        return (mSpatialModel.Mesh);
    }

    /**************************************************************************//**
    *
    * \brief Return number of degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return (mNumNodes * mNumStateDofsPerNode);
    }

    /**************************************************************************//**
     *
     * \brief Return total number of nodes
     * \return total number of nodes
     *
     ******************************************************************************/
    decltype(mNumNodes) numNodes() const
    {
        return mNumNodes;
    }

    /**************************************************************************//**
     *
     * \brief Return total number of cells
     * \return total number of cells
     *
     ******************************************************************************/
    decltype(mNumCells) numCells() const
    {
        return mNumCells;
    }

    /***********************************************************************//**
     * \brief Return number of spatial dimensions.
     * \return number of spatial dimensions
    ***************************************************************************/
    decltype(mNumSpatialDims) numSpatialDims() const
    {
        return mNumSpatialDims;
    }

    /***********************************************************************//**
     * \brief Return number of nodes per cell.
     * \return number of nodes per cell
    ***************************************************************************/
    decltype(mNumNodesPerCell) numNodesPerCell() const
    {
        return mNumNodesPerCell;
    }

    /***********************************************************************//**
     * \brief Return number of projected pressure gradient degrees of freedom per node.
     * \return number of projected pressure gradient degrees of freedom per node
    ***************************************************************************/
    decltype(mNumStateDofsPerNode) numDofsPerNode() const
    {
        return mNumStateDofsPerNode;
    }

    /***********************************************************************//**
     * \brief Return number of projected pressure gradient degrees of freedom per cell.
     * \return number of projected pressure gradient degrees of freedom per cell
    ***************************************************************************/
    decltype(mNumStateDofsPerCell) numDofsPerCell() const
    {
        return mNumStateDofsPerCell;
    }

    /***********************************************************************//**
     * \brief Return number of pressure degrees of freedom per node.
     * \return number of pressure degrees of freedom per node
    ***************************************************************************/
    decltype(mNumNodeStateDofsPerNode) numNodeStatePerNode() const
    {
        return mNumNodeStateDofsPerNode;
    }

    /***********************************************************************//**
     * \brief Return number of pressure degrees of freedom per cell.
     * \return number of pressure degrees of freedom per cell
    ***************************************************************************/
    decltype(mNumNodeStateDofsPerCell) numNodeStatePerCell() const
    {
        return mNumNodeStateDofsPerCell;
    }

    /***********************************************************************//**
     * \brief Return number of configuration degrees of freedom per cell.
     * \return number of configuration degrees of freedom per cell
    ***************************************************************************/
    decltype(mNumConfigDofsPerCell) numConfigDofsPerCell() const
    {
        return mNumConfigDofsPerCell;
    }

    /***************************************************************************//**
     * \brief Evaluate and assemble residual
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled residual
    *******************************************************************************/
    Plato::ScalarVector
    value(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename Residual::ConfigScalarType;
        using StateScalar     = typename Residual::StateScalarType;
        using NodeStateScalar = typename Residual::NodeStateScalarType;
        using ControlScalar   = typename Residual::ControlScalarType;
        using ResultScalar    = typename Residual::ResultScalarType;

        Plato::ScalarVector tReturnValue("Assembled Residual", mNumStateDofsPerNode * mNumNodes);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual", tNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mResidualFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

            // assemble to return view
            //
            mWorksetBase.assembleResidual(tResidual, tReturnValue, tDomain);
        }

        {
            auto tNumCells = mSpatialModel.Mesh.nelems();

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual", tNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsResidualFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

            // assemble to return view
            //
            mWorksetBase.assembleResidual(tResidual, tReturnValue);
        }

        return tReturnValue;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to configuration degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename GradientX::ConfigScalarType;
        using StateScalar     = typename GradientX::StateScalarType;
        using NodeStateScalar = typename GradientX::NodeStateScalarType;
        using ControlScalar   = typename GradientX::ControlScalarType;
        using ResultScalar    = typename GradientX::ResultScalarType;

        // Allocate Jacobian
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumStateDofsPerNode>(&tMesh);

        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumStateDofsPerNode> tMatEntryOrdinal(tJacobianMat, &tMesh);

        auto tMatEntries = tJacobianMat->entries();

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Configuration", tNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mJacobianXFunctions.at(tName)->evaluate(tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

            // Assemble Jacobian
            //
            mWorksetBase.assembleTransposeJacobian(mNumStateDofsPerCell, mNumConfigDofsPerCell, tMatEntryOrdinal, tJacobian, tMatEntries, tDomain);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Configuration", mNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianXFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

            // Assemble Jacobian
            //
            mWorksetBase.assembleTransposeJacobian(mNumStateDofsPerCell, mNumConfigDofsPerCell, tMatEntryOrdinal, tJacobian, tMatEntries);
        }

        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to configuration degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return Workset of Jacobian with respect to configuration degrees of freedom
    *******************************************************************************/
    Plato::ScalarArray3D
    gradient_x_workset(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename GradientX::ConfigScalarType;
        using StateScalar     = typename GradientX::StateScalarType;
        using NodeStateScalar = typename GradientX::NodeStateScalarType;
        using ControlScalar   = typename GradientX::ControlScalarType;
        using ResultScalar    = typename GradientX::ResultScalarType;

        Plato::ScalarArray3D tOutputJacobian("Jacobian WRT Configuration", mNumCells, mNumStateDofsPerCell, mNumConfigDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Configuration", tNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mJacobianXFunctions.at(tName)->evaluate(tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

            // assemble
            //
            Plato::transform_ad_type_to_pod_3Dview<mNumStateDofsPerCell, mNumConfigDofsPerCell>(tDomain, tJacobianWS, tOutputJacobian);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Configuration", mNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianXFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep);

            // assemble
            //
            Plato::transform_ad_type_to_pod_3Dview<mNumStateDofsPerCell, mNumConfigDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        }

        return tOutputJacobian;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to state degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian transpose
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u_T(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename Jacobian::ConfigScalarType;
        using StateScalar     = typename Jacobian::StateScalarType;
        using NodeStateScalar = typename Jacobian::NodeStateScalarType;
        using ControlScalar   = typename Jacobian::ControlScalarType;
        using ResultScalar    = typename Jacobian::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumStateDofsPerNode, mNumStateDofsPerNode>( &tMesh );

        Plato::BlockMatrixTransposeEntryOrdinal<mNumSpatialDims, mNumStateDofsPerNode> tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

        auto tJacobianMatEntries = tJacobianMat->entries();

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian State", tNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mJacobianUFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            mWorksetBase.assembleJacobianFad
                (mNumStateDofsPerCell, mNumStateDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian State", mNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianUFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            mWorksetBase.assembleJacobianFad
                (mNumStateDofsPerCell, mNumStateDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }

        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to state degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename Jacobian::ConfigScalarType;
        using StateScalar     = typename Jacobian::StateScalarType;
        using NodeStateScalar = typename Jacobian::NodeStateScalarType;
        using ControlScalar   = typename Jacobian::ControlScalarType;
        using ResultScalar    = typename Jacobian::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumStateDofsPerNode, mNumStateDofsPerNode>( &tMesh );

        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumStateDofsPerNode> tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

        auto tJacobianMatEntries = tJacobianMat->entries();

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian State", tNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mJacobianUFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assemble to return matrix
            //
            mWorksetBase.assembleJacobianFad
                (mNumStateDofsPerCell, mNumStateDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries, tDomain);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian State", mNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianUFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assemble to return matrix
            //
            mWorksetBase.assembleJacobianFad
                (mNumStateDofsPerCell, mNumStateDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to node state degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return workset of Jacobian with respect to node state degrees of freedom
    *******************************************************************************/
    Plato::ScalarArray3D
    gradient_n_workset(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename JacobianN::ConfigScalarType;
        using StateScalar     = typename JacobianN::StateScalarType;
        using NodeStateScalar = typename JacobianN::NodeStateScalarType;
        using ControlScalar   = typename JacobianN::ControlScalarType;
        using ResultScalar    = typename JacobianN::ResultScalarType;

        // create return array
        //
        Plato::ScalarArray3D tOutJacobian("POD Jacobian Node State", mNumCells, mNumStateDofsPerCell, mNumNodeStateDofsPerNode);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Node State", tNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mJacobianNFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep );

            Plato::transform_ad_type_to_pod_3Dview<mNumStateDofsPerCell, mNumNodeStateDofsPerNode>(tDomain, tJacobianWS, tOutJacobian);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Node State", mNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianNFunction->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep );

            Plato::transform_ad_type_to_pod_3Dview<mNumStateDofsPerCell, mNumNodeStateDofsPerNode>(mNumCells, tJacobianWS, tOutJacobian);
        }
        return (tOutJacobian);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to node state degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_n(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename JacobianN::ConfigScalarType;
        using StateScalar     = typename JacobianN::StateScalarType;
        using NodeStateScalar = typename JacobianN::NodeStateScalarType;
        using ControlScalar   = typename JacobianN::ControlScalarType;
        using ResultScalar    = typename JacobianN::ResultScalarType;

        // tJacobian has shape (Nc, (Nv x Nd), (Nv x Nn))
        //   Nc: number of cells
        //   Nv: number of vertices per cell
        //   Nd: dimensionality of the vector function
        //   Nn: dimensionality of the node state
        //   (I x J) is a strided 1D array indexed by i*J+j where i /in I and j /in J.
        //
        // (tJacobian is (Nc, (Nv x Nd)) and the third dimension, (Nv x Nn), is in the AD type)

        // create matrix with block size (Nd, Nn).
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumNodeStateDofsPerNode>( &tMesh );

        // create entry ordinal functor:
        // tJacobianMatEntryOrdinal(e, k, l) => G
        //   e: cell index
        //   k: row index /in (Nv x Nd)
        //   l: col index /in (Nv x Nn)
        //   G: entry index into CRS matrix
        //
        // Template parameters:
        //   mNumSpatialDims: Nv-1
        //   mNumSpatialDims: Nd
        //   mNumNodeStatePerNode:   Nn
        //
        // Note that the second two template parameters must match the block shape of the destination matrix, tJacobianMat
        //
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumNodeStateDofsPerNode>
            tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

        // Assemble from the AD-typed result, tJacobian, into the POD-typed global matrix, tJacobianMat.
        //
        // Arguments 1 and 2 below correspond to the size of tJacobian ((Nv x Nd), (Nv x Nn)) and the size of
        // tJacobianMat (Nd, Nn).
        //
        auto tJacobianMatEntries = tJacobianMat->entries();

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Node State", tNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mJacobianNFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            //
            mWorksetBase.assembleJacobianFad(
              mNumStateDofsPerCell,     /* (Nv x Nd) */
              mNumNodeStateDofsPerCell, /* (Nv x Nn) */
              tJacobianMatEntryOrdinal, /* entry ordinal functor */
              tJacobian,                /* source data */
              tJacobianMatEntries,      /* destination */
              tDomain
            );
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Node State", mNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianNFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            // assembly to return matrix
            //
            mWorksetBase.assembleJacobianFad(
              mNumStateDofsPerCell,     /* (Nv x Nd) */
              mNumNodeStateDofsPerCell, /* (Nv x Nn) */
              tJacobianMatEntryOrdinal, /* entry ordinal functor */
              tJacobian,                /* source data */
              tJacobianMatEntries       /* destination */
            );
        }

        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to node state degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian transpose
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_n_T(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aNodeState,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename JacobianN::ConfigScalarType;
        using StateScalar     = typename JacobianN::StateScalarType;
        using NodeStateScalar = typename JacobianN::NodeStateScalarType;
        using ControlScalar   = typename JacobianN::ControlScalarType;
        using ResultScalar    = typename JacobianN::ResultScalarType;

        // tJacobian has shape (Nc, (Nv x Nd), (Nv x Nn))
        //   Nc: number of cells
        //   Nv: number of vertices per cell
        //   Nd: dimensionality of the vector function
        //   Nn: dimensionality of the node state
        //   (I x J) is a strided 1D array indexed by i*J+j where i /in I and j /in J.
        //
        // (tJacobian is (Nc, (Nv x Nd)) and the third dimension, (Nv x Nn), is in the AD type)

        // create *transpose* matrix with block size (Nn, Nd).
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumNodeStateDofsPerNode, mNumStateDofsPerNode>( &tMesh );

        // create entry ordinal functor:
        // tJacobianMatEntryOrdinal(e, k, l) => G
        //   e: cell index
        //   k: row index /in (Nv x Nd)
        //   l: col index /in (Nv x Nn)
        //   G: entry index into CRS matrix
        //
        // Template parameters:
        //   mNumSpatialDims: Nv-1
        //   mNumNodeStatePerNode:   Nn
        //   mNumDofsPerNode: Nd
        //
        // Note that the second two template parameters must match the block shape of the destination matrix, tJacobianMat
        //
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumNodeStateDofsPerNode, mNumStateDofsPerNode>
            tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

        // Assemble from the AD-typed result, tJacobian, into the POD-typed global matrix, tJacobianMat.
        //
        // The transpose is being assembled, (i.e., tJacobian is transposed before assembly into tJacobianMat), so
        // arguments 1 and 2 below correspond to the size of tJacobian ((Nv x Nd), (Nv x Nn)) and the size of the
        // *transpose* of tJacobianMat (Transpose(Nn, Nd) => (Nd, Nn)).
        //
        auto tJacobianMatEntries = tJacobianMat->entries();

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Node State", tNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mJacobianNFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            mWorksetBase.assembleTransposeJacobian(
                mNumStateDofsPerCell,     /* (Nv x Nd) */
                mNumNodeStateDofsPerCell, /* (Nv x Nn) */
                tJacobianMatEntryOrdinal, /* entry ordinal functor */
                tJacobian,                /* source data */
                tJacobianMatEntries,      /* destination */
                tDomain
            );
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Node State", mNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianNFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            mWorksetBase.assembleTransposeJacobian(
                mNumStateDofsPerCell,     /* (Nv x Nd) */
                mNumNodeStateDofsPerCell, /* (Nv x Nn) */
                tJacobianMatEntryOrdinal, /* entry ordinal functor */
                tJacobian,                /* source data */
                tJacobianMatEntries       /* destination */
            );
        }
        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to control degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(
        const Plato::ScalarVectorT<Plato::Scalar> & aState,
        const Plato::ScalarVectorT<Plato::Scalar> & aNodeState,
        const Plato::ScalarVectorT<Plato::Scalar> & aControl,
              Plato::Scalar                         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename GradientZ::ConfigScalarType;
        using StateScalar     = typename GradientZ::StateScalarType;
        using NodeStateScalar = typename GradientZ::NodeStateScalarType;
        using ControlScalar   = typename GradientZ::ControlScalarType;
        using ResultScalar    = typename GradientZ::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumStateDofsPerNode>( &tMesh );

        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumStateDofsPerNode> tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

        auto tJacobianMatEntries = tJacobianMat->entries();

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", tNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mJacobianZFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            mWorksetBase.assembleTransposeJacobian(
                mNumStateDofsPerCell,
                mNumNodesPerCell,
                tJacobianMatEntryOrdinal,
                tJacobian,
                tJacobianMatEntries,
                tDomain
            );
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", mNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianZFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

            mWorksetBase.assembleTransposeJacobian(
                mNumStateDofsPerCell,
                mNumNodesPerCell,
                tJacobianMatEntryOrdinal,
                tJacobian,
                tJacobianMatEntries
            );
        }
        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to control degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return Workset of Jacobian with respect to control degrees of freedom
    *******************************************************************************/
    Plato::ScalarArray3D
    gradient_z_workset(
        const Plato::ScalarVectorT<Plato::Scalar> & aState,
        const Plato::ScalarVectorT<Plato::Scalar> & aNodeState,
        const Plato::ScalarVectorT<Plato::Scalar> & aControl,
              Plato::Scalar                         aTimeStep = 0.0
    ) const
    {
        using ConfigScalar    = typename GradientZ::ConfigScalarType;
        using StateScalar     = typename GradientZ::StateScalarType;
        using NodeStateScalar = typename GradientZ::NodeStateScalarType;
        using ControlScalar   = typename GradientZ::ControlScalarType;
        using ResultScalar    = typename GradientZ::ResultScalarType;

        Plato::ScalarArray3D tOutputJacobian("Output Jacobian WRT Control", mNumCells, mNumStateDofsPerCell, mNumNodesPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS, tDomain);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", tNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("JacobianControl", tNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mJacobianZFunctions.at(tName)->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep );

            Plato::transform_ad_type_to_pod_3Dview<mNumStateDofsPerCell, mNumNodesPerCell>(tDomain, tJacobianWS, tOutputJacobian);
        }

        {
            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumStateDofsPerCell);
            mWorksetBase.worksetState(aState, tStateWS);

            // Workset node state
            //
            Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStateDofsPerCell);
            mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("JacobianControl", mNumCells, mNumStateDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsJacobianZFunction->evaluate_boundary(mSpatialModel, tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobianWS, aTimeStep );

            Plato::transform_ad_type_to_pod_3Dview<mNumStateDofsPerCell, mNumNodesPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        }
        return tOutputJacobian;
    }

    /***************************************************************************//**
     * \brief Update physics-based parameters within a frequency of optimization iterations
     * \param [in] aStates     global states for all time steps
     * \param [in] aControls   current controls, i.e. design variables
     * \param [in] aTimeStep   current time step increment
    *******************************************************************************/
    void
    updateProblem(
        const Plato::ScalarMultiVector & aStates,
        const Plato::ScalarVector      & aControls,
              Plato::Scalar              aTimeStep = 0.0
    ) const
    {
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            mResidualFunctions.at(tName)->updateProblem(aStates, aControls, aTimeStep);
            mJacobianUFunctions.at(tName)->updateProblem(aStates, aControls, aTimeStep);
            mJacobianNFunctions.at(tName)->updateProblem(aStates, aControls, aTimeStep);
            mJacobianXFunctions.at(tName)->updateProblem(aStates, aControls, aTimeStep);
            mJacobianZFunctions.at(tName)->updateProblem(aStates, aControls, aTimeStep);
        }

        mBoundaryLoadsResidualFunction->updateProblem(aStates, aControls, aTimeStep);
        mBoundaryLoadsJacobianUFunction->updateProblem(aStates, aControls, aTimeStep);
        mBoundaryLoadsJacobianNFunction->updateProblem(aStates, aControls, aTimeStep);
        mBoundaryLoadsJacobianXFunction->updateProblem(aStates, aControls, aTimeStep);
        mBoundaryLoadsJacobianZFunction->updateProblem(aStates, aControls, aTimeStep);
    }
};
// class VectorFunctionVMS

} // namespace Plato

#endif

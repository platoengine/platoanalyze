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

    using Residual  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;  /*!< automatic differentiation evaluation type */
    using Jacobian  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian;  /*!< automatic differentiation evaluation type */
    using JacobianN = typename Plato::Evaluation<typename PhysicsT::SimplexT>::JacobianN; /*!< automatic differentiation evaluation type */
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX; /*!< automatic differentiation evaluation type */
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ; /*!< automatic differentiation evaluation type */

    std::shared_ptr<Plato::AbstractVectorFunctionVMS<Residual>>  mVectorFunctionVMSResidual;  /*!< interface to cell-level operations */
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<Jacobian>>  mVectorFunctionVMSJacobianU; /*!< interface to cell-level operations */
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<JacobianN>> mVectorFunctionVMSJacobianN; /*!< interface to cell-level operations */
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<GradientX>> mVectorFunctionVMSJacobianX; /*!< interface to cell-level operations */
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<GradientZ>> mVectorFunctionVMSJacobianZ; /*!< interface to cell-level operations */

    Plato::DataMap& mDataMap; /*!< output data map */
    Plato::WorksetBase<PhysicsT> mWorksetBase; /*!< assembly routine interface */

private:
    /***************************************************************************//**
     * \brief Evaluate and residual workset
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return residual workset
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename Residual::ResultScalarType>
    valueWorkset(const Plato::ScalarVector & aState,
                 const Plato::ScalarVector & aNodeState,
                 const Plato::ScalarVector & aControl,
                 const Plato::Scalar & aTimeStep) const
    {
        using ConfigScalar    = typename Residual::ConfigScalarType;
        using StateScalar     = typename Residual::StateScalarType;
        using NodeStateScalar = typename Residual::NodeStateScalarType;
        using ControlScalar   = typename Residual::ControlScalarType;
        using ResultScalar    = typename Residual::ResultScalarType;

        // Workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumStateDofsPerCell);
        mWorksetBase.worksetState(aState, tStateWS);

        // Workset node state
        //
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStateDofsPerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // create result
        //
        Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual",mNumCells, mNumStateDofsPerCell);

        // evaluate function
        //
        mVectorFunctionVMSResidual->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

        return (tResidual);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to configuration degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return Jacobian workset
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename GradientX::ResultScalarType>
    jacobianConfigWorkset(const Plato::ScalarVector & aState,
                          const Plato::ScalarVector & aNodeState,
                          const Plato::ScalarVector & aControl,
                          const Plato::Scalar & aTimeStep) const
    {
        using ConfigScalar    = typename GradientX::ConfigScalarType;
        using StateScalar     = typename GradientX::StateScalarType;
        using NodeStateScalar = typename GradientX::NodeStateScalarType;
        using ControlScalar   = typename GradientX::ControlScalarType;
        using ResultScalar    = typename GradientX::ResultScalarType;

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
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
        mVectorFunctionVMSJacobianX->evaluate(tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

        return (tJacobian);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to state degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return Jacobian workset
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename Jacobian::ResultScalarType>
    jacobianStateWorkset(const Plato::ScalarVector & aState,
                         const Plato::ScalarVector & aNodeState,
                         const Plato::ScalarVector & aControl,
                         const Plato::Scalar & aTimeStep) const
    {
        using ConfigScalar    = typename Jacobian::ConfigScalarType;
        using StateScalar     = typename Jacobian::StateScalarType;
        using NodeStateScalar = typename Jacobian::NodeStateScalarType;
        using ControlScalar   = typename Jacobian::ControlScalarType;
        using ResultScalar    = typename Jacobian::ResultScalarType;

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumStateDofsPerCell);
        mWorksetBase.worksetState(aState, tStateWS);

        // Workset node state
        //
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset",mNumCells, mNumNodeStateDofsPerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // create return view
        //
        Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian State",mNumCells,mNumStateDofsPerCell);

        // evaluate function
        //
        mVectorFunctionVMSJacobianU->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

        return (tJacobian);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to node state degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return Jacobian workset
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename JacobianN::ResultScalarType>
    jacobianNodeStateWorkset(const Plato::ScalarVector & aState,
                             const Plato::ScalarVector & aNodeState,
                             const Plato::ScalarVector & aControl,
                             const Plato::Scalar & aTimeStep) const
    {
        using ConfigScalar    = typename JacobianN::ConfigScalarType;
        using StateScalar     = typename JacobianN::StateScalarType;
        using NodeStateScalar = typename JacobianN::NodeStateScalarType;
        using ControlScalar   = typename JacobianN::ControlScalarType;
        using ResultScalar    = typename JacobianN::ResultScalarType;

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
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
        mVectorFunctionVMSJacobianN->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

        return (tJacobian);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to control degrees of freedom
     * \param [in] aState     projected pressure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return Jacobian workset
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename GradientZ::ResultScalarType>
    jacobianControlWorkset(const Plato::ScalarVector & aState,
                           const Plato::ScalarVector & aNodeState,
                           const Plato::ScalarVector & aControl,
                           const Plato::Scalar & aTimeStep) const
    {
        using ConfigScalar    = typename GradientZ::ConfigScalarType;
        using StateScalar     = typename GradientZ::StateScalarType;
        using NodeStateScalar = typename GradientZ::NodeStateScalarType;
        using ControlScalar   = typename GradientZ::ControlScalarType;
        using ResultScalar    = typename GradientZ::ResultScalarType;

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
        mWorksetBase.worksetControl(aControl, tControlWS);

        // Workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumStateDofsPerCell);
        mWorksetBase.worksetState(aState, tStateWS);

        // Workset node state
        //
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset",mNumCells, mNumNodeStateDofsPerCell);
        mWorksetBase.worksetNodeState(aNodeState, tNodeStateWS);

        // create result
        //
        Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl",mNumCells,mNumStateDofsPerCell);

        // evaluate function
        //
        mVectorFunctionVMSJacobianZ->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

        return (tJacobian);
    }

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
    VectorFunctionVMS(Omega_h::Mesh& aMesh,
                   Omega_h::MeshSets& aMeshSets,
                   Plato::DataMap& aDataMap,
                   Teuchos::ParameterList& aParamList,
                   std::string aProblemType) :
            mNumNodes(aMesh.nverts()),
            mNumCells(aMesh.nelems()),
            mDataMap(aDataMap),
            mWorksetBase(aMesh)
    {
      typename PhysicsT::FunctionFactory tFunctionFactory;

      mVectorFunctionVMSResidual = tFunctionFactory.template createVectorFunctionVMS<Residual>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionVMSJacobianU = tFunctionFactory.template createVectorFunctionVMS<Jacobian>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionVMSJacobianN = tFunctionFactory.template createVectorFunctionVMS<JacobianN>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionVMSJacobianZ = tFunctionFactory.template createVectorFunctionVMS<GradientZ>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);

      mVectorFunctionVMSJacobianX = tFunctionFactory.template createVectorFunctionVMS<GradientX>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);
    }

    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aMesh mesh data base
    * \param [in] aDataMap problem-specific data map
    *
    ******************************************************************************/
    VectorFunctionVMS(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            mVectorFunctionVMSResidual(),
            mVectorFunctionVMSJacobianU(),
            mVectorFunctionVMSJacobianN(),
            mVectorFunctionVMSJacobianX(),
            mVectorFunctionVMSJacobianZ(),
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
        return (mVectorFunctionVMSResidual->getMesh());
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

    /**************************************************************************//**
    *
    * \brief Allocate residual evaluator
    * \param [in] aResidual residual evaluator
    * \param [in] aJacobian Jacobian evaluator
    *
    ******************************************************************************/
    void allocateResidual(const std::shared_ptr<Plato::AbstractVectorFunctionVMS<Residual>>& aResidual,
                          const std::shared_ptr<Plato::AbstractVectorFunctionVMS<Jacobian>>& aJacobian)
    {
        mVectorFunctionVMSResidual = aResidual;
        mVectorFunctionVMSJacobianU = aJacobian;
    }

    /**************************************************************************//**
    *
    * \brief Allocate partial derivative with respect to control evaluator
    * \param [in] aGradientZ partial derivative with respect to control evaluator
    *
    ******************************************************************************/
    void allocateJacobianZ(const std::shared_ptr<Plato::AbstractVectorFunctionVMS<GradientZ>>& aGradientZ)
    {
        mVectorFunctionVMSJacobianZ = aGradientZ; 
    }

    /**************************************************************************//**
    *
    * \brief Allocate partial derivative with respect to configuration evaluator
    * \param [in] GradientX partial derivative with respect to configuration evaluator
    *
    ******************************************************************************/
    void allocateJacobianX(const std::shared_ptr<Plato::AbstractVectorFunctionVMS<GradientX>>& aGradientX)
    {
        mVectorFunctionVMSJacobianX = aGradientX; 
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
    value(const Plato::ScalarVector & aState,
          const Plato::ScalarVector & aNodeState,
          const Plato::ScalarVector & aControl,
          Plato::Scalar aTimeStep = 0.0) const
    {
        auto tResidual = this->valueWorkset(aState, aNodeState, aControl, aTimeStep);
        Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace>
            tAssembledResidual("Assembled Residual", mNumStateDofsPerNode * mNumNodes);
        mWorksetBase.assembleResidual(tResidual, tAssembledResidual);

        return tAssembledResidual;
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
    gradient_x(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
        // Allocate Jacobian
        auto tMesh = mVectorFunctionVMSJacobianX->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumStateDofsPerNode>(&tMesh);

        // Assemble Jacobian
        auto tJacobianMatEntries = tJacobianMat->entries();
        auto tJacobian = this->jacobianConfigWorkset(aState, aNodeState, aControl, aTimeStep);
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumStateDofsPerNode>
            tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);
        mWorksetBase.assembleTransposeJacobian(
           mNumStateDofsPerCell,
           mNumConfigDofsPerCell,
           tJacobianMatEntryOrdinal,
           tJacobian,
           tJacobianMatEntries
        );

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
    gradient_x_workset(const Plato::ScalarVector & aState,
                       const Plato::ScalarVector & aNodeState,
                       const Plato::ScalarVector & aControl,
                       Plato::Scalar aTimeStep = 0.0) const
    {
        auto tJacobianWS = this->jacobianConfigWorkset(aState, aNodeState, aControl, aTimeStep);
        Plato::ScalarArray3D tOutputJacobian("Jacobian WRT Configuration", mNumCells, mNumStateDofsPerCell, mNumConfigDofsPerCell);
        Plato::transform_ad_type_to_pod_3Dview<mNumStateDofsPerCell, mNumConfigDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
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
    gradient_u_T(const Plato::ScalarVector & aState,
                 const Plato::ScalarVector & aNodeState,
                 const Plato::ScalarVector & aControl,
                 Plato::Scalar aTimeStep = 0.0) const
    {
      // Allocate Jacobian
      auto tMesh = mVectorFunctionVMSJacobianU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumStateDofsPerNode, mNumStateDofsPerNode>( &tMesh );

      // Assemble Jacobian
      auto tJacobianMatEntries = tJacobianMat->entries();
      auto tJacobian = this->jacobianStateWorkset(aState, aNodeState, aControl, aTimeStep);
      Plato::BlockMatrixTransposeEntryOrdinal<mNumSpatialDims, mNumStateDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );
      mWorksetBase.assembleJacobianFad(
        mNumStateDofsPerCell,
        mNumStateDofsPerCell,
        tJacobianMatEntryOrdinal,
        tJacobian,
        tJacobianMatEntries
      );

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
    gradient_u(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
      // Allocate Jacobian
      auto tMesh = mVectorFunctionVMSJacobianU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumStateDofsPerNode, mNumStateDofsPerNode>( &tMesh );

      // Assemble Jacobian
      auto tJacobianMatEntries = tJacobianMat->entries();
      auto tJacobian = this->jacobianStateWorkset(aState, aNodeState, aControl, aTimeStep);
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumStateDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );
      mWorksetBase.assembleJacobianFad(
        mNumStateDofsPerCell,
        mNumStateDofsPerCell,
        tJacobianMatEntryOrdinal,
        tJacobian,
        tJacobianMatEntries
      );

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
    gradient_n_workset(const Plato::ScalarVector & aState,
                       const Plato::ScalarVector & aNodeState,
                       const Plato::ScalarVector & aControl,
                       Plato::Scalar aTimeStep = 0.0) const
    {
        auto tJacobianWS = this->jacobianNodeStateWorkset(aState, aNodeState, aControl, aTimeStep);
        Plato::ScalarArray3D tOutJacobian("POD Jacobian Node State", mNumCells, mNumStateDofsPerCell, mNumNodeStateDofsPerNode);
        Plato::transform_ad_type_to_pod_3Dview<mNumStateDofsPerCell, mNumNodeStateDofsPerNode>(mNumCells, tJacobianWS, tOutJacobian);
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
    gradient_n(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
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
        auto tMesh = mVectorFunctionVMSJacobianN->getMesh();
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
        auto tJacobian = this->jacobianNodeStateWorkset(aState, aNodeState, aControl, aTimeStep);
        mWorksetBase.assembleJacobianFad(
          mNumStateDofsPerCell,     /* (Nv x Nd) */
          mNumNodeStateDofsPerCell, /* (Nv x Nn) */
          tJacobianMatEntryOrdinal, /* entry ordinal functor */
          tJacobian,                /* source data */
          tJacobianMatEntries       /* destination */
        );

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
    gradient_n_T(const Plato::ScalarVector & aState,
                 const Plato::ScalarVector & aNodeState,
                 const Plato::ScalarVector & aControl,
                 Plato::Scalar aTimeStep = 0.0) const
    {
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
        auto tMesh = mVectorFunctionVMSJacobianN->getMesh();
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
        auto tJacobian = this->jacobianNodeStateWorkset(aState, aNodeState, aControl, aTimeStep);
        mWorksetBase.assembleTransposeJacobian(
            mNumStateDofsPerCell,     /* (Nv x Nd) */
            mNumNodeStateDofsPerCell, /* (Nv x Nn) */
            tJacobianMatEntryOrdinal, /* entry ordinal functor */
            tJacobian,                /* source data */
            tJacobianMatEntries       /* destination */
        );

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
    gradient_z(const Plato::ScalarVectorT<Plato::Scalar> & aState,
               const Plato::ScalarVectorT<Plato::Scalar> & aNodeState,
               const Plato::ScalarVectorT<Plato::Scalar> & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    {
      // Allocate Jacobian
      auto tMesh = mVectorFunctionVMSJacobianZ->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumStateDofsPerNode>( &tMesh );

      // Assemble Jacobian
      auto tJacobianMatEntries = tJacobianMat->entries();
      auto tJacobian = this->jacobianControlWorkset(aState, aNodeState, aControl, aTimeStep);
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumStateDofsPerNode> tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );
      mWorksetBase.assembleTransposeJacobian(
        mNumStateDofsPerCell,
        mNumNodesPerCell,
        tJacobianMatEntryOrdinal,
        tJacobian,
        tJacobianMatEntries
      );

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
    gradient_z_workset(const Plato::ScalarVectorT<Plato::Scalar> & aState,
                       const Plato::ScalarVectorT<Plato::Scalar> & aNodeState,
                       const Plato::ScalarVectorT<Plato::Scalar> & aControl,
                       Plato::Scalar aTimeStep = 0.0) const
    {
      auto tJacobianWS = this->jacobianControlWorkset(aState, aNodeState, aControl, aTimeStep);
      Plato::ScalarArray3D tOutputJacobian("Output Jacobian WRT Control", mNumCells, mNumStateDofsPerCell, mNumNodesPerCell);
      Plato::transform_ad_type_to_pod_3Dview<mNumStateDofsPerCell, mNumNodesPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
      return tOutputJacobian;
    }
};
// class VectorFunctionVMS

} // namespace Plato

#endif

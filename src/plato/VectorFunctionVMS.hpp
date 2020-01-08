#ifndef VECTOR_FUNCTION_VMS_HPP
#define VECTOR_FUNCTION_VMS_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "plato/WorksetBase.hpp"
#include "plato/AbstractVectorFunctionVMS.hpp"
#include "plato/SimplexFadTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! constraint class

   This class takes as a template argument a vector function in the form:

   and manages the evaluation of the function and derivatives wrt state
   and control.
  
*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunctionVMS : public Plato::WorksetBase<PhysicsT>
{
  private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode;
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims;
    using Plato::WorksetBase<PhysicsT>::mNumControl;
    using Plato::WorksetBase<PhysicsT>::mNumNodes;
    using Plato::WorksetBase<PhysicsT>::mNumCells;

    using Plato::WorksetBase<PhysicsT>::mNumNodeStatePerNode;
    using Plato::WorksetBase<PhysicsT>::mNumNodeStatePerCell;

    using Plato::WorksetBase<PhysicsT>::mGlobalStateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal;

    using Residual  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using Jacobian  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian;
    using JacobianN = typename Plato::Evaluation<typename PhysicsT::SimplexT>::JacobianN;
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;

    std::shared_ptr<Plato::AbstractVectorFunctionVMS<Residual>>  mVectorFunctionVMSResidual;
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<Jacobian>>  mVectorFunctionVMSJacobianU;
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<JacobianN>> mVectorFunctionVMSJacobianN;
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<GradientX>> mVectorFunctionVMSJacobianX;
    std::shared_ptr<Plato::AbstractVectorFunctionVMS<GradientZ>> mVectorFunctionVMSJacobianZ;

    Plato::DataMap& mDataMap;

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
            Plato::WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap)
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
            Plato::WorksetBase<PhysicsT>(aMesh),
            mVectorFunctionVMSResidual(),
            mVectorFunctionVMSJacobianU(),
            mVectorFunctionVMSJacobianN(),
            mVectorFunctionVMSJacobianX(),
            mVectorFunctionVMSJacobianZ(),
            mDataMap(aDataMap)
    {
    }

    /**************************************************************************//**
    *
    * \brief Return number of degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return mNumNodes*mNumDofsPerNode;
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
     * \brief Return number of global degrees of freedom per node.
     * \return number of global degrees of freedom per node
    ***************************************************************************/
    decltype(mNumDofsPerNode) numGlobalDofsPerNode() const
    {
        return mNumDofsPerNode;
    }

    /***********************************************************************//**
     * \brief Return number of global degrees of freedom per cell.
     * \return number of global degrees of freedom per cell
    ***************************************************************************/
    decltype(mNumDofsPerCell) numGlobalDofsPerCell() const
    {
        return mNumDofsPerCell;
    }

    /***********************************************************************//**
     * \brief Return number of pressure gradient degrees of freedom per node.
     * \return number of pressure gradient degrees of freedom per node
    ***************************************************************************/
    decltype(mNumNodeStatePerNode) numNodeStatePerNode() const
    {
        return mNumNodeStatePerNode;
    }

    /***********************************************************************//**
     * \brief Return number of pressure gradient degrees of freedom per cell.
     * \return number of pressure gradient degrees of freedom per cell
    ***************************************************************************/
    decltype(mNumNodeStatePerCell) numNodeStatePerCell() const
    {
        return mNumNodeStatePerCell;
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
     * \brief Evaluate and residual workset
     * \param [in] aState     projected presuure gradient
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
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // Workset node state
        //
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result
        //
        Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual",mNumCells, mNumDofsPerCell);

        // evaluate function
        //
        mVectorFunctionVMSResidual->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tResidual, aTimeStep );

        return (tResidual);
    }

    /***************************************************************************//**
     * \brief Evaluate and assemble residual
     * \param [in] aState     projected presuure gradient
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
            tAssembledResidual("Assembled Residual", mNumDofsPerNode * mNumNodes);
        Plato::WorksetBase<PhysicsT>::assembleResidual(tResidual, tAssembledResidual);

        return tAssembledResidual;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to configuration degrees of freedom
     * \param [in] aState     projected presuure gradient
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
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // Workset state
        //
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
        Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

        // Workset node state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // create return view
        //
        Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian Configuration", mNumCells, mNumDofsPerCell);

        // evaluate function
        //
        mVectorFunctionVMSJacobianX->evaluate(tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

        return (tJacobian);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to configuration degrees of freedom
     * \param [in] aState     projected presuure gradient
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
        // Allocate return Jacobian
        auto tMesh = mVectorFunctionVMSJacobianX->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumDofsPerNode>(&tMesh);

        // Assemble Jacobian
        auto tJacobianMatEntries = tJacobianMat->entries();
        auto tJacobian = this->jacobianConfigWorkset(aState, aNodeState, aControl, aTimeStep);
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode> tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);
        Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian(mNumDofsPerCell, mNumConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

        return tJacobianMat;
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to state degrees of freedom
     * \param [in] aState     projected presuure gradient
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
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // Workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // Workset node state
        //
        Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset",mNumCells, mNumNodeStatePerCell);
        Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // create return view
        //
        Plato::ScalarMultiVectorT<ResultScalar> tJacobian("Jacobian State",mNumCells,mNumDofsPerCell);

        // evaluate function
        //
        mVectorFunctionVMSJacobianU->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

        return (tJacobian);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian with respect to state degrees of freedom
     * \param [in] aState     projected presuure gradient
     * \param [in] aNodeState pressure field
     * \param [in] aControl   design variables
     * \param [in] aTimeStep  current time step
     * \return assembled Jacobian
    *******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u_T(const Plato::ScalarVector & aState,
                 const Plato::ScalarVector & aNodeState,
                 const Plato::ScalarVector & aControl,
                 Plato::Scalar aTimeStep = 0.0) const
    {

      // Allocate return Jacobian
      auto tMesh = mVectorFunctionVMSJacobianU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( &tMesh );

      // Assemble Jacobian
      auto tJacobianMatEntries = tJacobianMat->entries();
      auto tJacobian = this->jacobianStateWorkset(aState, aNodeState, aControl, aTimeStep);
      Plato::BlockMatrixTransposeEntryOrdinal<mNumSpatialDims, mNumDofsPerNode> tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );
      Plato::WorksetBase<PhysicsT>::assembleJacobianFad(mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
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
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset node state
      //
      Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset",mNumCells, mNumNodeStatePerCell);
      Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",mNumCells,mNumDofsPerCell);

      // evaluate function
      //
      mVectorFunctionVMSJacobianU->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionVMSJacobianU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleJacobianFad(mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_n(const Plato::ScalarVector & aState,
               const Plato::ScalarVector & aNodeState,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
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
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset node state
      //
      Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
      Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianNodeState", mNumCells, mNumDofsPerCell);

      // evaluate function
      //
      mVectorFunctionVMSJacobianN->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

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
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumNodeStatePerNode>( &tMesh );

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
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumNodeStatePerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      // Assemble from the AD-typed result, tJacobian, into the POD-typed global matrix, tJacobianMat.
      //
      // Arguments 1 and 2 below correspond to the size of tJacobian ((Nv x Nd), (Nv x Nn)) and the size of
      // tJacobianMat (Nd, Nn).
      //
      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleJacobianFad(
        mNumDofsPerCell,          /* (Nv x Nd) */
        mNumNodeStatePerCell,            /* (Nv x Nn) */
        tJacobianMatEntryOrdinal, /* entry ordinal functor */
        tJacobian,                /* source data */
        tJacobianMatEntries       /* destination */
      );

      return tJacobianMat;
    }
    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_n_T(const Plato::ScalarVector & aState,
                 const Plato::ScalarVector & aNodeState,
                 const Plato::ScalarVector & aControl,
                 Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
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
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset node state
      //
      Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset", mNumCells, mNumNodeStatePerCell);
      Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianNodeState", mNumCells, mNumDofsPerCell);

      // evaluate function
      //
      mVectorFunctionVMSJacobianN->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

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
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumNodeStatePerNode, mNumDofsPerNode>( &tMesh );

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
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumNodeStatePerNode, mNumDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      // Assemble from the AD-typed result, tJacobian, into the POD-typed global matrix, tJacobianMat.
      //
      // The transpose is being assembled, (i.e., tJacobian is transposed before assembly into tJacobianMat), so
      // arguments 1 and 2 below correspond to the size of tJacobian ((Nv x Nd), (Nv x Nn)) and the size of the
      // *transpose* of tJacobianMat (Transpose(Nn, Nd) => (Nd, Nn)).
      //
      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian(
        mNumDofsPerCell,          /* (Nv x Nd) */
        mNumNodeStatePerCell,            /* (Nv x Nn) */
        tJacobianMatEntryOrdinal, /* entry ordinal functor */
        tJacobian,                /* source data */
        tJacobianMatEntries       /* destination */
      );

      return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(const Plato::ScalarVectorT<Plato::Scalar> & aState,
               const Plato::ScalarVectorT<Plato::Scalar> & aNodeState,
               const Plato::ScalarVectorT<Plato::Scalar> & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
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
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);
 
      // Workset state
      //
      Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

      // Workset node state
      //
      Plato::ScalarMultiVectorT<NodeStateScalar> tNodeStateWS("Node State Workset",mNumCells, mNumNodeStatePerCell);
      Plato::WorksetBase<PhysicsT>::worksetNodeState(aNodeState, tNodeStateWS);

      // create result 
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl",mNumCells,mNumDofsPerCell);

      // evaluate function 
      //
      mVectorFunctionVMSJacobianZ->evaluate( tStateWS, tNodeStateWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionVMSJacobianZ->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian(mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }
};
// class VectorFunctionVMS

} // namespace Plato

#endif

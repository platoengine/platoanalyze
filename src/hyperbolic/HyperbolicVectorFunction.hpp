#ifndef VECTOR_FUNCTION_HYPERBOLIC_HPP
#define VECTOR_FUNCTION_HYPERBOLIC_HPP

#include <memory>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include "../../src/WorksetBase.hpp"
#include "HyperbolicSimplexFadTypes.hpp"
#include "HyperbolicAbstractVectorFunction.hpp"

namespace Plato
{

namespace Hyperbolic
{

/******************************************************************************/
/*! constraint class

   This class takes as a template argument a vector function in the form:

   F = F(\phi, U^k, V^k, A^k, X)

   and manages the evaluation of the function and derivatives with respect to 
   displacement, U^k, velocity, V^k, acceleration, V^k, and control, X.
  
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

    using Plato::WorksetBase<PhysicsT>::mStateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal;

    using Residual  = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradientU = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientU;
    using GradientV = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientV;
    using GradientA = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientA;
    using GradientX = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;

    std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<Residual>>  mVectorFunctionResidual;
    std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientU>> mVectorFunctionGradientU;
    std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientV>> mVectorFunctionGradientV;
    std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientA>> mVectorFunctionGradientA;
    std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientX>> mVectorFunctionGradientX;
    std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientZ>> mVectorFunctionGradientZ;

    Plato::DataMap& mDataMap;

  public:

    /**************************************************************************//**
    *
    * @brief Constructor
    * @param [in] aMesh mesh data base
    * @param [in] aMeshSets mesh sets data base
    * @param [in] aDataMap problem-specific data map 
    * @param [in] aParamList Teuchos parameter list with input data
    * @param [in] aProblemType problem type 
    *
    ******************************************************************************/
    VectorFunction(Omega_h::Mesh& aMesh,
                   Omega_h::MeshSets& aMeshSets,
                   Plato::DataMap& aDataMap,
                   Teuchos::ParameterList& aParamList,
                   std::string& aProblemType) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap)
    {
      typename PhysicsT::FunctionFactory tFunctionFactory;

      mVectorFunctionResidual = tFunctionFactory.template createVectorFunctionHyperbolic<Residual>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);
      mVectorFunctionGradientU = tFunctionFactory.template createVectorFunctionHyperbolic<GradientU>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);
      mVectorFunctionGradientV = tFunctionFactory.template createVectorFunctionHyperbolic<GradientV>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);
      mVectorFunctionGradientA = tFunctionFactory.template createVectorFunctionHyperbolic<GradientA>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);
      mVectorFunctionGradientZ = tFunctionFactory.template createVectorFunctionHyperbolic<GradientZ>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);
      mVectorFunctionGradientX = tFunctionFactory.template createVectorFunctionHyperbolic<GradientX>(aMesh, aMeshSets, aDataMap, aParamList, aProblemType);
    }

    /**************************************************************************//**
    *
    * @brief Constructor
    * @param [in] aMesh mesh data base
    * @param [in] aDataMap problem-specific data map 
    *
    ******************************************************************************/
    VectorFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mVectorFunctionResidual(),
            mVectorFunctionGradientU(),
            mVectorFunctionGradientV(),
            mVectorFunctionGradientA(),
            mVectorFunctionGradientX(),
            mVectorFunctionGradientZ(),
            mDataMap(aDataMap)
    {
    }

    /**************************************************************************//**
    *
    * @brief Allocate residual evaluator
    * @param [in] aResidual  residual evaluator
    * @param [in] aGradientU gradient evaluator
    *
    ******************************************************************************/
    void allocateResidual(const std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<Residual>>& aResidual,
                          const std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientU>>& aGradientU)
    {
        mVectorFunctionResidual  = aResidual;
        mVectorFunctionGradientU = aGradientU;
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to control evaluator
    * @param [in] aGradientZ partial derivative with respect to control evaluator
    *
    ******************************************************************************/
    void allocateGradientZ(const std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientZ>>& aGradientZ)
    {
        mVectorFunctionGradientZ = aGradientZ; 
    }

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to configuration evaluator
    * @param [in] GradientX partial derivative with respect to configuration evaluator
    *
    ******************************************************************************/
    void allocateGradientX(const std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientX>>& aGradientX)
    {
        mVectorFunctionGradientX = aGradientX; 
    }

    /**************************************************************************//**
    *
    * @brief Return local number of degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return mNumNodes*mNumDofsPerNode;
    }

    /**************************************************************************//**
    *
    * @brief Return state names
    *
    ******************************************************************************/
    std::vector<std::string> getDofNames() const
    {
      return mVectorFunctionResidual->getDofNames();
    }

    /**************************************************************************/
    Plato::ScalarVector
    value(const Plato::ScalarVector & aDisplacement,
          const Plato::ScalarVector & aVelocity,
          const Plato::ScalarVector & aAcceleration,
          const Plato::ScalarVector & aControl,
          Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar       = typename Residual::ConfigScalarType;
      using DisplacementScalar = typename Residual::DisplacementScalarType;
      using VelocityScalar     = typename Residual::VelocityScalarType;
      using AccelerationScalar = typename Residual::AccelerationScalarType;
      using ControlScalar      = typename Residual::ControlScalarType;
      using ResultScalar       = typename Residual::ResultScalarType;

      // Workset displacement
      //
      Plato::ScalarMultiVectorT<DisplacementScalar> tDisplacementWS("Displacement Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aDisplacement, tDisplacementWS);

      // Workset velocity
      //
      Plato::ScalarMultiVectorT<VelocityScalar> tVelocityWS("Velocity Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aVelocity, tVelocityWS);

      // Workset acceleration
      //
      Plato::ScalarMultiVectorT<AccelerationScalar> tAccelerationWS("Acceleration Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aAcceleration, tAccelerationWS);

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
      mVectorFunctionResidual->evaluate( tDisplacementWS, tVelocityWS, tAccelerationWS, tControlWS, tConfigWS, tResidual, aTimeStep );

      // create and assemble to return view
      //
      Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace>  tReturnValue("Assembled Residual",mNumDofsPerNode*mNumNodes);
      Plato::WorksetBase<PhysicsT>::assembleResidual( tResidual, tReturnValue );

      return tReturnValue;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(const Plato::ScalarVector & aDisplacement,
               const Plato::ScalarVector & aVelocity,
               const Plato::ScalarVector & aAcceleration,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
        using ConfigScalar       = typename GradientX::ConfigScalarType;
        using DisplacementScalar = typename GradientX::DisplacementScalarType;
        using VelocityScalar     = typename GradientX::VelocityScalarType;
        using AccelerationScalar = typename GradientX::AccelerationScalarType;
        using ControlScalar      = typename GradientX::ControlScalarType;
        using ResultScalar       = typename GradientX::ResultScalarType;

        // Workset config
        //
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // Workset displacement
        //
        Plato::ScalarMultiVectorT<DisplacementScalar> tDisplacementWS("Displacement Workset", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aDisplacement, tDisplacementWS);

        // Workset velocity
        //
        Plato::ScalarMultiVectorT<VelocityScalar> tVelocityWS("Velocity Workset", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aVelocity, tVelocityWS);

        // Workset acceleration
        //
        Plato::ScalarMultiVectorT<AccelerationScalar> tAccelerationWS("Acceleration Workset", mNumCells, mNumDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aAcceleration, tAccelerationWS);

        // Workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // create return view
        //
        Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", mNumCells, mNumDofsPerCell);

        // evaluate function
        //
        mVectorFunctionGradientX->evaluate(tDisplacementWS, tVelocityWS, tAccelerationWS, tControlWS, tConfigWS, tJacobian, aTimeStep);

        // create return matrix
        //
        auto tMesh = mVectorFunctionGradientX->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumSpatialDims, mNumDofsPerNode>(&tMesh);

        // assembly to return matrix
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode>
            tJacobianMatEntryOrdinal(tJacobianMat, &tMesh);

        auto tJacobianMatEntries = tJacobianMat->entries();
        Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian(mNumDofsPerCell, mNumConfigDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(const Plato::ScalarVector & aDisplacement,
               const Plato::ScalarVector & aVelocity,
               const Plato::ScalarVector & aAcceleration,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar       = typename GradientU::ConfigScalarType;
      using DisplacementScalar = typename GradientU::DisplacementScalarType;
      using VelocityScalar     = typename GradientU::VelocityScalarType;
      using AccelerationScalar = typename GradientU::AccelerationScalarType;
      using ControlScalar      = typename GradientU::ControlScalarType;
      using ResultScalar       = typename GradientU::ResultScalarType;

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset displacement
      // 
      Plato::ScalarMultiVectorT<DisplacementScalar> tDisplacementWS("Displacement Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aDisplacement, tDisplacementWS);

      // Workset velocity
      // 
      Plato::ScalarMultiVectorT<VelocityScalar> tVelocityWS("Velocity Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aVelocity, tVelocityWS);

      // Workset acceleration
      // 
      Plato::ScalarMultiVectorT<AccelerationScalar> tAccelerationWS("Acceleration Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aAcceleration, tAccelerationWS);

      // Workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",mNumCells,mNumDofsPerCell);

      // evaluate function
      //
      mVectorFunctionGradientU->evaluate( tDisplacementWS, tVelocityWS, tAccelerationWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionGradientU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleJacobian(mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_v(const Plato::ScalarVector & aDisplacement,
               const Plato::ScalarVector & aVelocity,
               const Plato::ScalarVector & aAcceleration,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar       = typename GradientV::ConfigScalarType;
      using DisplacementScalar = typename GradientV::DisplacementScalarType;
      using VelocityScalar     = typename GradientV::VelocityScalarType;
      using AccelerationScalar = typename GradientV::AccelerationScalarType;
      using ControlScalar      = typename GradientV::ControlScalarType;
      using ResultScalar       = typename GradientV::ResultScalarType;

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset displacement
      // 
      Plato::ScalarMultiVectorT<DisplacementScalar> tDisplacementWS("Displacement Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aDisplacement, tDisplacementWS);

      // Workset velocity
      // 
      Plato::ScalarMultiVectorT<VelocityScalar> tVelocityWS("Velocity Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aVelocity, tVelocityWS);

      // Workset acceleration
      // 
      Plato::ScalarMultiVectorT<AccelerationScalar> tAccelerationWS("Acceleration Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aAcceleration, tAccelerationWS);

      // Workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",mNumCells,mNumDofsPerCell);

      // evaluate function
      //
      mVectorFunctionGradientV->evaluate( tDisplacementWS, tVelocityWS, tAccelerationWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionGradientU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleJacobian(mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_a(const Plato::ScalarVector & aDisplacement,
               const Plato::ScalarVector & aVelocity,
               const Plato::ScalarVector & aAcceleration,
               const Plato::ScalarVector & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar       = typename GradientA::ConfigScalarType;
      using DisplacementScalar = typename GradientA::DisplacementScalarType;
      using VelocityScalar     = typename GradientA::VelocityScalarType;
      using AccelerationScalar = typename GradientA::AccelerationScalarType;
      using ControlScalar      = typename GradientA::ControlScalarType;
      using ResultScalar       = typename GradientA::ResultScalarType;

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset displacement
      // 
      Plato::ScalarMultiVectorT<DisplacementScalar> tDisplacementWS("Displacement Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aDisplacement, tDisplacementWS);

      // Workset velocity
      // 
      Plato::ScalarMultiVectorT<VelocityScalar> tVelocityWS("Velocity Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aVelocity, tVelocityWS);

      // Workset acceleration
      // 
      Plato::ScalarMultiVectorT<AccelerationScalar> tAccelerationWS("Acceleration Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aAcceleration, tAccelerationWS);

      // Workset control
      // 
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

      // create return view
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState",mNumCells,mNumDofsPerCell);

      // evaluate function
      //
      mVectorFunctionGradientA->evaluate( tDisplacementWS, tVelocityWS, tAccelerationWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionGradientU->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumDofsPerNode, mNumDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
          tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleJacobian(mNumDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(const Plato::ScalarVectorT<Plato::Scalar> & aDisplacement,
               const Plato::ScalarVectorT<Plato::Scalar> & aVelocity,
               const Plato::ScalarVectorT<Plato::Scalar> & aAcceleration,
               const Plato::ScalarVectorT<Plato::Scalar> & aControl,
               Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      using ConfigScalar       = typename GradientZ::ConfigScalarType;
      using DisplacementScalar = typename GradientZ::DisplacementScalarType;
      using VelocityScalar     = typename GradientZ::VelocityScalarType;
      using AccelerationScalar = typename GradientZ::AccelerationScalarType;
      using ControlScalar      = typename GradientZ::ControlScalarType;
      using ResultScalar       = typename GradientZ::ResultScalarType;

      // Workset config
      // 
      Plato::ScalarArray3DT<ConfigScalar>
          tConfigWS("Config Workset",mNumCells, mNumNodesPerCell, mNumSpatialDims);
      Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

      // Workset control
      //
      Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset",mNumCells,mNumNodesPerCell);
      Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);
 
      // Workset displacement
      //
      Plato::ScalarMultiVectorT<DisplacementScalar> tDisplacementWS("Displacement Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aDisplacement, tDisplacementWS);

      // Workset velocity
      //
      Plato::ScalarMultiVectorT<VelocityScalar> tVelocityWS("Velocity Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aVelocity, tVelocityWS);

      // Workset acceleration
      //
      Plato::ScalarMultiVectorT<AccelerationScalar> tAccelerationWS("Acceleration Workset",mNumCells,mNumDofsPerCell);
      Plato::WorksetBase<PhysicsT>::worksetState(aAcceleration, tAccelerationWS);

      // create result 
      //
      Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl",mNumCells,mNumDofsPerCell);

      // evaluate function 
      //
      mVectorFunctionGradientZ->evaluate( tDisplacementWS, tVelocityWS, tAccelerationWS, tControlWS, tConfigWS, tJacobian, aTimeStep );

      // create return matrix
      //
      auto tMesh = mVectorFunctionGradientZ->getMesh();
      Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
              Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControl, mNumDofsPerNode>( &tMesh );

      // assembly to return matrix
      Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
        tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

      auto tJacobianMatEntries = tJacobianMat->entries();
      Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian(mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);

      return (tJacobianMat);
    }
}; // class VectorFunction

} // namespace Hyperbolic

} // namespace Plato

#endif

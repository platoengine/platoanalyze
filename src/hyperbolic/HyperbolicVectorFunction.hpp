#ifndef VECTOR_FUNCTION_HYPERBOLIC_HPP
#define VECTOR_FUNCTION_HYPERBOLIC_HPP

#include <memory>

#include "SpatialModel.hpp"
#include "../WorksetBase.hpp"
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
   state, U^k, state dot, V^k, state dot dot, V^k, and control, X.

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

    using Residual  = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradientU = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientU;
    using GradientV = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientV;
    using GradientA = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientA;
    using GradientX = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Hyperbolic::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    using ResidualFunction  = std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<Residual>>;
    using GradientUFunction = std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientU>>;
    using GradientVFunction = std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientV>>;
    using GradientAFunction = std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientA>>;
    using GradientXFunction = std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Hyperbolic::AbstractVectorFunction<GradientZ>>;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;

    std::map<std::string, ResidualFunction>  mResidualFunctions;
    std::map<std::string, GradientUFunction> mGradientUFunctions;
    std::map<std::string, GradientVFunction> mGradientVFunctions;
    std::map<std::string, GradientAFunction> mGradientAFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    ResidualFunction  mBoundaryLoadsResidualFunction;
    GradientUFunction mBoundaryLoadsGradientUFunction;
    GradientVFunction mBoundaryLoadsGradientVFunction;
    GradientAFunction mBoundaryLoadsGradientAFunction;
    GradientXFunction mBoundaryLoadsGradientXFunction;
    GradientZFunction mBoundaryLoadsGradientZFunction;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap;

  public:

    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap problem-specific data map 
    * \param [in] aParamList Teuchos parameter list with input data
    * \param [in] aProblemType problem type 
    *
    ******************************************************************************/
    VectorFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList,
              std::string            & aProblemType
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap)
    {
        typename PhysicsT::FunctionFactory tFunctionFactory;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mResidualFunctions[tName]  = tFunctionFactory.template createVectorFunctionHyperbolic<Residual>
                (tDomain, aDataMap, aParamList, aProblemType);
            mGradientUFunctions[tName] = tFunctionFactory.template createVectorFunctionHyperbolic<GradientU>
                (tDomain, aDataMap, aParamList, aProblemType);
            mGradientVFunctions[tName] = tFunctionFactory.template createVectorFunctionHyperbolic<GradientV>
                (tDomain, aDataMap, aParamList, aProblemType);
            mGradientAFunctions[tName] = tFunctionFactory.template createVectorFunctionHyperbolic<GradientA>
                (tDomain, aDataMap, aParamList, aProblemType);
            mGradientZFunctions[tName] = tFunctionFactory.template createVectorFunctionHyperbolic<GradientZ>
                (tDomain, aDataMap, aParamList, aProblemType);
            mGradientXFunctions[tName] = tFunctionFactory.template createVectorFunctionHyperbolic<GradientX>
                (tDomain, aDataMap, aParamList, aProblemType);
        }
        // any block can compute the boundary terms for the entire mesh.  We'll use the first block.
        auto tFirstBlockName = aSpatialModel.Domains.front().getDomainName();

        mBoundaryLoadsResidualFunction  = mResidualFunctions[tFirstBlockName];
        mBoundaryLoadsGradientUFunction = mGradientUFunctions[tFirstBlockName];
        mBoundaryLoadsGradientVFunction = mGradientVFunctions[tFirstBlockName];
        mBoundaryLoadsGradientAFunction = mGradientAFunctions[tFirstBlockName];
        mBoundaryLoadsGradientZFunction = mGradientZFunctions[tFirstBlockName];
        mBoundaryLoadsGradientXFunction = mGradientXFunctions[tFirstBlockName];

    }

    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aMesh mesh data base
    * \param [in] aDataMap problem-specific data map 
    *
    ******************************************************************************/
    VectorFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap)
    {
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

    /**************************************************************************//**
    *
    * \brief Return state names
    *
    ******************************************************************************/
    std::vector<std::string> getDofNames() const
    {
        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
        return mResidualFunctions.at(tFirstBlockName)->getDofNames();
    }

    /**************************************************************************//**
    *
    * \brief Return state dot names
    *
    ******************************************************************************/
    std::vector<std::string> getDofDotNames() const
    {
        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
        return mResidualFunctions.at(tFirstBlockName)->getDofDotNames();
    }

    /**************************************************************************//**
    *
    * \brief Call the output state function in the residual
    * 
    * \param [in] aSolutions State solutions database
    * \return output solutions database
    * 
    ******************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const
    {
        return mBoundaryLoadsResidualFunction->getSolutionStateOutputData(aSolutions);
    }

    /**************************************************************************/
    Plato::ScalarVector
    value(
      const Plato::ScalarVector & aState,
      const Plato::ScalarVector & aStateDot,
      const Plato::ScalarVector & aStateDotDot,
      const Plato::ScalarVector & aControl,
            Plato::Scalar         aTimeStep,
            Plato::Scalar         aCurrentTime = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename Residual::ConfigScalarType;
        using StateScalar       = typename Residual::StateScalarType;
        using StateDotScalar    = typename Residual::StateDotScalarType;
        using StateDotDotScalar = typename Residual::StateDotDotScalarType;
        using ControlScalar     = typename Residual::ControlScalarType;
        using ResultScalar      = typename Residual::ResultScalarType;

        Plato::ScalarVector tReturnValue("Assembled Residual", mNumDofsPerNode * mNumNodes);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS, tDomain);

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS, tDomain);

            // Workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDotDot, tStateDotDotWS, tDomain);

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
            mResidualFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResidual, aTimeStep, aCurrentTime );

            // create and assemble to return view
            //
            Plato::WorksetBase<PhysicsT>::assembleResidual( tResidual, tReturnValue, tDomain );
        }

        {
            auto tNumCells = mSpatialModel.Mesh.nelems();

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS);

            // Workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDotDot, tStateDotDotWS);

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
            mBoundaryLoadsResidualFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tResidual, aTimeStep, aCurrentTime );

            // create and assemble to return view
            //
            Plato::WorksetBase<PhysicsT>::assembleResidual( tResidual, tReturnValue );
        }

        return tReturnValue;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(
      const Plato::ScalarVector & aState,
      const Plato::ScalarVector & aStateDot,
      const Plato::ScalarVector & aStateDotDot,
      const Plato::ScalarVector & aControl,
            Plato::Scalar         aTimeStep,
            Plato::Scalar         aCurrentTime = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename GradientX::ConfigScalarType;
        using StateScalar       = typename GradientX::StateScalarType;
        using StateDotScalar    = typename GradientX::StateDotScalarType;
        using StateDotDotScalar = typename GradientX::StateDotDotScalarType;
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

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS, tDomain);

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS, tDomain);

            // Workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDotDot, tStateDotDotWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientXFunctions.at(tName)->evaluate(tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime);

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

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS);

            // Workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDotDot, tStateDotDotWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianConfiguration", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsGradientXFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime);

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
    gradient_u(
      const Plato::ScalarVector & aState,
      const Plato::ScalarVector & aStateDot,
      const Plato::ScalarVector & aStateDotDot,
      const Plato::ScalarVector & aControl,
            Plato::Scalar         aTimeStep,
            Plato::Scalar         aCurrentTime = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename GradientU::ConfigScalarType;
        using StateScalar       = typename GradientU::StateScalarType;
        using StateDotScalar    = typename GradientU::StateDotScalarType;
        using StateDotDotScalar = typename GradientU::StateDotDotScalarType;
        using ControlScalar     = typename GradientU::ControlScalarType;
        using ResultScalar      = typename GradientU::ResultScalarType;

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

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS, tDomain);

            // Workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDotDot, tStateDotDotWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientUFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

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

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS);

            // Workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDotDot, tStateDotDotWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsGradientUFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

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
    gradient_v(
      const Plato::ScalarVector & aState,
      const Plato::ScalarVector & aStateDot,
      const Plato::ScalarVector & aStateDotDot,
      const Plato::ScalarVector & aControl,
            Plato::Scalar         aTimeStep,
            Plato::Scalar         aCurrentTime = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename GradientV::ConfigScalarType;
        using StateScalar       = typename GradientV::StateScalarType;
        using StateDotScalar    = typename GradientV::StateDotScalarType;
        using StateDotDotScalar = typename GradientV::StateDotDotScalarType;
        using ControlScalar     = typename GradientV::ControlScalarType;
        using ResultScalar      = typename GradientV::ResultScalarType;

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

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS, tDomain);

            // Workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDotDot, tStateDotDotWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientVFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

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

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS);

            // Workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDotDot, tStateDotDotWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsGradientVFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

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
    gradient_a(
      const Plato::ScalarVector & aState,
      const Plato::ScalarVector & aStateDot,
      const Plato::ScalarVector & aStateDotDot,
      const Plato::ScalarVector & aControl,
            Plato::Scalar         aTimeStep,
            Plato::Scalar         aCurrentTime = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename GradientA::ConfigScalarType;
        using StateScalar       = typename GradientA::StateScalarType;
        using StateDotScalar    = typename GradientA::StateDotScalarType;
        using StateDotDotScalar = typename GradientA::StateDotDotScalarType;
        using ControlScalar     = typename GradientA::ControlScalarType;
        using ResultScalar      = typename GradientA::ResultScalarType;

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

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS, tDomain);

            // Workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDotDot, tStateDotDotWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientAFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

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

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS);

            // Workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDotDot, tStateDotDotWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianState", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsGradientAFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

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
      const Plato::ScalarVector & aStateDot,
      const Plato::ScalarVector & aStateDotDot,
      const Plato::ScalarVector & aControl,
            Plato::Scalar       aTimeStep,
            Plato::Scalar       aCurrentTime = 0.0
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename GradientZ::ConfigScalarType;
        using StateScalar       = typename GradientZ::StateScalarType;
        using StateDotScalar    = typename GradientZ::StateDotScalarType;
        using StateDotDotScalar = typename GradientZ::StateDotDotScalarType;
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
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS, tDomain);

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS, tDomain);

            // Workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDotDot, tStateDotDotWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientZFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

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
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS);

            // Workset state dot dot
            //
            Plato::ScalarMultiVectorT<StateDotDotScalar> tStateDotDotWS("StateDotDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDotDot, tStateDotDotWS);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobian("JacobianControl", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsGradientZFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tStateDotDotWS, tControlWS, tConfigWS, tJacobian, aTimeStep, aCurrentTime );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
                tJacobianMatEntryOrdinal( tJacobianMat, &tMesh );

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumNodesPerCell, tJacobianMatEntryOrdinal, tJacobian, tJacobianMatEntries);
        }
        return tJacobianMat;
    }
}; // class VectorFunction

} // namespace Hyperbolic

} // namespace Plato

#endif

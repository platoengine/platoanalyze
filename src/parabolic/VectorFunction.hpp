#ifndef VECTOR_FUNCTION_PARABOLIC_HPP
#define VECTOR_FUNCTION_PARABOLIC_HPP

#include <memory>

#include "SpatialModel.hpp"
#include "../WorksetBase.hpp"
#include "parabolic/ParabolicSimplexFadTypes.hpp"
#include "parabolic/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
/*! constraint class

   This class takes as a template argument a vector function in the form:

   F = F(\phi, U^k, U^{k-1}, X)

   and manages the evaluation of the function and derivatives wrt state, U^k;
   previous state, U^{k-1}; and control, X.

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

    using Residual  = typename Plato::Parabolic::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradientU = typename Plato::Parabolic::Evaluation<typename PhysicsT::SimplexT>::GradientU;
    using GradientV = typename Plato::Parabolic::Evaluation<typename PhysicsT::SimplexT>::GradientV;
    using GradientX = typename Plato::Parabolic::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Parabolic::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    using ResidualFunction  = std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<Residual>>;
    using GradientUFunction = std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<GradientU>>;
    using GradientVFunction = std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<GradientV>>;
    using GradientXFunction = std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<GradientZ>>;

    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims*mNumNodesPerCell;

    std::map<std::string, ResidualFunction>  mResidualFunctions;
    std::map<std::string, GradientUFunction> mGradientUFunctions;
    std::map<std::string, GradientVFunction> mGradientVFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    ResidualFunction  mBoundaryLoadsResidualFunction;
    GradientUFunction mBoundaryLoadsGradientUFunction;
    GradientVFunction mBoundaryLoadsGradientVFunction;
    GradientXFunction mBoundaryLoadsGradientXFunction;
    GradientZFunction mBoundaryLoadsGradientZFunction;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap;

  public:

    /**************************************************************************//**
    *
    * @brief Constructor
    * @param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * @param [in] aDataMap problem-specific data map
    * @param [in] aParamList Teuchos parameter list with input data
    * @param [in] aProblemType problem type
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
            mResidualFunctions[tName]  = tFunctionFactory.template createVectorFunctionParabolic<Residual >(tDomain, aDataMap, aParamList, aProblemType);
            mGradientUFunctions[tName] = tFunctionFactory.template createVectorFunctionParabolic<GradientU>(tDomain, aDataMap, aParamList, aProblemType);
            mGradientVFunctions[tName] = tFunctionFactory.template createVectorFunctionParabolic<GradientV>(tDomain, aDataMap, aParamList, aProblemType);
            mGradientZFunctions[tName] = tFunctionFactory.template createVectorFunctionParabolic<GradientZ>(tDomain, aDataMap, aParamList, aProblemType);
            mGradientXFunctions[tName] = tFunctionFactory.template createVectorFunctionParabolic<GradientX>(tDomain, aDataMap, aParamList, aProblemType);
        }

        // any block can compute the boundary terms for the entire mesh.  We'll use the first block.
        auto tFirstBlockName = aSpatialModel.Domains[0].getDomainName();

        mBoundaryLoadsResidualFunction  = mResidualFunctions[tFirstBlockName];
        mBoundaryLoadsGradientUFunction = mGradientUFunctions[tFirstBlockName];
        mBoundaryLoadsGradientVFunction = mGradientVFunctions[tFirstBlockName];
        mBoundaryLoadsGradientZFunction = mGradientZFunctions[tFirstBlockName];
        mBoundaryLoadsGradientXFunction = mGradientXFunctions[tFirstBlockName];

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
            mDataMap(aDataMap)
    {
    }

    /**************************************************************************//**
    *
    * @brief Allocate residual evaluator
    * @param [in] aResidual residual evaluator
    * @param [in] aGradientU GradientU evaluator
    *
    ******************************************************************************/
// TODO is this necessary?
/*
    void allocateResidual(const std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<Residual>>& aResidual,
                          const std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<GradientU>>& aGradientU)
    {
        mVectorFunctionResidual = aResidual;
        mVectorFunctionGradientU = aGradientU;
    }
*/

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to control evaluator
    * @param [in] aGradientZ partial derivative with respect to control evaluator
    *
    ******************************************************************************/
// TODO is this necessary?
/*
    void allocateGradientZ(const std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<GradientZ>>& aGradientZ)
    {
        mVectorFunctionGradientZ = aGradientZ;
    }
*/

    /**************************************************************************//**
    *
    * @brief Allocate partial derivative with respect to configuration evaluator
    * @param [in] GradientX partial derivative with respect to configuration evaluator
    *
    ******************************************************************************/
// TODO is this necessary?
/*
    void allocateGradientX(const std::shared_ptr<Plato::Parabolic::AbstractVectorFunction<GradientX>>& aGradientX)
    {
        mVectorFunctionGradientX = aGradientX;
    }
*/

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
        auto tFirstBlockName = mSpatialModel.Domains.front().getDomainName();
        return mResidualFunctions.at(tFirstBlockName)->getDofNames();
    }

    /**************************************************************************/
    Plato::ScalarVector
    value(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aStateDot,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        using ConfigScalar   = typename Residual::ConfigScalarType;
        using StateScalar    = typename Residual::StateScalarType;
        using StateDotScalar = typename Residual::StateDotScalarType;
        using ControlScalar  = typename Residual::ControlScalarType;
        using ResultScalar   = typename Residual::ResultScalarType;

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
            mResidualFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tResidual, aTimeStep );

            // create and assemble to return view
            //
            Plato::WorksetBase<PhysicsT>::assembleResidual( tResidual, tReturnValue, tDomain );
        }

        {
            // Workset state
            //
            Plato::ScalarMultiVectorT<StateScalar> tStateWS("State Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

            // Workset state dot
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResidual("Cells Residual", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsResidualFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tControlWS, tConfigWS, tResidual, aTimeStep );

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
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        using ConfigScalar   = typename GradientX::ConfigScalarType;
        using StateScalar    = typename GradientX::StateScalarType;
        using StateDotScalar = typename GradientX::StateDotScalarType;
        using ControlScalar  = typename GradientX::ControlScalarType;
        using ResultScalar   = typename GradientX::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tGradientMat =
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

            // Workset prev state
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tGradient("GradientConfiguration", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientXFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tGradient, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode>
                tGradientMatEntryOrdinal(tGradientMat, &tMesh);

            auto tGradientMatEntries = tGradientMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumConfigDofsPerCell, tGradientMatEntryOrdinal, tGradient, tGradientMatEntries, tDomain);
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

            // Workset prev state
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tGradient("GradientConfiguration", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsGradientXFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tControlWS, tConfigWS, tGradient, aTimeStep);

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumSpatialDims, mNumDofsPerNode>
                tGradientMatEntryOrdinal(tGradientMat, &tMesh);

            auto tGradientMatEntries = tGradientMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumConfigDofsPerCell, tGradientMatEntryOrdinal, tGradient, tGradientMatEntries);
        }
        return tGradientMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aStateDot,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        using ConfigScalar   = typename GradientU::ConfigScalarType;
        using StateScalar    = typename GradientU::StateScalarType;
        using StateDotScalar = typename GradientU::StateDotScalarType;
        using ControlScalar  = typename GradientU::ControlScalarType;
        using ResultScalar   = typename GradientU::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tGradientMat =
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

            // Workset prev state
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tGradient("GradientState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientUFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tGradient, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
                tGradientMatEntryOrdinal( tGradientMat, &tMesh );

            auto tGradientMatEntries = tGradientMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tGradientMatEntryOrdinal, tGradient, tGradientMatEntries, tDomain);
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

            // Workset prev state
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tGradient("GradientState", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsGradientUFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tControlWS, tConfigWS, tGradient, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
                tGradientMatEntryOrdinal( tGradientMat, &tMesh );

            auto tGradientMatEntries = tGradientMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tGradientMatEntryOrdinal, tGradient, tGradientMatEntries);
        }

        return tGradientMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_v(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aStateDot,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        using ConfigScalar   = typename GradientV::ConfigScalarType;
        using StateScalar    = typename GradientV::StateScalarType;
        using StateDotScalar = typename GradientV::StateDotScalarType;
        using ControlScalar  = typename GradientV::ControlScalarType;
        using ResultScalar   = typename GradientV::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tGradientMat =
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

            // Workset prev state
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells,mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tGradient("GradientState", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientVFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tGradient, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
                tGradientMatEntryOrdinal( tGradientMat, &tMesh );

            auto tGradientMatEntries = tGradientMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tGradientMatEntryOrdinal, tGradient, tGradientMatEntries, tDomain);
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

            // Workset prev state
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells,mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
            Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tGradient("GradientState", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsGradientVFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tControlWS, tConfigWS, tGradient, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumDofsPerNode, mNumDofsPerNode>
                tGradientMatEntryOrdinal( tGradientMat, &tMesh );

            auto tGradientMatEntries = tGradientMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleJacobianFad
                (mNumDofsPerCell, mNumDofsPerCell, tGradientMatEntryOrdinal, tGradient, tGradientMatEntries);
        }

        return tGradientMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_z(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aStateDot,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        using ConfigScalar   = typename GradientZ::ConfigScalarType;
        using StateScalar    = typename GradientZ::StateScalarType;
        using StateDotScalar = typename GradientZ::StateDotScalarType;
        using ControlScalar  = typename GradientZ::ControlScalarType;
        using ResultScalar   = typename GradientZ::ResultScalarType;

        // create return matrix
        //
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tGradientMat =
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

            // Workset prev state
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tGradient("GradientControl", tNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mGradientZFunctions.at(tName)->evaluate( tStateWS, tStateDotWS, tControlWS, tConfigWS, tGradient, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
              tGradientMatEntryOrdinal( tGradientMat, &tMesh );

            auto tGradientMatEntries = tGradientMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumNodesPerCell, tGradientMatEntryOrdinal, tGradient, tGradientMatEntries, tDomain);
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

            // Workset prev state
            //
            Plato::ScalarMultiVectorT<StateDotScalar> tStateDotWS("StateDot Workset", mNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aStateDot, tStateDotWS);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tGradient("GradientControl", mNumCells, mNumDofsPerCell);

            // evaluate function
            //
            mBoundaryLoadsGradientZFunction->evaluate_boundary(mSpatialModel, tStateWS, tStateDotWS, tControlWS, tConfigWS, tGradient, aTimeStep );

            // assembly to return matrix
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControl, mNumDofsPerNode>
              tGradientMatEntryOrdinal( tGradientMat, &tMesh );

            auto tGradientMatEntries = tGradientMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumDofsPerCell, mNumNodesPerCell, tGradientMatEntryOrdinal, tGradient, tGradientMatEntries);
        }
        return (tGradientMat);
    }
};
// class VectorFunction

} // namespace Parabolic

} // namespace Plato

#endif

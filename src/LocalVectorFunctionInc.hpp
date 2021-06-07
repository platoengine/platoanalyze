#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>

#include <Teuchos_ParameterList.hpp>

#include "WorksetBase.hpp"
#include "AbstractLocalVectorFunctionInc.hpp"
#include "SimplexFadTypes.hpp"
#include "TimeData.hpp"

namespace Plato
{

/******************************************************************************/
/*! local vector function class

   This class takes as a template argument a vector function in the form:

   H = H(U^k, U^{k-1}, C^k, C^{k-1}, X)

   and manages the evaluation of the function and derivatives wrt global state, U^k; 
   previous global state, U^{k-1}; local state, C^k; 
   previous local state, C^{k-1}; and control, X.
*/
/******************************************************************************/
template<typename PhysicsT>
class LocalVectorFunctionInc
{
private:
    static constexpr auto mNumGlobalDofsPerCell = PhysicsT::mNumDofsPerCell;
    static constexpr auto mNumLocalDofsPerCell = PhysicsT::mNumLocalDofsPerCell;
    static constexpr auto mNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode;
    static constexpr auto mNumSpatialDims = PhysicsT::mNumSpatialDims;
    static constexpr auto mNumControl = PhysicsT::mNumControl;

    const Plato::OrdinalType mNumNodes;
    const Plato::OrdinalType mNumCells;

    using Residual        = typename Plato::Evaluation<PhysicsT>::Residual;
    using GlobalJacobian  = typename Plato::Evaluation<PhysicsT>::Jacobian;
    using GlobalJacobianP = typename Plato::Evaluation<PhysicsT>::JacobianP;
    using LocalJacobian   = typename Plato::Evaluation<PhysicsT>::LocalJacobian;
    using LocalJacobianP  = typename Plato::Evaluation<PhysicsT>::LocalJacobianP;
    using GradientX       = typename Plato::Evaluation<PhysicsT>::GradientX;
    using GradientZ       = typename Plato::Evaluation<PhysicsT>::GradientZ;

    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;

    Plato::WorksetBase<PhysicsT> mWorksetBase;

    using ResidualFunction   = std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<Residual>>;
    using JacobianUFunction  = std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GlobalJacobian>>;
    using JacobianUPFunction = std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GlobalJacobianP>>;
    using JacobianCFunction  = std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<LocalJacobian>>;
    using JacobianCPFunction = std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<LocalJacobianP>>;
    using JacobianXFunction  = std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GradientX>>;
    using JacobianZFunction  = std::shared_ptr<Plato::AbstractLocalVectorFunctionInc<GradientZ>>;

    std::map<std::string, ResidualFunction>   mResidualFunctions;
    std::map<std::string, JacobianUFunction>  mJacobianUFunctions;
    std::map<std::string, JacobianUPFunction> mJacobianUPFunctions;
    std::map<std::string, JacobianCFunction>  mJacobianCFunctions;
    std::map<std::string, JacobianCPFunction> mJacobianCPFunctions;
    std::map<std::string, JacobianXFunction>  mJacobianXFunctions;
    std::map<std::string, JacobianZFunction>  mJacobianZFunctions;

    ResidualFunction   mBoundaryLoadsResidualFunction;
    JacobianUFunction  mBoundaryLoadsJacobianUFunction;
    JacobianUPFunction mBoundaryLoadsJacobianUPFunction;
    JacobianCFunction  mBoundaryLoadsJacobianCFunction;
    JacobianCPFunction mBoundaryLoadsJacobianCPFunction;
    JacobianXFunction  mBoundaryLoadsJacobianXFunction;
    JacobianZFunction  mBoundaryLoadsJacobianZFunction;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap;

public:

    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aSpatialModel Plato Analyze spatial model
    * \param [in] aDataMap problem-specific data map
    * \param [in] aParamList Teuchos parameter list with input data
    *
    ******************************************************************************/
    LocalVectorFunctionInc(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aParamList
    ) :
        mWorksetBase  (aSpatialModel.Mesh),
        mNumCells     (aSpatialModel.Mesh.nelems()),
        mNumNodes     (aSpatialModel.Mesh.nverts()),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap)
    {
        typename PhysicsT::FunctionFactory tFunctionFactory;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();
            mResidualFunctions[tName]   = tFunctionFactory.template createLocalVectorFunctionInc<Residual>        (tDomain, aDataMap, aParamList);
            mJacobianUFunctions[tName]  = tFunctionFactory.template createLocalVectorFunctionInc<GlobalJacobian>  (tDomain, aDataMap, aParamList);
            mJacobianUPFunctions[tName] = tFunctionFactory.template createLocalVectorFunctionInc<GlobalJacobianP> (tDomain, aDataMap, aParamList);
            mJacobianCFunctions[tName]  = tFunctionFactory.template createLocalVectorFunctionInc<LocalJacobian>   (tDomain, aDataMap, aParamList);
            mJacobianCPFunctions[tName] = tFunctionFactory.template createLocalVectorFunctionInc<LocalJacobianP>  (tDomain, aDataMap, aParamList);
            mJacobianXFunctions[tName]  = tFunctionFactory.template createLocalVectorFunctionInc<GradientX>       (tDomain, aDataMap, aParamList);
            mJacobianZFunctions[tName]  = tFunctionFactory.template createLocalVectorFunctionInc<GradientZ>       (tDomain, aDataMap, aParamList);
        }
    }

    /**************************************************************************//**
    *
    * \brief Constructor
    * \param [in] aMesh mesh data base
    * \param [in] aDataMap problem-specific data map
    *
    ******************************************************************************/
    LocalVectorFunctionInc(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        mWorksetBase  (aSpatialModel.Mesh),
        mNumCells     (aSpatialModel.Mesh.nelems()),
        mNumNodes     (aSpatialModel.Mesh.nverts()),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap)
    {
    }

    /**************************************************************************//**
    *
    * \brief Return total number of local degrees of freedom
    *
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return mNumCells * mNumLocalDofsPerCell;
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
    decltype(mNumGlobalDofsPerCell) numGlobalDofsPerCell() const
    {
        return mNumGlobalDofsPerCell;
    }

    /***********************************************************************//**
     * \brief Return number of local degrees of freedom per cell.
     * \return number of local degrees of freedom per cell
    ***************************************************************************/
    decltype(mNumLocalDofsPerCell) numLocalDofsPerCell() const
    {
        return mNumLocalDofsPerCell;
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
    * \brief Update the local state variables
    * \param [in]  aGlobalState global state at current time step
    * \param [in]  aPrevGlobalState global state at previous time step
    * \param [out] aLocalState local state at current time step
    * \param [in]  aPrevLocalState local state at previous time step
    * \param [in]  aControl control parameters
    * \param [in]  aTimeData time data
    ******************************************************************************/
    void
    updateLocalState(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aPrevGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aPrevLocalState,
        const Plato::ScalarVector & aControl,
        const Plato::TimeData     & aTimeData
    ) const
    {
        using ConfigScalar         = typename Residual::ConfigScalarType;
        using StateScalar          = typename Residual::StateScalarType;
        using PrevStateScalar      = typename Residual::PrevStateScalarType;
        using LocalStateScalar     = typename Residual::LocalStateScalarType;
        using PrevLocalStateScalar = typename Residual::PrevLocalStateScalarType;
        using ControlScalar        = typename Residual::ControlScalarType;
        using ResultScalar         = typename Residual::ResultScalarType;

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset global state
            //
            Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset prev global state
            //
            Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev Global State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset prev local state
            //
            Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Prev Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // Workset config
            // 
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // update the local state variables
            //
            mResidualFunctions.at(tName)->updateLocalState( tGlobalStateWS, tPrevGlobalStateWS, 
                                                            tLocalStateWS , tPrevLocalStateWS,
                                                            tControlWS    , tConfigWS, 
                                                            aTimeData );

            Plato::flatten_vector_workset<mNumLocalDofsPerCell>(tDomain, tLocalStateWS, aLocalState);
        }
    }

    /**************************************************************************//**
    * \brief Compute the local residual vector
    * \param [in]  aGlobalState global state at current time step
    * \param [in]  aPrevGlobalState global state at previous time step
    * \param [in]  aLocalState local state at current time step
    * \param [in]  aPrevLocalState local state at previous time step
    * \param [in]  aControl control parameters
    * \param [in]  aTimeData time data
    * \return local residual vector
    ******************************************************************************/
    Plato::ScalarVectorT<typename Residual::ResultScalarType>
    value(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aPrevGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aPrevLocalState,
        const Plato::ScalarVector & aControl,
        const Plato::TimeData     & aTimeData
    ) const
    {
        using ConfigScalar         = typename Residual::ConfigScalarType;
        using StateScalar          = typename Residual::StateScalarType;
        using PrevStateScalar      = typename Residual::PrevStateScalarType;
        using LocalStateScalar     = typename Residual::LocalStateScalarType;
        using PrevLocalStateScalar = typename Residual::PrevLocalStateScalarType;
        using ControlScalar        = typename Residual::ControlScalarType;
        using ResultScalar         = typename Residual::ResultScalarType;

        const auto tTotalNumLocalDofs = mNumCells * mNumLocalDofsPerCell;
        Plato::ScalarVector tResidualVector("Residual Vector", tTotalNumLocalDofs);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset prev global state
            //
            Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset prev local state
            //
            Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Prev Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResidualWS("Residual", tNumCells, mNumLocalDofsPerCell);

            // evaluate function
            //
            mResidualFunctions.at(tName)->evaluate(tGlobalStateWS, tPrevGlobalStateWS,
                                                   tLocalStateWS,  tPrevLocalStateWS,
                                                   tControlWS, tConfigWS, tResidualWS, aTimeData);

            Plato::flatten_vector_workset<mNumLocalDofsPerCell>(tDomain, tResidualWS, tResidualVector);
        }
        return tResidualVector;
    }

    /**************************************************************************//**
    * \brief Compute the local residual workset
    * \param [in]  aGlobalState global state at current time step
    * \param [in]  aPrevGlobalState global state at previous time step
    * \param [in]  aLocalState local state at current time step
    * \param [in]  aPrevLocalState local state at previous time step
    * \param [in]  aControl control parameters
    * \param [in]  aTimeData time data
    * \return local residual workset
    ******************************************************************************/
    Plato::ScalarMultiVectorT<typename Residual::ResultScalarType>
    valueWorkSet(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aPrevGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aPrevLocalState,
        const Plato::ScalarVector & aControl,
        const Plato::TimeData     & aTimeData
    ) const
    {
        using ConfigScalar         = typename Residual::ConfigScalarType;
        using StateScalar          = typename Residual::StateScalarType;
        using PrevStateScalar      = typename Residual::PrevStateScalarType;
        using LocalStateScalar     = typename Residual::LocalStateScalarType;
        using PrevLocalStateScalar = typename Residual::PrevLocalStateScalarType;
        using ControlScalar        = typename Residual::ControlScalarType;
        using ResultScalar         = typename Residual::ResultScalarType;

        // create return view
        //
        Plato::ScalarMultiVectorT<ResultScalar> tFullResidualWS("Residual", mNumCells, mNumLocalDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset prev global state
            //
            Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset prev local state
            //
            Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Prev Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResidualWS("Residual", tNumCells, mNumLocalDofsPerCell);

            // evaluate function
            //
            mResidualFunctions.at(tName)->evaluate(tGlobalStateWS, tPrevGlobalStateWS,
                                                   tLocalStateWS,  tPrevLocalStateWS,
                                                   tControlWS, tConfigWS, tResidualWS, aTimeData);

            Plato::assemble_vector_workset<mNumLocalDofsPerCell>(tDomain, tResidualWS, tFullResidualWS);
        }
        return tFullResidualWS;
    }

    /**************************************************************************//**
    * \brief Compute the gradient wrt configuration of the local residual vector
    * \param [in]  aGlobalState global state at current time step
    * \param [in]  aPrevGlobalState global state at previous time step
    * \param [in]  aLocalState local state at current time step
    * \param [in]  aPrevLocalState local state at previous time step
    * \param [in]  aControl control parameters
    * \param [in]  aTimeData time data
    * \return gradient wrt configuration of the local residual vector
    ******************************************************************************/
    Plato::ScalarArray3D
    gradient_x(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aPrevGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aPrevLocalState,
        const Plato::ScalarVector & aControl,
        const Plato::TimeData     & aTimeData
    ) const
    {
        using ConfigScalar         = typename GradientX::ConfigScalarType;
        using StateScalar          = typename GradientX::StateScalarType;
        using PrevStateScalar      = typename GradientX::PrevStateScalarType;
        using LocalStateScalar     = typename GradientX::LocalStateScalarType;
        using PrevLocalStateScalar = typename GradientX::PrevLocalStateScalarType;
        using ControlScalar        = typename GradientX::ControlScalarType;
        using ResultScalar         = typename GradientX::ResultScalarType;

        constexpr auto tNumConfigDofsPerCell = mNumNodesPerCell * mNumSpatialDims;
        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Configuration", mNumCells, mNumLocalDofsPerCell, tNumConfigDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aGlobalState, tGlobalStateWS);

            // Workset prev global state
            //
            Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset prev local state
            //
            Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Prev Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Configuration Workset", tNumCells, mNumLocalDofsPerCell);

            // evaluate function
            //
            mJacobianXFunctions.at(tName)->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                                    tLocalStateWS,  tPrevLocalStateWS, 
                                                    tControlWS, tConfigWS, tJacobianWS, aTimeData);

            Plato::transform_ad_type_to_pod_3Dview<mNumLocalDofsPerCell, tNumConfigDofsPerCell>(tDomain, tJacobianWS, tOutputJacobian);
        }
        return tOutputJacobian;
    }

    /**************************************************************************//**
    * \brief Compute the gradient wrt global state of the local residual vector
    * \param [in]  aGlobalState global state at current time step
    * \param [in]  aPrevGlobalState global state at previous time step
    * \param [in]  aLocalState local state at current time step
    * \param [in]  aPrevLocalState local state at previous time step
    * \param [in]  aControl control parameters
    * \param [in]  aTimeData time data
    * \return gradient wrt global state of the local residual vector
    ******************************************************************************/
    ScalarArray3D
    gradient_u(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aPrevGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aPrevLocalState,
        const Plato::ScalarVector & aControl,
        const Plato::TimeData     & aTimeData
    ) const
    {
        using ConfigScalar         = typename GlobalJacobian::ConfigScalarType;
        using StateScalar          = typename GlobalJacobian::StateScalarType;
        using PrevStateScalar      = typename GlobalJacobian::PrevStateScalarType;
        using LocalStateScalar     = typename GlobalJacobian::LocalStateScalarType;
        using PrevLocalStateScalar = typename GlobalJacobian::PrevLocalStateScalarType;
        using ControlScalar        = typename GlobalJacobian::ControlScalarType;
        using ResultScalar         = typename GlobalJacobian::ResultScalarType;

        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Current Global State", mNumCells, mNumLocalDofsPerCell, mNumGlobalDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset prev global state
            //
            Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset prev local state
            //
            Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Prev Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Current Global State Workset", tNumCells, mNumLocalDofsPerCell);

            // evaluate function
            //
            mJacobianUFunctions.at(tName)->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                                    tLocalStateWS,  tPrevLocalStateWS, 
                                                    tControlWS, tConfigWS, tJacobianWS, aTimeData);

            Plato::transform_ad_type_to_pod_3Dview<mNumLocalDofsPerCell, mNumGlobalDofsPerCell>(tDomain, tJacobianWS, tOutputJacobian);
        }
        return tOutputJacobian;
    }

    /**************************************************************************//**
    * \brief Compute the gradient wrt previous global state of the local residual vector
    * \param [in]  aGlobalState global state at current time step
    * \param [in]  aPrevGlobalState global state at previous time step
    * \param [in]  aLocalState local state at current time step
    * \param [in]  aPrevLocalState local state at previous time step
    * \param [in]  aControl control parameters
    * \param [in]  aTimeData time data
    * \return gradient wrt previous global state of the local residual vector
    ******************************************************************************/
    Plato::ScalarArray3D
    gradient_up(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aPrevGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aPrevLocalState,
        const Plato::ScalarVector & aControl,
        const Plato::TimeData     & aTimeData
    ) const
    {
        using ConfigScalar         = typename GlobalJacobianP::ConfigScalarType;
        using StateScalar          = typename GlobalJacobianP::StateScalarType;
        using PrevStateScalar      = typename GlobalJacobianP::PrevStateScalarType;
        using LocalStateScalar     = typename GlobalJacobianP::LocalStateScalarType;
        using PrevLocalStateScalar = typename GlobalJacobianP::PrevLocalStateScalarType;
        using ControlScalar        = typename GlobalJacobianP::ControlScalarType;
        using ResultScalar         = typename GlobalJacobianP::ResultScalarType;

        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Previous Global State", mNumCells, mNumLocalDofsPerCell, mNumGlobalDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset prev global state
            //
            Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset prev local state
            //
            Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Prev Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Previous Global State Workset", tNumCells, mNumLocalDofsPerCell);

            // evaluate function
            //
            mJacobianUPFunctions.at(tName)->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                                     tLocalStateWS,  tPrevLocalStateWS, 
                                                     tControlWS, tConfigWS, tJacobianWS, aTimeData);

            Plato::transform_ad_type_to_pod_3Dview<mNumLocalDofsPerCell, mNumGlobalDofsPerCell>(tDomain, tJacobianWS, tOutputJacobian);
        }
        return tOutputJacobian;
    }

    /**************************************************************************//**
    * \brief Compute the gradient wrt local state of the local residual vector
    * \param [in]  aGlobalState global state at current time step
    * \param [in]  aPrevGlobalState global state at previous time step
    * \param [in]  aLocalState local state at current time step
    * \param [in]  aPrevLocalState local state at previous time step
    * \param [in]  aControl control parameters
    * \param [in]  aTimeData time data
    * \return gradient wrt local state of the local residual vector
    ******************************************************************************/
    Plato::ScalarArray3D
    gradient_c(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aPrevGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aPrevLocalState,
        const Plato::ScalarVector & aControl,
        const Plato::TimeData     & aTimeData
    ) const
    {
        using ConfigScalar         = typename LocalJacobian::ConfigScalarType;
        using StateScalar          = typename LocalJacobian::StateScalarType;
        using PrevStateScalar      = typename LocalJacobian::PrevStateScalarType;
        using LocalStateScalar     = typename LocalJacobian::LocalStateScalarType;
        using PrevLocalStateScalar = typename LocalJacobian::PrevLocalStateScalarType;
        using ControlScalar        = typename LocalJacobian::ControlScalarType;
        using ResultScalar         = typename LocalJacobian::ResultScalarType;

        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Current Local State", mNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset prev global state
            //
            Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset prev local state
            //
            Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Prev Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Current Local State Workset", tNumCells, mNumLocalDofsPerCell);

            // evaluate function
            //
            mJacobianCFunctions.at(tName)->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                                    tLocalStateWS,  tPrevLocalStateWS, 
                                                    tControlWS, tConfigWS, tJacobianWS, aTimeData);

            Plato::transform_ad_type_to_pod_3Dview<mNumLocalDofsPerCell, mNumLocalDofsPerCell>(tDomain, tJacobianWS, tOutputJacobian);
        }
        return tOutputJacobian;
    }

    /**************************************************************************//**
    * \brief Compute the gradient wrt previous local state of the local residual vector
    * \param [in]  aGlobalState global state at current time step
    * \param [in]  aPrevGlobalState global state at previous time step
    * \param [in]  aLocalState local state at current time step
    * \param [in]  aPrevLocalState local state at previous time step
    * \param [in]  aControl control parameters
    * \param [in]  aTimeData time data
    * \return gradient wrt previous local state of the local residual vector
    ******************************************************************************/
    Plato::ScalarArray3D
    gradient_cp(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aPrevGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aPrevLocalState,
        const Plato::ScalarVector & aControl,
        const Plato::TimeData     & aTimeData
    ) const
    {
        using ConfigScalar         = typename LocalJacobianP::ConfigScalarType;
        using StateScalar          = typename LocalJacobianP::StateScalarType;
        using PrevStateScalar      = typename LocalJacobianP::PrevStateScalarType;
        using LocalStateScalar     = typename LocalJacobianP::LocalStateScalarType;
        using PrevLocalStateScalar = typename LocalJacobianP::PrevLocalStateScalarType;
        using ControlScalar        = typename LocalJacobianP::ControlScalarType;
        using ResultScalar         = typename LocalJacobianP::ResultScalarType;

        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Previous Local State", mNumCells, mNumLocalDofsPerCell, mNumLocalDofsPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset prev global state
            //
            Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset prev local state
            //
            Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Prev Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create return view
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Previous Local State Workset", tNumCells, mNumLocalDofsPerCell);

            // evaluate function
            //
            mJacobianCPFunctions.at(tName)->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                                     tLocalStateWS,  tPrevLocalStateWS,
                                                     tControlWS, tConfigWS, tJacobianWS, aTimeData);

            Plato::transform_ad_type_to_pod_3Dview<mNumLocalDofsPerCell, mNumLocalDofsPerCell>(tDomain, tJacobianWS, tOutputJacobian);
        }
        return tOutputJacobian;
    }

    /**************************************************************************//**
    * \brief Compute the gradient wrt control of the local residual vector
    * \param [in]  aGlobalState global state at current time step
    * \param [in]  aPrevGlobalState global state at previous time step
    * \param [in]  aLocalState local state at current time step
    * \param [in]  aPrevLocalState local state at previous time step
    * \param [in]  aControl control parameters
    * \param [in]  aTimeData time data
    * \return gradient wrt control of the local residual vector
    ******************************************************************************/
    Plato::ScalarArray3D
    gradient_z(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aPrevGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aPrevLocalState,
        const Plato::ScalarVector & aControl,
        const Plato::TimeData     & aTimeData
    ) const
    {
        using ConfigScalar         = typename GradientZ::ConfigScalarType;
        using StateScalar          = typename GradientZ::StateScalarType;
        using PrevStateScalar      = typename GradientZ::PrevStateScalarType;
        using LocalStateScalar     = typename GradientZ::LocalStateScalarType;
        using PrevLocalStateScalar = typename GradientZ::PrevLocalStateScalarType;
        using ControlScalar        = typename GradientZ::ControlScalarType;
        using ResultScalar         = typename GradientZ::ResultScalarType;

        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Control", mNumCells, mNumLocalDofsPerCell, mNumNodesPerCell);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            mWorksetBase.worksetConfig(tConfigWS, tDomain);

            // Workset global state
            //
            Plato::ScalarMultiVectorT<StateScalar> tGlobalStateWS("State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset prev global state
            //
            Plato::ScalarMultiVectorT<PrevStateScalar> tPrevGlobalStateWS("Prev State Workset", tNumCells, mNumGlobalDofsPerCell);
            mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset prev local state
            //
            Plato::ScalarMultiVectorT<PrevLocalStateScalar> tPrevLocalStateWS("Prev Local State Workset", tNumCells, mNumLocalDofsPerCell);
            mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("Control Workset", tNumCells, mNumNodesPerCell);
            mWorksetBase.worksetControl(aControl, tControlWS, tDomain);

            // create result 
            //
            Plato::ScalarMultiVectorT<ResultScalar> tJacobianWS("Jacobian Control Workset", tNumCells, mNumLocalDofsPerCell);

            // evaluate function 
            //
            mJacobianZFunctions.at(tName)->evaluate(tGlobalStateWS, tPrevGlobalStateWS, 
                                                    tLocalStateWS,  tPrevLocalStateWS, 
                                                    tControlWS, tConfigWS, tJacobianWS, aTimeData);

            Plato::transform_ad_type_to_pod_3Dview<mNumLocalDofsPerCell, mNumNodesPerCell>(tDomain, tJacobianWS, tOutputJacobian);
        }
        return tOutputJacobian;
    }

    /***************************************************************************//**
     * \brief Update physics-based parameters within a frequency of optimization iterations
     * \param [in] aGlobalStates global states for all time steps
     * \param [in] aLocalStates  local states for all time steps
     * \param [in] aControls     current controls, i.e. design variables
     * \param [in] aTimeData     current time data
    *******************************************************************************/
    void updateProblem(
        const Plato::ScalarMultiVector & aGlobalStates,
        const Plato::ScalarMultiVector & aLocalStates,
        const Plato::ScalarVector      & aControls,
        const Plato::TimeData          & aTimeData
    ) const
    {
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            mResidualFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeData);
            mJacobianUFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeData);
            mJacobianUPFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeData);
            mJacobianCFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeData);
            mJacobianCPFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeData);
            mJacobianXFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeData);
            mJacobianZFunctions.at(tName)->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeData);
        }
    }
};
// class LocalVectorFunctionInc

} // namespace Plato

#include "Plasticity.hpp"
#include "ThermoPlasticity.hpp"

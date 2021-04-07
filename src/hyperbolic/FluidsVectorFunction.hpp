/*
 * FluidsVectorFunction.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include "BLAS2.hpp"
#include "Assembly.hpp"

#include "hyperbolic/FluidsWorkSetsUtils.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"
#include "hyperbolic/AbstractVectorFunction.hpp"

namespace Plato
{

namespace Fluids
{

/******************************************************************************/
/*! vector function class

   This class takes as a template argument a vector function in the form:

   \f$ F = F(\phi, U^k, P^k, T^k, X) \f$

   and manages the evaluation of the function and derivatives with respect to
   control, \f$\phi\f$, momentum state, \f$ U^k \f$, mass state, \f$ P^k \f$,
   energy state, \f$ T^k \f$, and configuration, \f$ X \f$.

*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunction
{
private:
    static constexpr auto mNumDofsPerNode = PhysicsT::mNumDofsPerNode; /*!< number of degrees of freedom per node */
    static constexpr auto mNumDofsPerCell = PhysicsT::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    static constexpr auto mNumSpatialDims        = PhysicsT::SimplexT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell       = PhysicsT::SimplexT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumPressDofsPerCell   = PhysicsT::SimplexT::mNumMassDofsPerCell;     /*!< number of mass dofs per cell */
    static constexpr auto mNumTempDofsPerCell    = PhysicsT::SimplexT::mNumEnergyDofsPerCell;   /*!< number of energy dofs per cell */
    static constexpr auto mNumVelDofsPerCell     = PhysicsT::SimplexT::mNumMomentumDofsPerCell; /*!< number of momentum dofs per cell */
    static constexpr auto mNumControlDofsPerCell = PhysicsT::SimplexT::mNumControlDofsPerCell;  /*!< number of design variable per cell */
    static constexpr auto mNumPressDofsPerNode   = PhysicsT::SimplexT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */
    static constexpr auto mNumTempDofsPerNode    = PhysicsT::SimplexT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumVelDofsPerNode     = PhysicsT::SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumControlDofsPerNode = PhysicsT::SimplexT::mNumControlDofsPerNode;  /*!< number of design variable per node */

    static constexpr auto mNumConfigDofsPerNode = PhysicsT::SimplexT::mNumConfigDofsPerNode; /*!< number of configuration degrees of freedom per cell */
    static constexpr auto mNumConfigDofsPerCell = PhysicsT::SimplexT::mNumConfigDofsPerCell; /*!< number of configuration degrees of freedom per cell */

    // forward automatic differentiation (FAD) evaluation types
    using ResidualEvalT      = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::Residual; /*!< residual FAD evaluation type */
    using GradConfigEvalT    = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradConfig; /*!< gradient wrt configuration FAD evaluation type */
    using GradControlEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradControl; /*!< gradient wrt control FAD evaluation type */
    using GradCurVelEvalT    = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurMomentum; /*!< gradient wrt current momentum FAD evaluation type */
    using GradPrevVelEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPrevMomentum; /*!< gradient wrt previous momentum FAD evaluation type */
    using GradCurTempEvalT   = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurEnergy; /*!< gradient wrt current energy FAD evaluation type */
    using GradPrevTempEvalT  = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPrevEnergy; /*!< gradient wrt previous energy FAD evaluation type */
    using GradCurPressEvalT  = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradCurMass; /*!< gradient wrt current mass FAD evaluation type */
    using GradPrevPressEvalT = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPrevMass; /*!< gradient wrt previous mass FAD evaluation type */
    using GradPredictorEvalT = typename Plato::Fluids::Evaluation<typename PhysicsT::SimplexT>::GradPredictor; /*!< gradient wrt momentum predictor FAD evaluation type */

    // element residual vector function types
    using ResidualFuncT      = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, ResidualEvalT>>; /*!< vector function of type residual */
    using GradConfigFuncT    = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradConfigEvalT>>; /*!< vector function of type gradient wrt configuration */
    using GradControlFuncT   = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradControlEvalT>>; /*!< vector function of type gradient wrt control */
    using GradCurVelFuncT    = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradCurVelEvalT>>; /*!< vector function of type gradient wrt current velocity */
    using GradPrevVelFuncT   = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPrevVelEvalT>>; /*!< vector function of type gradient wrt previous velocity */
    using GradCurTempFuncT   = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradCurTempEvalT>>; /*!< vector function of type gradient wrt current temperature */
    using GradPrevTempFuncT  = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPrevTempEvalT>>; /*!< vector function of type gradient wrt previous temperature */
    using GradCurPressFuncT  = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradCurPressEvalT>>; /*!< vector function of type gradient wrt current pressure */
    using GradPrevPressFuncT = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPrevPressEvalT>>; /*!< vector function of type gradient wrt previous pressure */
    using GradPredictorFuncT = std::shared_ptr<Plato::Fluids::AbstractVectorFunction<PhysicsT, GradPredictorEvalT>>; /*!< vector function of type gradient wrt velocity predictor */

    // element vector functions per element block, i.e. domain
    std::unordered_map<std::string, ResidualFuncT>      mResidualFuncs; /*!< vector function list of type residual */
    std::unordered_map<std::string, GradConfigFuncT>    mGradConfigFuncs; /*!< vector function list of type gradient wrt configuration */
    std::unordered_map<std::string, GradControlFuncT>   mGradControlFuncs; /*!< vector function list of type gradient wrt control */
    std::unordered_map<std::string, GradCurVelFuncT>    mGradCurVelFuncs; /*!< vector function list of type gradient wrt current velocity */
    std::unordered_map<std::string, GradPrevVelFuncT>   mGradPrevVelFuncs; /*!< vector function list of type gradient wrt previous velocity */
    std::unordered_map<std::string, GradCurTempFuncT>   mGradCurTempFuncs; /*!< vector function list of type gradient wrt current temperature */
    std::unordered_map<std::string, GradPrevTempFuncT>  mGradPrevTempFuncs; /*!< vector function list of type gradient wrt previous temperature */
    std::unordered_map<std::string, GradCurPressFuncT>  mGradCurPressFuncs; /*!< vector function list of type gradient wrt current pressure */
    std::unordered_map<std::string, GradPrevPressFuncT> mGradPrevPressFuncs; /*!< vector function list of type gradient wrt previous pressure */
    std::unordered_map<std::string, GradPredictorFuncT> mGradPredictorFuncs; /*!< vector function list of type gradient wrt velocity predictor */

    Plato::DataMap& mDataMap; /*!< output database */
    const Plato::SpatialModel& mSpatialModel; /*!< spatial model metadata - owns mesh metadata for all the domains, i.e. element blocks */
    Plato::LocalOrdinalMaps<PhysicsT> mLocalOrdinalMaps; /*!< local-to-global ordinal maps */
    Plato::VectorEntryOrdinal<mNumSpatialDims,mNumDofsPerNode> mStateOrdinalsMap; /*!< local-to-global ordinal vector field map */

public:
    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aTag     vector function tag/type
    * \param [in] aModel   struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap output database
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
    VectorFunction
    (const std::string            & aTag,
     const Plato::SpatialModel    & aModel,
           Plato::DataMap         & aDataMap,
           Teuchos::ParameterList & aInputs) :
        mSpatialModel(aModel),
        mDataMap(aDataMap),
        mLocalOrdinalMaps(aModel.Mesh),
        mStateOrdinalsMap(&aModel.Mesh)
    {
        this->initialize(aTag, aDataMap, aInputs);
    }

    /**************************************************************************//**
    * \fn integer getNumSpatialDims
    * \brief Return number of spatial dimensions.
    * \return number of spatial dimensions (integer)
    ******************************************************************************/
    decltype(mNumSpatialDims) getNumSpatialDims() const
    {
        return mNumSpatialDims;
    }

    /**************************************************************************//**
    * \fn integer getNumDofsPerCell
    * \brief Return number of degrees of freedom per cell.
    * \return degrees of freedom per cell (integer)
    ******************************************************************************/
    decltype(mNumDofsPerCell) getNumDofsPerCell() const
    {
        return mNumDofsPerCell;
    }

    /**************************************************************************//**
    * \fn integer getNumDofsPerNode
    * \brief Return number of degrees of freedom per node.
    * \return degrees of freedom per node (integer)
    ******************************************************************************/
    decltype(mNumDofsPerNode) getNumDofsPerNode() const
    {
        return mNumDofsPerNode;
    }

    /**************************************************************************//**
    * \fn Plato::ScalarVector value
    * \brief Return vector function residual.
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return vector function residual
    ******************************************************************************/
    Plato::ScalarVector value
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aPrimal)
    const
    {
        using ResultScalarT = typename ResidualEvalT::ResultScalarType;

        auto tNumNodes = mSpatialModel.Mesh.nverts();
        auto tLength = tNumNodes * mNumDofsPerNode;
        Plato::ScalarVector tReturnValue("Assembled Residual", tLength);

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
                (tDomain, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tNumCells = tDomain.numCells();
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mResidualFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell,mNumDofsPerNode>(tDomain, mStateOrdinalsMap, tResultWS, tReturnValue);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
                (tNumCells, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mResidualFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell,mNumDofsPerNode>(tNumCells, mStateOrdinalsMap, tResultWS, tReturnValue);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mResidualFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_residual<mNumNodesPerCell,mNumDofsPerNode>(tNumCells, mStateOrdinalsMap, tResultWS, tReturnValue);
        }

        return tReturnValue;
    }

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientConfig
    * \brief Return gradient of residual with respet to (wrt) configuration variables (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt configuration
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientConfig
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aPrimal)
    const
    {
        using ResultScalarT = typename GradConfigEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumConfigDofsPerNode, mNumDofsPerNode>(&tMesh);

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradConfigEvalT>
                (tDomain, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradConfigFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumConfigDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradConfigEvalT>
                (tNumCells, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradConfigFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumConfigDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradConfigFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumConfigDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientControl
    * \brief Return gradient of residual with respet to (wrt) control variables (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt control
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientControl
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables)
    const
    {
        using ResultScalarT = typename GradControlEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumControlDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradControlEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradControlFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumControlDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradControlEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradControlFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumControlDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumControlDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradControlFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumControlDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientPredictor
    * \brief Return gradient of residual with respet to (wrt) predictor (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt predictor
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPredictor
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aPrimal)
    const
    {
        using ResultScalarT = typename GradPredictorEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradPredictorEvalT>
                (tDomain, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPredictorFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradPredictorEvalT>
                (tNumCells, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPredictorFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPredictorFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientPreviousVel
    * \brief Return gradient of residual with respet to (wrt) previous velocity (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt previous velocity
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aPrimal)
    const
    {
        using ResultScalarT = typename GradPrevVelEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradPrevVelEvalT>
                (tDomain, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevVelFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradPrevVelEvalT>
                (tNumCells, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevVelFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevVelFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientPreviousPress
    * \brief Return gradient of residual with respet to (wrt) previous pressure (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt previous pressure
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aPrimal)
    const
    {
        using ResultScalarT = typename GradPrevPressEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumPressDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradPrevPressEvalT>
                (tDomain, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevPressFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradPrevPressEvalT>
                (tNumCells, aControls, aPrimal, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevPressFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevPressFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientPreviousTemp
    * \brief Return gradient of residual with respet to (wrt) previous temperature (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt previous temperature
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientPreviousTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradPrevTempEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumTempDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradPrevTempEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevTempFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradPrevTempEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradPrevTempFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradPrevTempFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientCurrentVel
    * \brief Return gradient of residual with respet to (wrt) current velocity (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt current velocity
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentVel
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradCurVelEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumVelDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradCurVelEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurVelFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradCurVelEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurVelFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumVelDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurVelFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumVelDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientCurrentPress
    * \brief Return gradient of residual with respet to (wrt) current pressure (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt current pressure
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentPress
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradCurPressEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumPressDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradCurPressEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurPressFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntires = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntires);
        }

        // evaluate boundary forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradCurPressEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate prescribed forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurPressFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumPressDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurPressFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumPressDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

    /**************************************************************************//**
    * \fn Teuchos::RCP<Plato::CrsMatrixType> gradientCurrentTemp
    * \brief Return gradient of residual with respet to (wrt) current temperature (i.e. Jacobian).
    * \param [in] aControls control variables
    * \param [in] aPrimal   primal state database
    * \return Jacobian wrt current temperature
    ******************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradientCurrentTemp
    (const Plato::ScalarVector & aControls,
     const Plato::Primal       & aVariables) const
    {
        using ResultScalarT = typename GradCurTempEvalT::ResultScalarType;

        // create return matrix
        auto tMesh = mSpatialModel.Mesh;
        Teuchos::RCP<Plato::CrsMatrixType> tJacobian =
            Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumTempDofsPerNode, mNumDofsPerNode>( &tMesh );

        // evaluate internal forces
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = tDomain.numCells();
            Plato::Fluids::build_vector_function_worksets<GradCurTempEvalT>
                (tDomain, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate internal forces
            auto tName = tDomain.getDomainName();
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurTempFuncs.at(tName)->evaluate(tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tDomain, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        // evaluate prescribed forces
        {
            Plato::WorkSets tInputWorkSets;
            auto tNumCells = mSpatialModel.Mesh.nelems();
            Plato::Fluids::build_vector_function_worksets<GradCurTempEvalT>
                (tNumCells, aControls, aVariables, mLocalOrdinalMaps, tInputWorkSets);

            // evaluate boundary forces
            Plato::ScalarMultiVectorT<ResultScalarT> tResultWS("Cells Results", tNumCells, mNumDofsPerCell);
            mGradCurTempFuncs.begin()->second->evaluatePrescribed(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumTempDofsPerNode, mNumDofsPerNode> tJacEntryOrdinal(tJacobian, &tMesh);
            auto tJacobianMatrixEntries = tJacobian->entries();
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);

            // evaluate balancing forces
            Plato::blas2::fill(0.0, tResultWS);
            mGradCurTempFuncs.begin()->second->evaluateBoundary(mSpatialModel, tInputWorkSets, tResultWS);
            Plato::assemble_transpose_jacobian(tNumCells, mNumDofsPerCell, mNumTempDofsPerCell,
                                               tJacEntryOrdinal, tResultWS, tJacobianMatrixEntries);
        }

        return tJacobian;
    }

private:
    /**************************************************************************//**
    * \brief Initialize member metadata.
    * \param [in] aTag     vector function tag/type
    * \param [in] aDataMap output database
    * \param [in] aInputs  input file metadata
    ******************************************************************************/
    void initialize
    (const std::string      & aTag,
     Plato::DataMap         & aDataMap,
     Teuchos::ParameterList & aInputs)
    {
        typename PhysicsT::FunctionFactory tVecFuncFactory;
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mResidualFuncs[tName]  = tVecFuncFactory.template createVectorFunction<PhysicsT, ResidualEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradControlFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradControlEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradConfigFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradConfigEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradCurPressFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradCurPressEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradPrevPressFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPrevPressEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradCurTempFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradCurTempEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradPrevTempFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPrevTempEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradCurVelFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradCurVelEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradPrevVelFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPrevVelEvalT>
                (aTag, tDomain, aDataMap, aInputs);

            mGradPredictorFuncs[tName] = tVecFuncFactory.template createVectorFunction<PhysicsT, GradPredictorEvalT>
                (aTag, tDomain, aDataMap, aInputs);
        }
    }
};
// class VectorFunction

}
// namespace Fluids

}
// namespace Plato

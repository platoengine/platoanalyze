/*
 * GlobalVectorFunctionInc.hpp
 *
 *  Created on: Mar 1, 2020
 */

#pragma once

#include "WorksetBase.hpp"
#include "SimplexFadTypes.hpp"
#include "InfinitesimalStrainPlasticity.hpp"
#include "AbstractGlobalVectorFunctionInc.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Global vector function evaluation manager.  This interface manages
 * the evaluation and assembly of the residual, gradients and Jacobians of a
 * global vector function.
 *
 * \tparam physics type, e.g. infinitesimal strain plasticity
*******************************************************************************/
template<typename PhysicsT>
class GlobalVectorFunctionInc
{
// Private access member data
private:
    using Residual        = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;       /*!< automatic differentiation (AD) type for the residual */
    using GradientX       = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX;      /*!< AD type for the configuration */
    using GradientZ       = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ;      /*!< AD type for the controls */
    using JacobianPgrad   = typename Plato::Evaluation<typename PhysicsT::SimplexT>::JacobianN;      /*!< AD type for the projected pressure gradient */
    using LocalJacobianC   = typename Plato::Evaluation<typename PhysicsT::SimplexT>::LocalJacobian; /*!< AD type for the current local states */
    using LocalJacobianP  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::LocalJacobianP; /*!< AD type for the previous local states */
    using GlobalJacobianC  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian;      /*!< AD type for the current global states */
    using GlobalJacobianP = typename Plato::Evaluation<typename PhysicsT::SimplexT>::JacobianP;      /*!< AD type for the previous global states */

    static constexpr auto mNumControl = PhysicsT::SimplexT::mNumControl;                   /*!< number of control fields, i.e. vectors, number of materials */
    static constexpr auto mNumSpatialDims = PhysicsT::SimplexT::mNumSpatialDims;           /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell = PhysicsT::SimplexT::mNumNodesPerCell;         /*!< number of nodes per cell (i.e. element) */
    static constexpr auto mNumGlobalDofsPerNode = PhysicsT::SimplexT::mNumDofsPerNode;     /*!< number of global degrees of freedom per node */
    static constexpr auto mNumGlobalDofsPerCell = PhysicsT::SimplexT::mNumDofsPerCell;     /*!< number of global degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumLocalDofsPerCell = PhysicsT::SimplexT::mNumLocalDofsPerCell; /*!< number of local degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumNodeStatePerNode = PhysicsT::SimplexT::mNumNodeStatePerNode; /*!< number of pressure gradient degrees of freedom per node */
    static constexpr auto mNumNodeStatePerCell = PhysicsT::SimplexT::mNumNodeStatePerCell; /*!< number of pressure gradient degrees of freedom per cell (i.e. element) */
    static constexpr auto mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;      /*!< number of configuration (i.e. coordinates) degrees of freedom per cell (i.e. element) */

    const Plato::OrdinalType mNumNodes; /*!< total number of nodes */
    const Plato::OrdinalType mNumCells; /*!< total number of cells (i.e. elements)*/

    Plato::DataMap& mDataMap;                  /*!< output data map */
    Plato::WorksetBase<PhysicsT> mWorksetBase; /*!< assembly routine interface */

    std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<Residual>>        mGlobalResidual;         /*!< global residual */
    std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<GradientX>>       mGlobalJacobianX;        /*!< global Jacobian with respect to configuration */
    std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<GradientZ>>       mGlobalJacobianZ;        /*!< global Jacobian with respect to controls */
    std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<LocalJacobianC>>  mGlobalJacobianCC;       /*!< global Jacobian with respect to current local states */
    std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<LocalJacobianP>>  mGlobalJacobianPC;       /*!< global Jacobian with respect to previous local states */
    std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<GlobalJacobianC>> mGlobalJacobianCU;       /*!< global Jacobian with respect to current global states */
    std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<GlobalJacobianP>> mGlobalJacobianPU;       /*!< global Jacobian with respect to previous global states */
    std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<JacobianPgrad>>   mGlobalJacProjPressGrad; /*!< global Jacobian with respect to projected pressure gradient */

// Private access functions
private:
    /***************************************************************************//**
     * \brief Evaluate the residual at each cell/element.
     * \param [in] aCurrentGlobalState current global state
     * \param [in] aPrevGlobalState    previous global state
     * \param [in] aCurrentLocalState  current local state
     * \param [in] aPrevLocalState     previous local state
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control, i.e. design, variables
     * \param [in] aTimeStep           current time step
     * \return residual workset, i.e. residual for each cell
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename Residual::ResultScalarType>
    residualWorkset(const Plato::ScalarVector & aCurrentGlobalState,
                    const Plato::ScalarVector & aPrevGlobalState,
                    const Plato::ScalarVector & aCurrentLocalState,
                    const Plato::ScalarVector & aPrevLocalState,
                    const Plato::ScalarVector & aProjPressGrad,
                    const Plato::ScalarVector & aControls,
                    const Plato::Scalar & aTimeStep) const
    {
        // Workset config
        using ConfigScalar = typename Residual::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Configuration Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename Residual::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tCurrentGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename Residual::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename Residual::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tCurrentLocalStateWS("Current Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename Residual::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename Residual::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar>
            tProjPressGradWS("Projected Pressure Gradient Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aProjPressGrad, tProjPressGradWS);

        // Workset control
        using ControlScalar = typename Residual::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar>
            tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // Workset residual
        using ResultScalar = typename Residual::ResultScalarType;
        Plato::ScalarMultiVectorT<ResultScalar>
            tResidualWS("Residual Workset", mNumCells, mNumGlobalDofsPerCell);

        // Evaluate global residual
        mGlobalResidual->evaluate(tCurrentGlobalStateWS, tPrevGlobalStateWS,
                                  tCurrentLocalStateWS, tPrevLocalStateWS,
                                  tProjPressGradWS, tControlWS, tConfigWS,
                                  tResidualWS, aTimeStep);

        return (tResidualWS);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian of residual with respect to controls
     * \param [in] aCurrentGlobalState current global state
     * \param [in] aPrevGlobalState    previous global state
     * \param [in] aCurrentLocalState  current local state
     * \param [in] aPrevLocalState     previous local state
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control, i.e. design, variables
     * \param [in] aTimeStep           current time step
     * \return Jacobian of residual with respect to controls for each cell
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename GradientZ::ResultScalarType>
    jacobianControlWorkset(const Plato::ScalarVector & aCurrentGlobalState,
                           const Plato::ScalarVector & aPrevGlobalState,
                           const Plato::ScalarVector & aCurrentLocalState,
                           const Plato::ScalarVector & aPrevLocalState,
                           const Plato::ScalarVector & aProjPressGrad,
                           const Plato::ScalarVector & aControls,
                           const Plato::Scalar & aTimeStep) const
    {
        // Workset config
        using ConfigScalar = typename GradientZ::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Configuration Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename GradientZ::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tCurrentGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename GradientZ::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename GradientZ::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tCurrentLocalStateWS("Current Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename GradientZ::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename GradientZ::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar>
            tProjPressGradWS("Projected Pressure Gradient Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aProjPressGrad, tProjPressGradWS);

        // Workset control
        using ControlScalar = typename GradientZ::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar>
            tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // Create Jacobian workset
        using JacobianScalar = typename GradientZ::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar>
            tJacobianWS("Jacobian Control Workset", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt controls
        mGlobalJacobianZ->evaluate(tCurrentGlobalStateWS, tPrevGlobalStateWS,
                                   tCurrentLocalStateWS, tPrevLocalStateWS,
                                   tProjPressGradWS, tControlWS, tConfigWS,
                                   tJacobianWS, aTimeStep);

        return (tJacobianWS);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian of residual with respect to configuration
     * \param [in] aCurrentGlobalState current global state
     * \param [in] aPrevGlobalState    previous global state
     * \param [in] aCurrentLocalState  current local state
     * \param [in] aPrevLocalState     previous local state
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control, i.e. design, variables
     * \param [in] aTimeStep           current time step
     * \return Jacobian of residual with respect to configuration for each cell
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename GradientX::ResultScalarType>
    jacobianConfigurationWorkset(const Plato::ScalarVector & aCurrentGlobalState,
                                 const Plato::ScalarVector & aPrevGlobalState,
                                 const Plato::ScalarVector & aCurrentLocalState,
                                 const Plato::ScalarVector & aPrevLocalState,
                                 const Plato::ScalarVector & aProjPressGrad,
                                 const Plato::ScalarVector & aControls,
                                 const Plato::Scalar & aTimeStep) const
    {
        // Workset config
        using ConfigScalar = typename GradientX::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Configuration Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename GradientX::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tCurrentGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename GradientX::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename GradientX::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tCurrentLocalStateWS("Current Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename GradientX::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename GradientX::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar>
            tProjPressGradWS("Projected Pressure Gradient Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aProjPressGrad, tProjPressGradWS);

        // Workset control
        using ControlScalar = typename GradientX::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar>
            tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // create return view
        using JacobianScalar = typename GradientX::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar>
            tJacobianWS("Jacobian Configuration", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt configuration
        mGlobalJacobianX->evaluate(tCurrentGlobalStateWS, tPrevGlobalStateWS,
                                   tCurrentLocalStateWS, tPrevLocalStateWS,
                                   tProjPressGradWS, tControlWS, tConfigWS,
                                   tJacobianWS, aTimeStep);

        return (tJacobianWS);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian of residual with respect to current global states
     * \param [in] aCurrentGlobalState current global state
     * \param [in] aPrevGlobalState    previous global state
     * \param [in] aCurrentLocalState  current local state
     * \param [in] aPrevLocalState     previous local state
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control, i.e. design, variables
     * \param [in] aTimeStep           current time step
     * \return Jacobian of residual with respect to current global states for each cell
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename GlobalJacobianC::ResultScalarType>
    jacobianCurrentGlobalStateWorkset(const Plato::ScalarVector & aCurrentGlobalState,
                                      const Plato::ScalarVector & aPrevGlobalState,
                                      const Plato::ScalarVector & aCurrentLocalState,
                                      const Plato::ScalarVector & aPrevLocalState,
                                      const Plato::ScalarVector & aProjPressGrad,
                                      const Plato::ScalarVector & aControls,
                                      const Plato::Scalar & aTimeStep) const
    {
        // Workset config
        using ConfigScalar = typename GlobalJacobianC::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Configuration Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename GlobalJacobianC::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tCurrentGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename GlobalJacobianC::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename GlobalJacobianC::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tCurrentLocalStateWS("Current Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename GlobalJacobianC::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename GlobalJacobianC::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar>
            tProjPressGradWS("Projected Pressure Gradient Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aProjPressGrad, tProjPressGradWS);

        // Workset control
        using ControlScalar = typename GlobalJacobianC::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar>
            tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // Workset Jacobian wrt current global states
        using JacobianScalar = typename GlobalJacobianC::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar>
            tJacobianWS("Jacobian Current Global State", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt the current global states
        mGlobalJacobianCU->evaluate(tCurrentGlobalStateWS, tPrevGlobalStateWS,
                                    tCurrentLocalStateWS, tPrevLocalStateWS,
                                    tProjPressGradWS, tControlWS, tConfigWS,
                                    tJacobianWS, aTimeStep);

        return (tJacobianWS);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian of residual with respect to previous global states
     * \param [in] aCurrentGlobalState current global state
     * \param [in] aPrevGlobalState    previous global state
     * \param [in] aCurrentLocalState  current local state
     * \param [in] aPrevLocalState     previous local state
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control, i.e. design, variables
     * \param [in] aTimeStep           current time step
     * \return Jacobian of residual with respect to previous global states for each cell
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename GlobalJacobianP::ResultScalarType>
    jacobianPreviousGlobalStateWorkset(const Plato::ScalarVector & aCurrentGlobalState,
                                       const Plato::ScalarVector & aPrevGlobalState,
                                       const Plato::ScalarVector & aCurrentLocalState,
                                       const Plato::ScalarVector & aPrevLocalState,
                                       const Plato::ScalarVector & aProjPressGrad,
                                       const Plato::ScalarVector & aControls,
                                       const Plato::Scalar & aTimeStep) const
    {
        // Workset config
        using ConfigScalar = typename GlobalJacobianP::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Configuration Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename GlobalJacobianP::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tCurrentGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename GlobalJacobianP::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename GlobalJacobianP::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tCurrentLocalStateWS("Current Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename GlobalJacobianP::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename GlobalJacobianP::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar>
            tProjPressGradWS("Projected Pressure Gradient Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aProjPressGrad, tProjPressGradWS);

        // Workset control
        using ControlScalar = typename GlobalJacobianP::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar>
            tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // Workset Jacobian wrt current global states
        using JacobianScalar = typename GlobalJacobianP::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar>
            tJacobianWS("Jacobian Previous Global State", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt the previous global states
        mGlobalJacobianPU->evaluate(tCurrentGlobalStateWS, tPrevGlobalStateWS,
                                    tCurrentLocalStateWS, tPrevLocalStateWS,
                                    tProjPressGradWS, tControlWS, tConfigWS,
                                    tJacobianWS, aTimeStep);

        return (tJacobianWS);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian of residual with respect to current local states
     * \param [in] aCurrentGlobalState current global state
     * \param [in] aPrevGlobalState    previous global state
     * \param [in] aCurrentLocalState  current local state
     * \param [in] aPrevLocalState     previous local state
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control, i.e. design, variables
     * \param [in] aTimeStep           current time step
     * \return Jacobian of residual with respect to current local states for each cell
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename LocalJacobianC::ResultScalarType>
    jacobianCurrentLocalStateWorkset(const Plato::ScalarVector & aCurrentGlobalState,
                                     const Plato::ScalarVector & aPrevGlobalState,
                                     const Plato::ScalarVector & aCurrentLocalState,
                                     const Plato::ScalarVector & aPrevLocalState,
                                     const Plato::ScalarVector & aProjPressGrad,
                                     const Plato::ScalarVector & aControls,
                                     const Plato::Scalar & aTimeStep) const
    {
        // Workset config
        using ConfigScalar = typename LocalJacobianC::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Configuration Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename LocalJacobianC::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tCurrentGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename LocalJacobianC::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename LocalJacobianC::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tCurrentLocalStateWS("Current Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename LocalJacobianC::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename LocalJacobianC::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar>
            tProjPressGradWS("Projected Pressure Gradient Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aProjPressGrad, tProjPressGradWS);

        // Workset control
        using ControlScalar = typename LocalJacobianC::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar>
            tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // Workset Jacobian wrt current local states
        using JacobianScalar = typename LocalJacobianC::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar> tJacobianWS("Jacobian Local State Workset", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt the current local states
        mGlobalJacobianCC->evaluate(tCurrentGlobalStateWS, tPrevGlobalStateWS,
                                    tCurrentLocalStateWS, tPrevLocalStateWS,
                                    tProjPressGradWS, tControlWS, tConfigWS,
                                    tJacobianWS, aTimeStep);

        return (tJacobianWS);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian of residual with respect to previous local states
     * \param [in] aCurrentGlobalState current global state
     * \param [in] aPrevGlobalState    previous global state
     * \param [in] aCurrentLocalState  current local state
     * \param [in] aPrevLocalState     previous local state
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control, i.e. design, variables
     * \param [in] aTimeStep           current time step
     * \return Jacobian of residual with respect to previous local states for each cell
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename LocalJacobianP::ResultScalarType>
    jacobianPreviousLocalStateWorkset(const Plato::ScalarVector & aCurrentGlobalState,
                                      const Plato::ScalarVector & aPrevGlobalState,
                                      const Plato::ScalarVector & aCurrentLocalState,
                                      const Plato::ScalarVector & aPrevLocalState,
                                      const Plato::ScalarVector & aProjPressGrad,
                                      const Plato::ScalarVector & aControls,
                                      const Plato::Scalar & aTimeStep) const
    {
        // Workset config
        using ConfigScalar = typename LocalJacobianP::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Configuration Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename LocalJacobianP::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tCurrentGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentGlobalState, tCurrentGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename LocalJacobianP::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename LocalJacobianP::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tCurrentLocalStateWS("Current Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename LocalJacobianP::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename LocalJacobianP::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar>
            tProjPressGradWS("Projected Pressure Gradient Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aProjPressGrad, tProjPressGradWS);

        // Workset control
        using ControlScalar = typename LocalJacobianP::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar>
            tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // Workset Jacobian wrt previous local states
        using JacobianScalar = typename LocalJacobianP::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar>
            tJacobianWS("Jacobian Previous Local State Workset", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt the previous local states
        mGlobalJacobianPC->evaluate(tCurrentGlobalStateWS, tPrevGlobalStateWS,
                                    tCurrentLocalStateWS, tPrevLocalStateWS,
                                    tProjPressGradWS, tControlWS, tConfigWS,
                                    tJacobianWS, aTimeStep);

        return (tJacobianWS);
    }

    /***************************************************************************//**
     * \brief Evaluate Jacobian of residual with respect to projected pressure gradient
     * \param [in] aCurrentGlobalState current global state
     * \param [in] aPrevGlobalState    previous global state
     * \param [in] aCurrentLocalState  current local state
     * \param [in] aPrevLocalState     previous local state
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control, i.e. design, variables
     * \param [in] aTimeStep           current time step
     * \return Jacobian of residual with respect to projected pressure gradient for each cell
    *******************************************************************************/
    Plato::ScalarMultiVectorT<typename JacobianPgrad::ResultScalarType>
    jacobianProjPressGradWorkset(const Plato::ScalarVector & aCurrentGlobalState,
                                 const Plato::ScalarVector & aPrevGlobalState,
                                 const Plato::ScalarVector & aCurrentLocalState,
                                 const Plato::ScalarVector & aPrevLocalState,
                                 const Plato::ScalarVector & aProjPressGrad,
                                 const Plato::ScalarVector & aControls,
                                 const Plato::Scalar & aTimeStep) const
    {
        // Workset config
        using ConfigScalar = typename JacobianPgrad::ConfigScalarType;
        Plato::ScalarArray3DT<ConfigScalar>
            tConfigWS("Configuration Workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        mWorksetBase.worksetConfig(tConfigWS);

        // Workset current global state
        using GlobalStateScalar = typename JacobianPgrad::StateScalarType;
        Plato::ScalarMultiVectorT<GlobalStateScalar>
            tCurrentGlobalStateWS("Current Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aCurrentLocalState, tCurrentGlobalStateWS);

        // Workset previous global state
        using PrevGlobalStateScalar = typename JacobianPgrad::PrevStateScalarType;
        Plato::ScalarMultiVectorT<PrevGlobalStateScalar>
            tPrevGlobalStateWS("Previous Global State Workset", mNumCells, mNumGlobalDofsPerCell);
        mWorksetBase.worksetState(aPrevGlobalState, tPrevGlobalStateWS);

        // Workset local state
        using LocalStateScalar = typename JacobianPgrad::LocalStateScalarType;
        Plato::ScalarMultiVectorT<LocalStateScalar>
            tCurrentLocalStateWS("Current Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aCurrentLocalState, tCurrentLocalStateWS);

        // Workset previous local state
        using PrevLocalStateScalar = typename JacobianPgrad::PrevLocalStateScalarType;
        Plato::ScalarMultiVectorT<PrevLocalStateScalar>
            tPrevLocalStateWS("Previous Local State Workset", mNumCells, mNumLocalDofsPerCell);
        mWorksetBase.worksetLocalState(aPrevLocalState, tPrevLocalStateWS);

        // Workset node state, i.e. projected pressure gradient
        using NodeStateScalar = typename JacobianPgrad::NodeStateScalarType;
        Plato::ScalarMultiVectorT<NodeStateScalar>
            tProjPressGradWS("Projected Pressure Gradient Workset", mNumCells, mNumNodeStatePerCell);
        mWorksetBase.worksetNodeState(aProjPressGrad, tProjPressGradWS);

        // Workset control
        using ControlScalar = typename JacobianPgrad::ControlScalarType;
        Plato::ScalarMultiVectorT<ControlScalar>
            tControlWS("Control Workset", mNumCells, mNumNodesPerCell);
        mWorksetBase.worksetControl(aControls, tControlWS);

        // create return view
        using JacobianScalar = typename JacobianPgrad::ResultScalarType;
        Plato::ScalarMultiVectorT<JacobianScalar>
            tJacobianWS("Jacobian Projected Pressure Gradient Workset", mNumCells, mNumGlobalDofsPerCell);

        // Call evaluate function - compute Jacobian wrt pressure gradient
        mGlobalJacProjPressGrad->evaluate(tCurrentGlobalStateWS, tPrevGlobalStateWS,
                                          tCurrentLocalStateWS, tPrevLocalStateWS,
                                          tProjPressGradWS, tControlWS, tConfigWS,
                                          tJacobianWS, aTimeStep);

        return (tJacobianWS);
    }

    /***********************************************************************//**
     * \brief Assemble Jacobian with respect to projected pressure gradient
     *
     * \tparam AViewType POD type for 2-D Kokkos::View
     *
     * \param [in] aJacobianWS Jacobian worset
     * \return Assembled Jacobian
    ***************************************************************************/
    template<typename AViewType>
    Teuchos::RCP<Plato::CrsMatrixType>
    assembleJacobianPressGrad(const Plato::ScalarMultiVectorT<AViewType>& aJacobianWS) const
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
        auto tMesh = mGlobalJacProjPressGrad->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tAssembledJacobian =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumGlobalDofsPerNode, mNumNodeStatePerNode>( &tMesh );

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
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumGlobalDofsPerNode, mNumNodeStatePerNode>
            tJacobianMatEntryOrdinal( tAssembledJacobian, &tMesh );

        // Assemble from the AD-typed result, tJacobian, into the POD-typed global matrix, tJacobianMat.
        //
        // Arguments 1 and 2 below correspond to the size of tJacobian ((Nv x Nd), (Nv x Nn)) and the size of
        // tJacobianMat (Nd, Nn).
        //
        auto tJacobianMatEntries = tAssembledJacobian->entries();
        mWorksetBase.assembleJacobian(
           mNumGlobalDofsPerCell,     // (Nv x Nd)
           mNumNodeStatePerCell,      // (Nv x Nn)
           tJacobianMatEntryOrdinal,  // entry ordinal functor
           aJacobianWS,               // source data
           tJacobianMatEntries        // destination
        );

        return tAssembledJacobian;
    }

    /***********************************************************************//**
     * \brief Assemble transpose of the Jacobian with respect to projected pressure gradient
     *
     * \tparam AViewType POD type for 2-D Kokkos::View
     *
     * \param [in] aJacobianWS Jacobian worset
     * \return Assembled transpose Jacobian
    ***************************************************************************/
    template<typename AViewType>
    Teuchos::RCP<Plato::CrsMatrixType>
    assembleTransposeJacobianPressGrad(const Plato::ScalarMultiVectorT<AViewType>& aJacobianWS) const
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
        auto tMesh = mGlobalJacProjPressGrad->getMesh();
        Teuchos::RCP<Plato::CrsMatrixType> tAssembledTransposeJacobian =
                Plato::CreateBlockMatrix<Plato::CrsMatrixType, mNumNodeStatePerNode, mNumGlobalDofsPerNode>( &tMesh );

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
        Plato::BlockMatrixEntryOrdinal<mNumSpatialDims, mNumNodeStatePerNode, mNumGlobalDofsPerNode>
            tJacobianMatEntryOrdinal( tAssembledTransposeJacobian, &tMesh );

        // Assemble from the AD-typed result, tJacobian, into the POD-typed global matrix, tJacobianMat.
        //
        // The transpose is being assembled, (i.e., tJacobian is transposed before assembly into tJacobianMat), so
        // arguments 1 and 2 below correspond to the size of tJacobian ((Nv x Nd), (Nv x Nn)) and the size of the
        // *transpose* of tJacobianMat (Transpose(Nn, Nd) => (Nd, Nn)).
        //
        auto tJacobianMatEntries = tAssembledTransposeJacobian->entries();
        mWorksetBase.assembleTransposeJacobian(
            mNumGlobalDofsPerCell,
            mNumNodeStatePerCell,
            tJacobianMatEntryOrdinal,
            aJacobianWS,
            tJacobianMatEntries
        );

        return tAssembledTransposeJacobian;
    }

// Public access functions
public:
    /***********************************************************************//**
     * \brief Constructor
     * \param [in] aMesh      mesh data base
     * \param [in] aMeshSets  mesh sets data base
     * \param [in] aDataMap   problem-specific data map
     * \param [in] aParamList Teuchos parameter list with input data
     * \param [in] aFuncType  global vector function type
    ***************************************************************************/
    GlobalVectorFunctionInc(Omega_h::Mesh& aMesh,
                            Omega_h::MeshSets& aMeshSets,
                            Plato::DataMap& aDataMap,
                            Teuchos::ParameterList& aParamList,
                            std::string& aFuncType) :
            mNumNodes(aMesh.nverts()),
            mNumCells(aMesh.nelems()),
            mDataMap(aDataMap),
            mWorksetBase(aMesh)
    {
        typename PhysicsT::FunctionFactory tFunctionFactory;

        mGlobalResidual = tFunctionFactory.template createGlobalVectorFunctionInc<Residual>
                                                           (aMesh, aMeshSets, aDataMap, aParamList, aFuncType);

        mGlobalJacobianCU = tFunctionFactory.template createGlobalVectorFunctionInc<GlobalJacobianC>
                                                            (aMesh, aMeshSets, aDataMap, aParamList, aFuncType);

        mGlobalJacobianPU = tFunctionFactory.template createGlobalVectorFunctionInc<GlobalJacobianP>
                                                             (aMesh, aMeshSets, aDataMap, aParamList, aFuncType);

        mGlobalJacobianCC = tFunctionFactory.template createGlobalVectorFunctionInc<LocalJacobianC>
                                                            (aMesh, aMeshSets, aDataMap, aParamList, aFuncType);

        mGlobalJacobianPC = tFunctionFactory.template createGlobalVectorFunctionInc<LocalJacobianP>
                                                             (aMesh, aMeshSets, aDataMap, aParamList, aFuncType);

        mGlobalJacobianZ = tFunctionFactory.template createGlobalVectorFunctionInc<GradientZ>
                                                            (aMesh, aMeshSets, aDataMap, aParamList, aFuncType);

        mGlobalJacobianX = tFunctionFactory.template createGlobalVectorFunctionInc<GradientX>
                                                            (aMesh, aMeshSets, aDataMap, aParamList, aFuncType);

        mGlobalJacProjPressGrad = tFunctionFactory.template createGlobalVectorFunctionInc<JacobianPgrad>
                                                           (aMesh, aMeshSets, aDataMap, aParamList, aFuncType);
    }

    /***********************************************************************//**
     * \brief Destructor
    ***************************************************************************/
    ~GlobalVectorFunctionInc(){ return; }

    /***************************************************************************//**
     * \brief Return reference to Omega_h mesh database
     * \return Omega_h mesh database
    *******************************************************************************/
    Omega_h::Mesh& getMesh() const
    {
        return (mGlobalResidual->getMesh());
    }

    /***********************************************************************//**
     * \brief Append interface for the evaluation of the global vector function residual.
     * \param [in] aInput] global vector function interface
    ***************************************************************************/
    void appendResidual(const std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<Residual>>& aInput)
    {
        mGlobalResidual = aInput;
    }

    /***********************************************************************//**
     * \brief Append interface for the evaluation of the global vector function
     *  Jacobian with respect to projected pressure gradient.
     * \param [in] aInput global vector function interface
    ***************************************************************************/
    void appendPartialProjPressGrad(const std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<JacobianPgrad>>& aInput)
    {
        mGlobalJacProjPressGrad = aInput;
    }

    /***********************************************************************//**
     * \brief Append interface for the evaluation of the global vector function
     *  Jacobian with respect to current local states.
     * \param [in] aInput global vector function interface
    ***************************************************************************/
    void appendPartialCurrentLocalState(const std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<LocalJacobianC>>& aInput)
    {
        mGlobalJacobianCC = aInput;
    }

    /***********************************************************************//**
     * \brief Append interface for the evaluation of the global vector function
     *  Jacobian with respect to previous local states.
     * \param [in] aInput global vector function interface
    ***************************************************************************/
    void appendPartialPreviousLocalState(const std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<LocalJacobianP>>& aInput)
    {
        mGlobalJacobianPC = aInput;
    }

    /***********************************************************************//**
     * \brief Append interface for the evaluation of the global vector function
     *  Jacobian with respect to current global states.
     * \param [in] aInput global vector function interface
    ***************************************************************************/
    void appendPartialCurrentGlobalState(const std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<GlobalJacobianC>>& aInput)
    {
        mGlobalJacobianCU = aInput;
    }

    /***********************************************************************//**
     * \brief Append interface for the evaluation of the global vector function
     *  Jacobian with respect to previous global states.
     * \param [in] aInput global vector function interface
    ***************************************************************************/
    void appendPartialPreviousGlobalState(const std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<GlobalJacobianP>>& aInput)
    {
        mGlobalJacobianPU = aInput;
    }

    /***********************************************************************//**
     * \brief Append interface for the evaluation of the global vector function
     *  Jacobian with respect to controls.
     * \param [in] aInput global vector function interface
    ***************************************************************************/
    void appendPartialControl(const std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<GradientZ>>& aInput)
    {
        mGlobalJacobianZ = aInput;
    }

    /***********************************************************************//**
     * \brief Append interface for the evaluation of the global vector function
     *  Jacobian with respect to configuration.
     * \param [in] aInput global vector function interface
    ***************************************************************************/
    void appendPartialConfiguration(const std::shared_ptr<Plato::AbstractGlobalVectorFunctionInc<GradientX>>& aInput)
    {
        mGlobalJacobianX = aInput;
    }

    /***********************************************************************//**
     * \brief Return total number of degrees of freedom
    ***************************************************************************/
    decltype(mNumNodes) size() const
    {
        return mNumNodes * mNumGlobalDofsPerNode;
    }

    /***********************************************************************//**
     * \brief Return total number of nodes
     * \return total number of nodes
    ***************************************************************************/
    decltype(mNumNodes) numNodes() const
    {
        return mNumNodes;
    }

    /***********************************************************************//**
     * \brief Return number of nodes per cell.
     * \return total number of nodes per cell
    ***************************************************************************/
    decltype(mNumNodesPerCell) numNodesPerCell() const
    {
        return mNumNodesPerCell;
    }

    /***********************************************************************//**
     * \brief Return total number of cells
     * \return total number of cells
    ***************************************************************************/
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
     * \brief Return number of global degrees of freedom per node.
     * \return number of global degrees of freedom per node
    ***************************************************************************/
    decltype(mNumGlobalDofsPerNode) numGlobalDofsPerNode() const
    {
        return mNumGlobalDofsPerNode;
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
     * \brief Assemble global vector function residual
     * \param [in] aCurrentGlobalState global state at current time step
     * \param [in] aPrevGlobalState    global state at previous time step
     * \param [in] aCurrentLocalState  local state at current time step
     * \param [in] aPrevLocalState     local state at previous time step
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control parameters
     * \param [in] aTimeStep           current time step
     * \return Assembled residual
    ***************************************************************************/
    Plato::ScalarVector
    value(const Plato::ScalarVector & aCurrentGlobalState,
          const Plato::ScalarVector & aPrevGlobalState,
          const Plato::ScalarVector & aCurrentLocalState,
          const Plato::ScalarVector & aPrevLocalState,
          const Plato::ScalarVector & aProjPressGrad,
          const Plato::ScalarVector & aControls,
          Plato::Scalar aTimeStep = 0.0) const
    {
        const auto tTotalNumDofs = mNumGlobalDofsPerNode * mNumNodes;
        Kokkos::View<Plato::Scalar*, Kokkos::LayoutRight, Plato::MemSpace>
            tAssembledResidual("Assembled Residual", tTotalNumDofs);

        auto tResidualWS = this->residualWorkset(aCurrentGlobalState, aPrevGlobalState,
                                                 aCurrentLocalState, aPrevLocalState,
                                                 aProjPressGrad, aControls, aTimeStep);
        mWorksetBase.assembleResidual( tResidualWS, tAssembledResidual );

        return tAssembledResidual;
    }

    /***********************************************************************//**
     * \brief Evaluate of global Jacobian with respect to controls
     * \param [in] aCurrentGlobalState global state at current time step
     * \param [in] aPrevGlobalState    global state at previous time step
     * \param [in] aCurrentLocalState  local state at current time step
     * \param [in] aPrevLocalState     local state at previous time step
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control parameters
     * \param [in] aTimeStep           current time step
     * \return Workset of global Jacobian with respect to controls
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_z(const Plato::ScalarVector & aCurrentGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aCurrentLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aProjPressGrad,
               const Plato::ScalarVector & aControls,
               Plato::Scalar aTimeStep = 0.0) const
    {
        auto tJacobianWS = this->jacobianControlWorkset(aCurrentGlobalState, aPrevGlobalState,
                                                        aCurrentLocalState, aPrevLocalState,
                                                        aProjPressGrad, aControls, aTimeStep);
        Plato::ScalarArray3D tOutputJacobian("Output Jacobian WRT Control", mNumCells, mNumGlobalDofsPerCell, mNumNodesPerCell);
        Plato::transform_ad_type_to_pod_3Dview<mNumGlobalDofsPerCell, mNumNodesPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /***********************************************************************//**
     * \brief Evaluate global Jacobian with respect to configuration
     * \param [in] aCurrentGlobalState global state at current time step
     * \param [in] aPrevGlobalState    global state at previous time step
     * \param [in] aCurrentLocalState  local state at current time step
     * \param [in] aPrevLocalState     local state at previous time step
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control parameters
     * \param [in] aTimeStep           current time step
     * \return Workset of global Jacobian with respect to configuration
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_x(const Plato::ScalarVector & aCurrentGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aCurrentLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aProjPressGrad,
               const Plato::ScalarVector & aControls,
               Plato::Scalar aTimeStep = 0.0) const
    {
        auto tJacobianWS = this->jacobianConfigurationWorkset(aCurrentGlobalState, aPrevGlobalState,
                                                              aCurrentLocalState, aPrevLocalState,
                                                              aProjPressGrad, aControls, aTimeStep);
        Plato::ScalarArray3D tOutputJacobian("Jacobian WRT Configuration", mNumCells, mNumGlobalDofsPerCell, mNumConfigDofsPerCell);
        Plato::transform_ad_type_to_pod_3Dview<mNumGlobalDofsPerCell, mNumConfigDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /***********************************************************************//**
     * \brief Evaluate global Jacobian with respect to current global states
     * \param [in] aCurrentGlobalState global state at current time step
     * \param [in] aPrevGlobalState    global state at previous time step
     * \param [in] aCurrentLocalState  local state at current time step
     * \param [in] aPrevLocalState     local state at previous time step
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control parameters
     * \param [in] aTimeStep           current time step
     * \return Workset of global Jacobian with respect to current global states
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_u(const Plato::ScalarVector & aCurrentGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aCurrentLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aProjPressGrad,
               const Plato::ScalarVector & aControls,
               Plato::Scalar aTimeStep = 0.0) const
    {
        auto tJacobianWS = this->jacobianCurrentGlobalStateWorkset(aCurrentGlobalState, aPrevGlobalState,
                                                                   aCurrentLocalState, aPrevLocalState,
                                                                   aProjPressGrad, aControls, aTimeStep);
        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Current State", mNumCells, mNumGlobalDofsPerCell, mNumGlobalDofsPerCell);
        Plato::transform_ad_type_to_pod_3Dview<mNumGlobalDofsPerCell, mNumGlobalDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);

        return tOutputJacobian;
    }

    /***********************************************************************//**
     * \brief Evaluate global Jacobian with respect to previous global states
     * \param [in] aCurrentGlobalState global state at current time step
     * \param [in] aPrevGlobalState    global state at previous time step
     * \param [in] aCurrentLocalState  local state at current time step
     * \param [in] aPrevLocalState     local state at previous time step
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control parameters
     * \param [in] aTimeStep           current time step
     * \return Workset of global Jacobian with respect to previous global states
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_up(const Plato::ScalarVector & aCurrentGlobalState,
                const Plato::ScalarVector & aPrevGlobalState,
                const Plato::ScalarVector & aCurrentLocalState,
                const Plato::ScalarVector & aPrevLocalState,
                const Plato::ScalarVector & aProjPressGrad,
                const Plato::ScalarVector & aControls,
                Plato::Scalar aTimeStep = 0.0) const
    {
        auto tJacobianWS = this->jacobianPreviousGlobalStateWorkset(aCurrentGlobalState, aPrevGlobalState,
                                                                    aCurrentLocalState, aPrevLocalState,
                                                                    aProjPressGrad, aControls, aTimeStep);
        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Previous Global State", mNumCells, mNumGlobalDofsPerCell, mNumGlobalDofsPerCell);
        Plato::transform_ad_type_to_pod_3Dview<mNumGlobalDofsPerCell, mNumGlobalDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /***********************************************************************//**
     * \brief Evaluate global Jacobian with respect to current local states
     * \param [in] aCurrentGlobalState global state at current time step
     * \param [in] aPrevGlobalState    global state at previous time step
     * \param [in] aCurrentLocalState  local state at current time step
     * \param [in] aPrevLocalState     local state at previous time step
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control parameters
     * \param [in] aTimeStep           current time step
     * \return Workset of global Jacobian with respect to current local states
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_c(const Plato::ScalarVector & aCurrentGlobalState,
               const Plato::ScalarVector & aPrevGlobalState,
               const Plato::ScalarVector & aCurrentLocalState,
               const Plato::ScalarVector & aPrevLocalState,
               const Plato::ScalarVector & aProjPressGrad,
               const Plato::ScalarVector & aControls,
               Plato::Scalar aTimeStep = 0.0) const
    {
        auto tJacobianWS = this->jacobianCurrentLocalStateWorkset(aCurrentGlobalState, aPrevGlobalState,
                                                                  aCurrentLocalState, aPrevLocalState,
                                                                  aProjPressGrad, aControls, aTimeStep);
        Plato::ScalarArray3D tOutputJacobian("Output Jacobian Current Local State", mNumCells, mNumGlobalDofsPerCell, mNumLocalDofsPerCell);
        Plato::transform_ad_type_to_pod_3Dview<mNumGlobalDofsPerCell, mNumLocalDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /***********************************************************************//**
     * \brief Evaluate global Jacobian with respect to previous local states
     * \param [in] aCurrentGlobalState global state at current time step
     * \param [in] aPrevGlobalState    global state at previous time step
     * \param [in] aCurrentLocalState  local state at current time step
     * \param [in] aPrevLocalState     local state at previous time step
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control parameters
     * \param [in] aTimeStep           current time step
     * \return Workset of global Jacobian with respect to previous local states
    ***************************************************************************/
    Plato::ScalarArray3D
    gradient_cp(const Plato::ScalarVector & aCurrentGlobalState,
                const Plato::ScalarVector & aPrevGlobalState,
                const Plato::ScalarVector & aCurrentLocalState,
                const Plato::ScalarVector & aPrevLocalState,
                const Plato::ScalarVector & aProjPressGrad,
                const Plato::ScalarVector & aControls,
                Plato::Scalar aTimeStep = 0.0) const
    {
        auto tJacobianWS = this->jacobianPreviousLocalStateWorkset(aCurrentGlobalState, aPrevGlobalState,
                                                                   aCurrentLocalState, aPrevLocalState,
                                                                   aProjPressGrad, aControls, aTimeStep);
        Plato::ScalarArray3D tOutputJacobian("Jacobian Previous Local State", mNumCells, mNumGlobalDofsPerCell, mNumLocalDofsPerCell);
        Plato::transform_ad_type_to_pod_3Dview<mNumGlobalDofsPerCell, mNumLocalDofsPerCell>(mNumCells, tJacobianWS, tOutputJacobian);
        return tOutputJacobian;
    }

    /***********************************************************************//**
     * \brief Compute transpose of global Jacobian with respect to projected pressure gradient
     * \param [in] aCurrentGlobalState global state at current time step
     * \param [in] aPrevGlobalState    global state at previous time step
     * \param [in] aCurrentLocalState  local state at current time step
     * \param [in] aPrevLocalState     local state at previous time step
     * \param [in] aProjPressGrad      projected pressure gradient
     * \param [in] aControls           control parameters
     * \param [in] aTimeStep           current time step
     * \return Assembled transpose of global Jacobian with respect to projected pressure gradient
    ***************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_n_T_assembled(const Plato::ScalarVector & aCurrentGlobalState,
                           const Plato::ScalarVector & aPrevGlobalState,
                           const Plato::ScalarVector & aCurrentLocalState,
                           const Plato::ScalarVector & aPrevLocalState,
                           const Plato::ScalarVector & aProjPressGrad,
                           const Plato::ScalarVector & aControls,
                           Plato::Scalar aTimeStep = 0.0) const
    {
        auto tJacobianWS = this->jacobianProjPressGradWorkset(aCurrentGlobalState, aPrevGlobalState,
                                                              aCurrentLocalState, aPrevLocalState,
                                                              aProjPressGrad, aControls, aTimeStep);
        auto tOutput = this->assembleTransposeJacobianPressGrad(tJacobianWS);
        return (tOutput);
    }

    /***************************************************************************//**
     * \brief Update physics-based parameters within a frequency of optimization iterations
     * \param [in] aGlobalStates global states for all time steps
     * \param [in] aLocalStates  local states for all time steps
     * \param [in] aControls     current controls, i.e. design variables
     * \param [in] aTimeStep     current time step increment
    *******************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aGlobalStates,
                       const Plato::ScalarMultiVector & aLocalStates,
                       const Plato::ScalarVector & aControls,
                       Plato::Scalar aTimeStep = 0.0) const
    {
        mGlobalResidual->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeStep);
        mGlobalJacobianX->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeStep);
        mGlobalJacobianZ->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeStep);
        mGlobalJacobianCC->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeStep);
        mGlobalJacobianPC->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeStep);
        mGlobalJacobianCU->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeStep);
        mGlobalJacobianPU->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeStep);
        mGlobalJacProjPressGrad->updateProblem(aGlobalStates, aLocalStates, aControls, aTimeStep);
    }
};
// class GlobalVectorFunctionInc

}
// namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::GlobalVectorFunctionInc<Plato::InfinitesimalStrainPlasticity<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::GlobalVectorFunctionInc<Plato::InfinitesimalStrainPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::GlobalVectorFunctionInc<Plato::InfinitesimalStrainPlasticity<3>>;
#endif


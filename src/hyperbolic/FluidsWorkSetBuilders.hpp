/*
 * FluidsWorkSetBuilders.hpp
 *
 *  Created on: Apr 6, 2021
 */

#pragma once

#include "Assembly.hpp"
#include "SpatialModel.hpp"
#include "ImplicitFunctors.hpp"
#include "hyperbolic/SimplexFluidsFadTypes.hpp"

namespace Plato
{

namespace Fluids
{

/***************************************************************************//**
 * \tparam PhysicsT physics type
 *
 * \struct Evaluation
 *
 * \brief Functionalities in this structure are used to build data work sets.
 * The data types are assigned based on the physics and automatic differentiation
 * (AD) evaluation types used for fluid flow applications.
 ******************************************************************************/
template<typename PhysicsT>
struct WorkSetBuilder
{
private:
    using SimplexPhysicsT = typename PhysicsT::SimplexT; /*!< holds static values used in fluid flow applications solved with simplex elements */

    using ConfigLocalOridnalMap   = Plato::NodeCoordinate<SimplexPhysicsT::mNumSpatialDims>; /*!< short name used for wrapper class holding coordinate information */

    using MassLocalOridnalMap     = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumMassDofsPerNode>; /*!< short name used for wrapper class mapping elements to local mass degrees of freedom  */
    using EnergyLocalOridnalMap   = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumEnergyDofsPerNode>; /*!< short name used for wrapper class mapping elements to local energy degrees of freedom  */
    using MomentumLocalOridnalMap = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumMomentumDofsPerNode>; /*!< short name used for wrapper class mapping elements to local momentum degrees of freedom  */
    using ControlLocalOridnalMap  = Plato::VectorEntryOrdinal<SimplexPhysicsT::mNumSpatialDims, SimplexPhysicsT::mNumControlDofsPerNode>; /*!< short name used for wrapper class mapping elements to local control degrees of freedom  */

    using ConfigFad   = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::ConfigFad; /*!< configuration forward AD type  */
    using ControlFad  = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::ControlFad; /*!< control forward AD type  */
    using MassFad     = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MassFad; /*!< mass forward AD type  */
    using EnergyFad   = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::EnergyFad; /*!< energy forward AD type  */
    using MomentumFad = typename Plato::Fluids::SimplexFadTypes<SimplexPhysicsT>::MomentumFad; /*!< momentum forward AD type  */

public:
    /***************************************************************************//**
     * \fn void buildMomentumWorkSet
     *
     * \brief build momentum field work set of POD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local momentum degrees of freedom
     * \param [in] aInput  one dimensional momentum field, i.e. flatten momentum field
     *
     * \param [in/out] aOutput two dimensional momentum work set
     ******************************************************************************/
    void buildMomentumWorkSet
    (const Plato::SpatialDomain               & aDomain,
     const MomentumLocalOridnalMap            & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumMomentumDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildMomentumWorkSet
     *
     * \brief build momentum field work set of POD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local momentum degrees of freedom
     * \param [in] aInput    one dimensional momentum field, i.e. flatten momentum field
     *
     * \param [in/out] aOutput two dimensional momentum work set
     ******************************************************************************/
    void buildMomentumWorkSet
    (const Plato::OrdinalType                 & aNumCells,
     const MomentumLocalOridnalMap            & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumMomentumDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildMomentumWorkSet
     *
     * \brief build momentum field work set of FAD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local momentum degrees of freedom
     * \param [in] aInput  one dimensional momentum field, i.e. flatten momentum field
     *
     * \param [in/out] aOutput two dimensional momentum work set
     ******************************************************************************/
    void buildMomentumWorkSet
    (const Plato::SpatialDomain             & aDomain,
     const MomentumLocalOridnalMap          & aMap,
     const Plato::ScalarVector              & aInput,
     Plato::ScalarMultiVectorT<MomentumFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumMomentumDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            MomentumFad>
        (aDomain, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildMomentumWorkSet
     *
     * \brief build momentum field work set of FAD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local momentum degrees of freedom
     * \param [in] aInput    one dimensional momentum field, i.e. flatten momentum field
     *
     * \param [in/out] aOutput two dimensional momentum work set
     ******************************************************************************/
    void buildMomentumWorkSet
    (const Plato::OrdinalType               & aNumCells,
     const MomentumLocalOridnalMap          & aMap,
     const Plato::ScalarVector              & aInput,
     Plato::ScalarMultiVectorT<MomentumFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumMomentumDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            MomentumFad>
        (aNumCells, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildEnergyWorkSet
     *
     * \brief build energy field work set of POD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local energy degrees of freedom
     * \param [in] aInput  one dimensional energy field, i.e. flatten energy field
     *
     * \param [in/out] aOutput two dimensional energy work set
     ******************************************************************************/
    void buildEnergyWorkSet
    (const Plato::SpatialDomain               & aDomain,
     const EnergyLocalOridnalMap              & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumEnergyDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildEnergyWorkSet
     *
     * \brief build energy field work set of POD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local energy degrees of freedom
     * \param [in] aInput    one dimensional energy field, i.e. flatten energy field
     *
     * \param [in/out] aOutput two dimensional energy work set
     ******************************************************************************/
    void buildEnergyWorkSet
    (const Plato::OrdinalType                 & aNumCells,
     const EnergyLocalOridnalMap              & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumEnergyDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildEnergyWorkSet
     *
     * \brief build energy field work set of FAD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local energy degrees of freedom
     * \param [in] aInput  one dimensional energy field, i.e. flatten energy field
     *
     * \param [in/out] aOutput two dimensional energy work set
     ******************************************************************************/
    void buildEnergyWorkSet
    (const Plato::SpatialDomain              & aDomain,
     const EnergyLocalOridnalMap             & aMap,
     const Plato::ScalarVector               & aInput,
     Plato::ScalarMultiVectorT<EnergyFad>    & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumEnergyDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            EnergyFad>
        (aDomain, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildEnergyWorkSet
     *
     * \brief build energy field work set of FAD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local energy degrees of freedom
     * \param [in] aInput    one dimensional energy field, i.e. flatten energy field
     *
     * \param [in/out] aOutput two dimensional energy work set
     ******************************************************************************/
    void buildEnergyWorkSet
    (const Plato::OrdinalType                & aNumCells,
     const EnergyLocalOridnalMap             & aMap,
     const Plato::ScalarVector               & aInput,
     Plato::ScalarMultiVectorT<EnergyFad>    & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumEnergyDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            EnergyFad>
        (aNumCells, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildMassWorkSet
     *
     * \brief build mass field work set of POD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local mass degrees of freedom
     * \param [in] aInput  one dimensional mass field, i.e. flatten mass field
     *
     * \param [in/out] aOutput two dimensional mass work set
     ******************************************************************************/
    void buildMassWorkSet
    (const Plato::SpatialDomain               & aDomain,
     const MassLocalOridnalMap                & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumMassDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildMassWorkSet
     *
     * \brief build mass field work set of POD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local mass degrees of freedom
     * \param [in] aInput    one dimensional mass field, i.e. flatten mass field
     *
     * \param [in/out] aOutput two dimensional mass work set
     ******************************************************************************/
    void buildMassWorkSet
    (const Plato::OrdinalType                 & aNumCells,
     const MassLocalOridnalMap                & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumMassDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildMassWorkSet
     *
     * \brief build mass field work set of FAD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local mass degrees of freedom
     * \param [in] aInput  one dimensional mass field, i.e. flatten mass field
     *
     * \param [in/out] aOutput two dimensional mass work set
     ******************************************************************************/
    void buildMassWorkSet
    (const Plato::SpatialDomain         & aDomain,
     const MassLocalOridnalMap          & aMap,
     const Plato::ScalarVector          & aInput,
     Plato::ScalarMultiVectorT<MassFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumMassDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            MassFad>
        (aDomain, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildMassWorkSet
     *
     * \brief build mass field work set of FAD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local mass degrees of freedom
     * \param [in] aInput    one dimensional mass field, i.e. flatten mass field
     *
     * \param [in/out] aOutput two dimensional mass work set
     ******************************************************************************/
    void buildMassWorkSet
    (const Plato::OrdinalType           & aNumCells,
     const MassLocalOridnalMap          & aMap,
     const Plato::ScalarVector          & aInput,
     Plato::ScalarMultiVectorT<MassFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumMassDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            MassFad>
        (aNumCells, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildControlWorkSet
     *
     * \brief build control field work set of POD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local control degrees of freedom
     * \param [in] aInput  one dimensional control field, i.e. flatten control field
     *
     * \param [in/out] aOutput two dimensional control work set
     ******************************************************************************/
    void buildControlWorkSet
    (const Plato::SpatialDomain               & aDomain,
     const ControlLocalOridnalMap             & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumControlDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildControlWorkSet
     *
     * \brief build control field work set of POD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local control degrees of freedom
     * \param [in] aInput    one dimensional control field, i.e. flatten control field
     *
     * \param [in/out] aOutput two dimensional control work set
     ******************************************************************************/
    void buildControlWorkSet
    (const Plato::OrdinalType                 & aNumCells,
     const ControlLocalOridnalMap             & aMap,
     const Plato::ScalarVector                & aInput,
     Plato::ScalarMultiVectorT<Plato::Scalar> & aOutput)
    {
        Plato::workset_state_scalar_scalar<
            SimplexPhysicsT::mNumControlDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildControlWorkSet
     *
     * \brief build control field work set of FAD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local control degrees of freedom
     * \param [in] aInput  one dimensional control field, i.e. flatten control field
     *
     * \param [in/out] aOutput two dimensional control work set
     ******************************************************************************/
    void buildControlWorkSet
    (const Plato::SpatialDomain            & aDomain,
     const ControlLocalOridnalMap          & aMap,
     const Plato::ScalarVector             & aInput,
     Plato::ScalarMultiVectorT<ControlFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumControlDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            ControlFad>
        (aDomain, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildControlWorkSet
     *
     * \brief build control field work set of FAD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local control degrees of freedom
     * \param [in] aInput    one dimensional control field, i.e. flatten control field
     *
     * \param [in/out] aOutput two dimensional control work set
     ******************************************************************************/
    void buildControlWorkSet
    (const Plato::OrdinalType              & aNumCells,
     const ControlLocalOridnalMap          & aMap,
     const Plato::ScalarVector             & aInput,
     Plato::ScalarMultiVectorT<ControlFad> & aOutput)
    {
        Plato::workset_state_scalar_fad<
            SimplexPhysicsT::mNumControlDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell,
            ControlFad>
        (aNumCells, aMap, aInput, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildConfigWorkSet
     *
     * \brief build configuration field work set of POD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local configuration degrees of freedom
     *
     * \param [in/out] aOutput three dimensional configuration work set
     ******************************************************************************/
    void buildConfigWorkSet
    (const Plato::SpatialDomain           & aDomain,
     const ConfigLocalOridnalMap          & aMap,
     Plato::ScalarArray3DT<Plato::Scalar> & aOutput)
    {
        Plato::workset_config_scalar<
            SimplexPhysicsT::mNumConfigDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aDomain, aMap, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildConfigWorkSet
     *
     * \brief build configuration field work set of POD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local configuration degrees of freedom
     *
     * \param [in/out] aOutput three dimensional configuration work set
     ******************************************************************************/
    void buildConfigWorkSet
    (const Plato::OrdinalType             & aNumCells,
     const ConfigLocalOridnalMap          & aMap,
     Plato::ScalarArray3DT<Plato::Scalar> & aOutput)
    {
        Plato::workset_config_scalar<
            SimplexPhysicsT::mNumConfigDofsPerNode,
            SimplexPhysicsT::mNumNodesPerCell>
        (aNumCells, aMap, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildConfigWorkSet
     *
     * \brief build configuration field work set of FAD type.
     *
     * \param [in] aDomain structure with computational domain metadata such as the mesh and entity sets.
     * \param [in] aMap    map from element to local configuration degrees of freedom
     *
     * \param [in/out] aOutput three dimensional configuration work set
     ******************************************************************************/
    void buildConfigWorkSet
    (const Plato::SpatialDomain       & aDomain,
     const ConfigLocalOridnalMap      & aMap,
     Plato::ScalarArray3DT<ConfigFad> & aOutput)
    {
        Plato::workset_config_fad<
            SimplexPhysicsT::mNumSpatialDims,
            SimplexPhysicsT::mNumNodesPerCell,
            SimplexPhysicsT::mNumConfigDofsPerNode,
            ConfigFad>
        (aDomain, aMap, aOutput);
    }

    /***************************************************************************//**
     * \fn void buildConfigWorkSet
     *
     * \brief build configuration field work set of FAD type.
     *
     * \param [in] aNumCells number of cells/elements
     * \param [in] aMap      map from element to local configuration degrees of freedom
     *
     * \param [in/out] aOutput three dimensional configuration work set
     ******************************************************************************/
    void buildConfigWorkSet
    (const Plato::OrdinalType         & aNumCells,
     const ConfigLocalOridnalMap      & aMap,
     Plato::ScalarArray3DT<ConfigFad> & aOutput)
    {
        Plato::workset_config_fad<
            SimplexPhysicsT::mNumSpatialDims,
            SimplexPhysicsT::mNumNodesPerCell,
            SimplexPhysicsT::mNumConfigDofsPerNode,
            ConfigFad>
        (aNumCells, aMap, aOutput);
    }
};
// struct WorkSetBuilder

}
// namespace Fluids

}
// namespace Plato

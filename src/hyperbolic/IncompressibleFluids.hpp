/*
 * IncompressibleFluids.hpp
 *
 *  Created on: Apr 7, 2021
 */

#pragma once

#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/FluidsFunctionFactory.hpp"

namespace Plato
{

/******************************************************************************//**
 * \class MomentumConservation
 *
 * \tparam SpaceDim    spatial dimensions (integer)
 * \tparam NumControls number of control fields (integer) - e.g. number of design materials
 *
 * \brief Defines static parameters used to solve the momentum conservation equation.
 *
 **********************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 Plato::OrdinalType NumControls = 1>
class MomentumConservation : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    typedef Plato::Fluids::FunctionFactory FunctionFactory; /*!< local vector/scalar function factory type */
    using SimplexT = Plato::SimplexFluids<SpaceDim, NumControls>; /*!< local simplex element type */

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexT::mNumMomentumDofsPerNode; /*!< number of momentum degress of freedom per node */
    static constexpr Plato::OrdinalType mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode; /*!< number of momentum degress of freedom per cell */
};
// class MomentumConservation

/******************************************************************************//**
 * \class MassConservation
 *
 * \tparam SpaceDim    spatial dimensions (integer)
 * \tparam NumControls number of control fields (integer) - e.g. number of design materials
 *
 * \brief Defines static parameters used to solve the mass conservation equation.
 *
 **********************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 Plato::OrdinalType NumControls = 1>
class MassConservation : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    typedef Plato::Fluids::FunctionFactory FunctionFactory; /*!< local vector/scalar function factory type */
    using SimplexT = Plato::SimplexFluids<SpaceDim, NumControls>; /*!< local simplex element type */

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexT::mNumMassDofsPerNode; /*!< number of mass degress of freedom per node */
    static constexpr Plato::OrdinalType mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode; /*!< number of mass degress of freedom per cell */
};
// class MassConservation

/******************************************************************************//**
 * \class EnergyConservation
 *
 * \tparam SpaceDim    spatial dimensions (integer)
 * \tparam NumControls number of control fields (integer) - e.g. number of design materials
 *
 * \brief Defines static parameters used to solve the energy conservation equation.
 *
 **********************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 Plato::OrdinalType NumControls = 1>
class EnergyConservation : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    typedef Plato::Fluids::FunctionFactory FunctionFactory; /*!< local vector/scalar function factory type */
    using SimplexT = Plato::SimplexFluids<SpaceDim, NumControls>; /*!< local simplex element type */

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexT::mNumEnergyDofsPerNode; /*!< number of energy degress of freedom per node */
    static constexpr Plato::OrdinalType mNumDofsPerCell = SimplexT::mNumNodesPerCell * mNumDofsPerNode; /*!< number of energy degress of freedom per cell */
};
// class EnergyConservation

/******************************************************************************//**
 * \class IncompressibleFluids
 *
 * \tparam SpaceDim    spatial dimensions (integer)
 * \tparam NumControls number of control fields (integer) - e.g. number of design materials
 *
 * \brief Defines static parameters used to solve incompressible fluid flow problems.
 *
 **********************************************************************************/
template
<Plato::OrdinalType SpaceDim,
 Plato::OrdinalType NumControls = 1>
class IncompressibleFluids : public Plato::SimplexFluids<SpaceDim, NumControls>
{
public:
    static constexpr auto mNumSpatialDims = SpaceDim; /*!< number of spatial dimensions */

    typedef Plato::Fluids::FunctionFactory FunctionFactory; /*!< local vector/scalar function factory type */
    using SimplexT = typename Plato::SimplexFluids<SpaceDim, NumControls>; /*!< local simplex element type */

    using MassPhysicsT     = typename Plato::MassConservation<SpaceDim, NumControls>; /*!< local mass conservation physics type */
    using EnergyPhysicsT   = typename Plato::EnergyConservation<SpaceDim, NumControls>; /*!< local energy conservation physics type */
    using MomentumPhysicsT = typename Plato::MomentumConservation<SpaceDim, NumControls>; /*!< local momentum conservation physics type */
};
// class IncompressibleFluids

}
// namespace Plato

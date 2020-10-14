/*
 * ComputationalFluidDynamicsTest.cpp
 *
 *  Created on: Oct 13, 2020
 */

#include "Simplex.hpp"
#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/***************************************************************************//**
 *  \brief Base class for simplex-based fluid mechanics problems
 *  \tparam SpaceDim    (integer) spatial dimensions
 *  \tparam NumControls (integer) number of design variable fields (default = 1)
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexFluidMechanics: public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;  /*!< number of spatial dimensions */
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per simplex cell */

    static constexpr Plato::OrdinalType mNumControls = NumControls; /*!< number of design variable fields */

    static constexpr Plato::OrdinalType mNumVoigtTerms =                      /*!< number of fluid stress terms */
        (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 3 : (((SpaceDim == 1) ? 1 : 0)));

    static constexpr Plato::OrdinalType mNumMassDofsPerNode     = 1; /*!< number of continuity degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumMassDofsPerCell     = mNumMassDofsPerNode * mNumNodesPerCell; /*!< number of continuity degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumEnergyDofsPerNode   = 1; /*!< number energy degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumEnergyDofsPerCell   = mNumEnergyDofsPerNode * mNumNodesPerCell; /*!< number of energy degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumMomentumDofsPerNode = SpaceDim; /*!< number of momentum degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumMomentumDofsPerCell = mNumMomentumDofsPerNode * mNumNodesPerCell; /*!< number of momentum degrees of freedom per cell */
};
// class SimplexFluidDynamics

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MassConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
    static constexpr auto mNumDofsPerNode  = Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMassDofsPerNode;
    static constexpr auto mNumNodesPerCell = Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMassDofsPerCell;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class EnergyConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
    static constexpr auto mNumDofsPerNode  = Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumEnergyDofsPerNode;
    static constexpr auto mNumNodesPerCell = Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumEnergyDofsPerCell;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MomentumConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
    static constexpr auto mNumDofsPerNode  = Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMomentumDofsPerNode;
    static constexpr auto mNumNodesPerCell = Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMomentumDofsPerCell;
};

namespace Hyperbolic
{

namespace FluidMechanics
{

/***************************************************************************//**
 *  \brief Base class for automatic differentiation types used in fluid problems
 *  \tparam SpaceDim    (integer) spatial dimensions
 *  \tparam SimplexPhysicsT simplex fluid dynamic physics type
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct EvaluationTypes
{
    static constexpr Plato::OrdinalType mNumControls = SimplexPhysicsT::mNumControls; /*!< number of design variable fields */
    static constexpr Plato::OrdinalType mNumSpatialDims = SimplexPhysicsT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr Plato::OrdinalType mNumNodesPerCell = SimplexPhysicsT::mNumNodesPerCell; /*!< number of nodes per simplex cell */
};

template <typename SimplexPhysicsT>
struct ResidualTypes : EvaluationTypes<SimplexPhysicsT>
{
    using ControlScalarType  = Plato::Scalar;
    using ConfigScalarType   = Plato::Scalar;
    using MassScalarType     = Plato::Scalar;
    using EnergyScalarType   = Plato::Scalar;
    using MomentumScalarType = Plato::Scalar;
    using ResultScalarType   = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradientMomentumTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = Plato::Scalar;
  using MassScalarType     = Plato::Scalar;
  using EnergyScalarType   = Plato::Scalar;
  using MomentumScalarType = SFadType;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientEnergyTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = Plato::Scalar;
  using MassScalarType     = Plato::Scalar;
  using EnergyScalarType   = SFadType;
  using MomentumScalarType = Plato::Scalar;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientMassTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = Plato::Scalar;
  using MassScalarType     = SFadType;
  using EnergyScalarType   = Plato::Scalar;
  using MomentumScalarType = Plato::Scalar;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientConfigTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ConfigFad;

  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = SFadType;
  using MassScalarType     = Plato::Scalar;
  using EnergyScalarType   = Plato::Scalar;
  using MomentumScalarType = Plato::Scalar;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientControlTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ControlFad;

  using ControlScalarType  = SFadType;
  using ConfigScalarType   = Plato::Scalar;
  using MassScalarType     = Plato::Scalar;
  using EnergyScalarType   = Plato::Scalar;
  using MomentumScalarType = Plato::Scalar;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct Evaluation
{
   using Residual         = ResidualTypes<SimplexPhysicsT>;
   using GradientMomentum = GradientMomentumTypes<SimplexPhysicsT>;
   using GradientEnergy   = GradientEnergyTypes<SimplexPhysicsT>;
   using GradientMass     = GradientMassTypes<SimplexPhysicsT>;
   using GradientControl  = GradientConfigTypes<SimplexPhysicsT>;
   using GradientConfig   = GradientControlTypes<SimplexPhysicsT>;
};

}
// namespace FluidMechanics

}
// namespace Hyperbolic

}
//namespace Plato

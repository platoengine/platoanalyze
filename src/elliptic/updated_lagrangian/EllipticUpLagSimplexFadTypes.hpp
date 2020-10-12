#pragma once

#include <Sacado.hpp>

#include "SimplexFadTypes.hpp"

namespace Plato
{

namespace Elliptic
{

namespace UpdatedLagrangian
{

template <typename SimplexPhysicsT>
struct EvaluationTypes
{
    static constexpr int NumNodesPerCell = SimplexPhysicsT::mNumNodesPerCell;
    static constexpr int NumControls     = SimplexPhysicsT::mNumControl;
    static constexpr int SpatialDim      = SimplexPhysicsT::mNumSpatialDims;
};

template <typename SimplexPhysicsT>
struct ResidualTypes : EvaluationTypes<SimplexPhysicsT>
{
  using GlobalStateScalarType = Plato::Scalar;
  using LocalStateScalarType  = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct JacobianTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using GlobalStateScalarType = SFadType;
  using LocalStateScalarType  = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientCTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::LocalStateFad;

  using GlobalStateScalarType = Plato::Scalar;
  using LocalStateScalarType  = SFadType;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientXTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ConfigFad;

  using GlobalStateScalarType = Plato::Scalar;
  using LocalStateScalarType  = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = SFadType;
  using ResultScalarType      = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientZTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ControlFad;

  using GlobalStateScalarType = Plato::Scalar;
  using LocalStateScalarType  = Plato::Scalar;
  using ControlScalarType     = SFadType;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename SimplexPhysicsT>
struct Evaluation {
   using Residual  = ResidualTypes<SimplexPhysicsT>;
   using Jacobian  = JacobianTypes<SimplexPhysicsT>;
   using GradientC = GradientCTypes<SimplexPhysicsT>;
   using GradientZ = GradientZTypes<SimplexPhysicsT>;
   using GradientX = GradientXTypes<SimplexPhysicsT>;
};

} // namespace UpdatedLagrangian

} // namespace Elliptic

} // namespace Plato

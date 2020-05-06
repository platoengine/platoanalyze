#ifndef HYPERBOLIC_SIMPLEX_FAD_TYPES
#define HYPERBOLIC_SIMPLEX_FAD_TYPES

#include <Sacado.hpp>

#include "../SimplexFadTypes.hpp"

namespace Plato {

namespace Hyperbolic {

template <typename SimplexPhysicsT>
struct EvaluationTypes
{
    static constexpr int NumNodesPerCell = SimplexPhysicsT::mNumNodesPerCell;
    static constexpr int NumControls = SimplexPhysicsT::mNumControl;
    static constexpr int SpatialDim = SimplexPhysicsT::mNumSpatialDims;
};

template <typename SimplexPhysicsT>
struct ResidualTypes : EvaluationTypes<SimplexPhysicsT>
{
  using DisplacementScalarType   = Plato::Scalar;
  using VelocityScalarType       = Plato::Scalar;
  using AccelerationScalarType   = Plato::Scalar;
  using ControlScalarType        = Plato::Scalar;
  using ConfigScalarType         = Plato::Scalar;
  using ResultScalarType         = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradientUTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using DisplacementScalarType   = SFadType;
  using VelocityScalarType       = Plato::Scalar;
  using AccelerationScalarType   = Plato::Scalar;
  using ControlScalarType        = Plato::Scalar;
  using ConfigScalarType         = Plato::Scalar;
  using ResultScalarType         = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientVTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using DisplacementScalarType   = Plato::Scalar;
  using VelocityScalarType       = SFadType;
  using AccelerationScalarType   = Plato::Scalar;
  using ControlScalarType        = Plato::Scalar;
  using ConfigScalarType         = Plato::Scalar;
  using ResultScalarType         = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientATypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using DisplacementScalarType   = Plato::Scalar;
  using VelocityScalarType       = Plato::Scalar;
  using AccelerationScalarType   = SFadType;
  using ControlScalarType        = Plato::Scalar;
  using ConfigScalarType         = Plato::Scalar;
  using ResultScalarType         = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientXTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ConfigFad;

  using DisplacementScalarType   = Plato::Scalar;
  using VelocityScalarType       = Plato::Scalar;
  using AccelerationScalarType   = Plato::Scalar;
  using ControlScalarType        = Plato::Scalar;
  using ConfigScalarType         = SFadType;
  using ResultScalarType         = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientZTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ControlFad;

  using DisplacementScalarType   = Plato::Scalar;
  using VelocityScalarType       = Plato::Scalar;
  using AccelerationScalarType   = Plato::Scalar;
  using ControlScalarType        = SFadType;
  using ConfigScalarType         = Plato::Scalar;
  using ResultScalarType         = SFadType;
};

template <typename SimplexPhysicsT>
struct Evaluation {
   using Residual       = ResidualTypes<SimplexPhysicsT>;
   using GradientU      = GradientUTypes<SimplexPhysicsT>;
   using GradientV      = GradientVTypes<SimplexPhysicsT>;
   using GradientA      = GradientATypes<SimplexPhysicsT>;
   using GradientZ      = GradientZTypes<SimplexPhysicsT>;
   using GradientX      = GradientXTypes<SimplexPhysicsT>;
};

}
  
} // namespace Plato

#endif

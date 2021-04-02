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
  using StateScalarType       = Plato::Scalar;
  using StateDotScalarType    = Plato::Scalar;
  using StateDotDotScalarType = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradientUTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using StateScalarType       = SFadType;
  using StateDotScalarType    = Plato::Scalar;
  using StateDotDotScalarType = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientVTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using StateScalarType       = Plato::Scalar;
  using StateDotScalarType    = SFadType;
  using StateDotDotScalarType = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientATypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using StateScalarType       = Plato::Scalar;
  using StateDotScalarType    = Plato::Scalar;
  using StateDotDotScalarType = SFadType;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientXTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ConfigFad;

  using StateScalarType       = Plato::Scalar;
  using StateDotScalarType    = Plato::Scalar;
  using StateDotDotScalarType = Plato::Scalar;
  using ControlScalarType     = Plato::Scalar;
  using ConfigScalarType      = SFadType;
  using ResultScalarType      = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientZTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ControlFad;

  using StateScalarType       = Plato::Scalar;
  using StateDotScalarType    = Plato::Scalar;
  using StateDotDotScalarType = Plato::Scalar;
  using ControlScalarType     = SFadType;
  using ConfigScalarType      = Plato::Scalar;
  using ResultScalarType      = SFadType;
};

template <typename SimplexPhysicsT>
struct Evaluation {
   using Residual  = ResidualTypes<SimplexPhysicsT>;
   using GradientU = GradientUTypes<SimplexPhysicsT>;
   using GradientV = GradientVTypes<SimplexPhysicsT>;
   using GradientA = GradientATypes<SimplexPhysicsT>;
   using GradientX = GradientXTypes<SimplexPhysicsT>;
   using GradientZ = GradientZTypes<SimplexPhysicsT>;
};

} // namespace Hyperbolic
  
} // namespace Plato

#endif

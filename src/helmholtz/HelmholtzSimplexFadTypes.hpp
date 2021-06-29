#ifndef HELMHOLTZ_SIMPLEX_FAD_TYPES
#define HELMHOLTZ_SIMPLEX_FAD_TYPES

#include <Sacado.hpp>

#include "SimplexFadTypes.hpp"

namespace Plato
{

namespace Helmholtz
{

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
  using StateScalarType   = Plato::Scalar;
  using ControlScalarType = Plato::Scalar;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct JacobianTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using StateScalarType   = SFadType;
  using ControlScalarType = Plato::Scalar;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientZTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ControlFad;

  using StateScalarType   = Plato::Scalar;
  using ControlScalarType = SFadType;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = SFadType;
};

template <typename SimplexPhysicsT>
struct Evaluation {
   using Residual  = ResidualTypes<SimplexPhysicsT>;
   using Jacobian  = JacobianTypes<SimplexPhysicsT>;
   using GradientZ = GradientZTypes<SimplexPhysicsT>;
};

} // namespace Helmholtz

} // namespace Plato

#endif

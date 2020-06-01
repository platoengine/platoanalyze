#pragma once

#include <Sacado.hpp>


namespace Plato
{

namespace Geometric
{

template<typename SimplexPhysics>
struct SimplexFadTypes {

  using ControlFad   = Sacado::Fad::SFad<Plato::Scalar,
                                         SimplexPhysics::mNumNodesPerCell>;
  using ConfigFad    = Sacado::Fad::SFad<Plato::Scalar,
                                         SimplexPhysics::mNumSpatialDims*
                                         SimplexPhysics::mNumNodesPerCell>;
};


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
  using ControlScalarType = Plato::Scalar;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradientXTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ConfigFad;

  using ControlScalarType = Plato::Scalar;
  using ConfigScalarType  = SFadType;
  using ResultScalarType  = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientZTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ControlFad;

  using ControlScalarType = SFadType;
  using ConfigScalarType  = Plato::Scalar;
  using ResultScalarType  = SFadType;
};

template <typename SimplexPhysicsT>
struct Evaluation {
   using Residual  = ResidualTypes<SimplexPhysicsT>;
   using GradientZ = GradientZTypes<SimplexPhysicsT>;
   using GradientX = GradientXTypes<SimplexPhysicsT>;
};

} // namespace Geometric

} // namespace Plato

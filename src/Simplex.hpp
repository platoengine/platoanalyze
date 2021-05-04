#pragma once

#include "PlatoStaticsTypes.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for simplex-based mechanics
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class Simplex
{
  public:
    static constexpr Plato::OrdinalType mNumSpatialDims       = SpaceDim;
    static constexpr Plato::OrdinalType mNumNodesPerFace      = SpaceDim;
    static constexpr Plato::OrdinalType mNumNodesPerCell      = SpaceDim+1;
    static constexpr Plato::OrdinalType mNumSpatialDimsOnFace = SpaceDim-1;
};
// class Simplex

} // namespace Plato

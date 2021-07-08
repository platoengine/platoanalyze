#ifndef SIMPLEX_HELMHOLTZ_HPP
#define SIMPLEX_HELMHOLTZ_HPP

#include "../Simplex.hpp"

namespace Plato
{

/******************************************************************************/
/*! Base class for simplex-based helmholtz
*/
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class SimplexHelmholtz : public Plato::Simplex<SpaceDim>
{ 
  public:
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;

    static constexpr Plato::OrdinalType mNumDofsPerNode  = 1;
    static constexpr Plato::OrdinalType mNumDofsPerCell  = mNumDofsPerNode*mNumNodesPerCell;

    static constexpr Plato::OrdinalType mNumControl = 1;

    static constexpr Plato::OrdinalType mNumNodeStatePerNode = 0;

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;

};
// class SimplexHelmholtz

} // namespace Plato

#endif

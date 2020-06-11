/*
 * VoigtMap.hpp
 *
 *  Created on: Jun 11, 2020
 */

namespace Plato {
  template<int SpaceDim> struct VoigtMap {};
  template<>
  struct VoigtMap<1> {
    Plato::OrdinalType I[1];
    Plato::OrdinalType J[1];
    VoigtMap() : I{1}, J{1} {}
  };
  template<>
  struct VoigtMap<2> {
    Plato::OrdinalType I[3];
    Plato::OrdinalType J[3];
    VoigtMap() : I{0, 1, 0}, J{0, 1, 1} {}
  };
  template<>
  struct VoigtMap<3> {
    Plato::OrdinalType I[6];
    Plato::OrdinalType J[6];
    VoigtMap() : I{0, 1, 2, 1, 0, 0}, J{0, 1, 2, 2, 2, 1} {}
  };
}
